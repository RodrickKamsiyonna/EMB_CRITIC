"""Energy-Based World Model (EBM-WM) with ResNet-18 Encoder and IDM.

Architecture summary:
- Visual encoder: ResNet-18 (from scratch by default) + MLP projector -> (B, T, D)
- Per-token standardization: each clean embedding is centered and divided by its own std.
- Corruption: per-sequence mask ratio r ~ U(0, 1) (at least 1 valid token selected),
  per-sequence gamma = sigmoid(z) with z ~ N(-1.5, 0.8^2),
  x = gamma * e + (1 - gamma) * 2 * eps for selected tokens, clean e for unselected.
- Backbone: Causal RoPE Transformer with internal pre-RMSNorm + final RMSNorm,
  NO input RMSNorm on noisy inputs, NO vocab embedding or LM head.
- Energy head: 2-layer MLP -> D-dim vector -> L2 norm (positive scalar token energy)
  -> sum over valid tokens = sequence energy E(x).
- Denoising loss: e_hat = x - (1 / alpha) * grad_x E(x) (alpha = 1), plain MSE against
  NON-DETACHED clean standardized embeddings over selected tokens and embedding dims.
- Regularization: SIGReg on clean embeddings (from LeWM).
- Inverse Dynamics Model (IDM): predicts frameskip action chunk from detached
  consecutive clean standardized embeddings (e_t, e_{t+1}).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from einops import rearrange


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMSNorm compatible with higher-order autograd (create_graph=True)."""
    orig_dtype = x.dtype
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed.to(orig_dtype)


def standardize_tokens(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Center each token embedding and divide by its own std across feature dim."""
    orig_dtype = x.dtype
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    # Use population std (unbiased=False) + eps inside sqrt for smooth higher-order grads
    std = torch.sqrt(centered.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = centered / std
    return out.to(orig_dtype)


class ResNet18Encoder(nn.Module):
    """ResNet-18 visual backbone (from scratch by default) + projector MLP."""

    def __init__(
        self,
        embed_dim: int = 192,
        projector_hidden_dim: int = 2048,
        pretrained: bool = False,
        projector_norm: str = "batch",  # "batch" (LeWM default) | "layer" | "none"
    ):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim

        if projector_norm == "batch":
            norm_layer: nn.Module = nn.BatchNorm1d(projector_hidden_dim)
        elif projector_norm == "layer":
            norm_layer = nn.LayerNorm(projector_hidden_dim)
        elif projector_norm == "none":
            norm_layer = nn.Identity()
        else:
            raise ValueError(f"Unsupported projector_norm='{projector_norm}'")

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, projector_hidden_dim),
            norm_layer,
            nn.GELU(),
            nn.Linear(projector_hidden_dim, embed_dim),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode frames of shape (B, T, C, H, W) or (N, C, H, W) -> (..., embed_dim)."""
        if pixels.ndim == 5:
            b, t = pixels.shape[:2]
            x = rearrange(pixels.float(), "b t c h w -> (b t) c h w")
            feat = self.backbone(x)
            emb = self.projector(feat)
            return rearrange(emb, "(b t) d -> b t d", b=b, t=t)
        elif pixels.ndim == 4:
            feat = self.backbone(pixels.float())
            return self.projector(feat)
        else:
            raise ValueError(f"Expected 4D or 5D pixels tensor, got shape {tuple(pixels.shape)}")


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings (RoPE) precomputed up to max_seq_len."""

    def __init__(self, head_dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", freqs.sin()[None, :, None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to tensor x of shape (B, T, n_head, head_dim)."""
        seq_len = x.size(1)
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        cos = self.cos_cached[:, :seq_len, :, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :seq_len, :, :].to(dtype=x.dtype, device=x.device)
        d_half = self.head_dim // 2
        x1 = x[..., :d_half]
        x2 = x[..., d_half:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with RoPE and QK-RMSNorm.

    Uses explicit dot-product attention so double-backward (create_graph=True for
    grad_x E(x)) works reliably on both CPU and GPU without FlashAttention limitations.
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        head_dim: int,
        max_seq_len: int = 1024,
        qk_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = head_dim
        self.inner_dim = n_head * head_dim
        self.qk_norm = qk_norm
        self.dropout = dropout
        self.scale = head_dim ** -0.5

        self.c_q = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.c_k = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.c_v = nn.Linear(embed_dim, self.inner_dim, bias=False)
        self.c_proj = nn.Linear(self.inner_dim, embed_dim, bias=False)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask[None, None, :, :], persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.c_q(x).view(b, t, self.n_head, self.head_dim)
        k = self.c_k(x).view(b, t, self.n_head, self.head_dim)
        v = self.c_v(x).view(b, t, self.n_head, self.head_dim)

        q = rope(q)
        k = rope(k)
        if self.qk_norm:
            q = rms_norm(q)
            k = rms_norm(k)

        q = q.transpose(1, 2)  # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention logits in float32 for stable second-order gradients under bf16
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale

        if t <= self.causal_mask.size(-1):
            mask = self.causal_mask[:, :, :t, :t]
        else:
            mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))[None, None, :, :]

        if valid_mask is not None:
            # valid_mask: (B, T) -> prevent attending to padded key tokens
            key_mask = valid_mask[:, None, None, :].to(torch.bool)
            mask = mask & key_mask

        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1).to(v.dtype)
        if self.dropout > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = torch.matmul(attn, v)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(b, t, self.inner_dim)
        return self.c_proj(out)


class FeedForwardMLP(nn.Module):
    """Feed-forward MLP block (ReLU^2 by default; also supports gelu / silu)."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: Optional[int] = None,
        activation: str = "relu2",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = mlp_dim or (4 * embed_dim)
        self.c_fc = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        if self.activation == "relu2":
            x = F.relu(x).square()
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "silu":
            x = F.silu(x)
        else:
            raise ValueError(f"Unsupported activation='{self.activation}'")
        if self.dropout > 0.0 and self.training:
            x = F.dropout(x, p=self.dropout, training=True)
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    """Single causal Transformer block with internal pre-RMSNorm."""

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        head_dim: int,
        mlp_dim: Optional[int] = None,
        max_seq_len: int = 1024,
        qk_norm: bool = True,
        activation: str = "relu2",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            n_head=n_head,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            qk_norm=qk_norm,
            dropout=dropout,
        )
        self.mlp = FeedForwardMLP(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), rope=rope, valid_mask=valid_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class EnergyHead(nn.Module):
    """Two-layer MLP mapping each final hidden state to a D-dim vector; L2 norm gives token energy."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "relu2",
        eps: float = 1e-8,
    ):
        super().__init__()
        h_dim = hidden_dim or (4 * embed_dim)
        self.fc1 = nn.Linear(embed_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, embed_dim)
        self.activation = activation
        self.eps = eps

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (token_energy: (B, T), energy_vec: (B, T, D))."""
        x = self.fc1(h)
        if self.activation == "relu2":
            x = F.relu(x).square()
        elif self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "silu":
            x = F.silu(x)
        else:
            raise ValueError(f"Unsupported activation='{self.activation}'")
        vec = self.fc2(x)
        # Compute L2 norm in float32 with epsilon for smooth second-order derivatives
        vec_f = vec.float()
        token_energy = torch.sqrt(vec_f.pow(2).sum(dim=-1) + self.eps)
        return token_energy, vec


class EBMTransformer(nn.Module):
    """Causal RoPE Transformer EBM operating directly on continuous token embeddings.

    Key design properties:
    - No vocabulary embedding table, no LM head, no auxiliary value-embedding table.
    - No input RMSNorm on noisy inputs entering block 0.
    - Retains internal pre-RMSNorm in every block and final RMSNorm after last block.
    - Two-layer MLP energy head -> D-dim vector -> L2 norm -> sum over tokens = E(x).
    """

    def __init__(
        self,
        embed_dim: int = 192,
        n_layer: int = 6,
        n_head: int = 3,
        head_dim: int = 64,
        mlp_dim: Optional[int] = None,
        energy_mlp_dim: Optional[int] = None,
        max_seq_len: int = 1024,
        qk_norm: bool = True,
        activation: str = "relu2",
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        energy_eps: float = 1e-8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        self.rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_head=n_head,
                    head_dim=head_dim,
                    mlp_dim=mlp_dim,
                    max_seq_len=max_seq_len,
                    qk_norm=qk_norm,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.energy_head = EnergyHead(
            embed_dim=embed_dim,
            hidden_dim=energy_mlp_dim or (4 * embed_dim),
            activation=activation,
            eps=energy_eps,
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Residual scaling on block output projections
        residual_std = 0.02 / math.sqrt(2 * self.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=residual_std)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (sequence_energy: (B,), token_energy: (B, T)) from input embeddings x: (B, T, D)."""
        # Explicitly NO input RMSNorm on noisy inputs x
        h = x
        for block in self.blocks:
            h = block(h, rope=self.rope, valid_mask=valid_mask)
        # Final RMSNorm retained
        h = rms_norm(h)
        token_energy, _ = self.energy_head(h)  # (B, T) in float32

        if valid_mask is not None:
            token_energy = token_energy * valid_mask.to(token_energy.dtype)
        seq_energy = token_energy.sum(dim=-1)  # (B,)
        return seq_energy, token_energy


class InverseDynamicsModel(nn.Module):
    """Predicts the action chunk (frameskip * action_dim) between two consecutive frames."""

    def __init__(
        self,
        embed_dim: int = 192,
        action_dim: int = 10,  # frameskip * raw_action_dim
        hidden_dim: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, emb_t: torch.Tensor, emb_next: torch.Tensor) -> torch.Tensor:
        """Predict action transitioning from emb_t to emb_next.
        emb_t, emb_next: (..., D) -> (..., action_dim)
        """
        pair = torch.cat([emb_t, emb_next], dim=-1)
        return self.net(pair)


class EBMWorldModel(nn.Module):
    """End-to-end EBM World Model combining ResNet-18, causal Transformer EBM, and detached IDM."""

    def __init__(
        self,
        encoder: ResNet18Encoder,
        ebm: EBMTransformer,
        idm: InverseDynamicsModel,
        alpha: float = 1.0,
        logit_normal_mean: float = -1.5,
        logit_normal_std: float = 0.8,
        noise_scale: float = 2.0,
        detach_x_skip: bool = True,
        detach_target: bool = False,
        detach_idm_input: bool = True,
        std_eps: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.ebm = ebm
        self.idm = idm
        self.alpha = float(alpha)
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)
        self.noise_scale = float(noise_scale)
        self.detach_x_skip = bool(detach_x_skip)
        self.detach_target = bool(detach_target)
        self.detach_idm_input = bool(detach_idm_input)
        self.std_eps = float(std_eps)

    def encode_pixels(self, pixels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode pixels -> (raw_emb, standardized_emb), both shape (B, T, D)."""
        raw_emb = self.encoder(pixels)
        std_emb = standardize_tokens(raw_emb, eps=self.std_eps)
        return raw_emb, std_emb

    def encode(self, info: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """LeWM-style encode interface returning raw_emb and standardized emb."""
        raw_emb, std_emb = self.encode_pixels(info["pixels"])
        info["raw_emb"] = raw_emb
        info["emb"] = std_emb
        return info

    def sample_corruption(
        self,
        clean_emb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Corrupt standardized clean embeddings according to the EBM masking & noise schedule.

        - Masking ratio: r ~ U(0, 1) per sequence; each valid token selected independently
          with probability r. If none selected in a sequence, select the first valid token.
        - Noise scale & bias: z ~ N(-1.5, 0.8^2) per sequence, gamma = sigmoid(z),
          x = gamma * e + (1 - gamma) * 2 * eps for selected tokens, clean e otherwise.
        """
        b, t, d = clean_emb.shape
        device = clean_emb.device
        dtype = clean_emb.dtype

        if valid_mask is None:
            valid_mask = torch.ones((b, t), dtype=torch.bool, device=device)
        else:
            valid_mask = valid_mask.to(dtype=torch.bool, device=device)

        # 1. Masking ratio r ~ U(0, 1) per sequence
        mask_ratio = torch.rand((b,), device=device, dtype=torch.float32)
        rand_u = torch.rand((b, t), device=device, dtype=torch.float32)
        selected = (rand_u < mask_ratio[:, None]) & valid_mask

        # Guarantee at least one valid token is selected per sequence (first valid token)
        none_selected = ~selected.any(dim=1)
        if none_selected.any():
            first_valid_idx = valid_mask.float().argmax(dim=1)  # (B,)
            batch_ids = torch.arange(b, device=device)[none_selected]
            selected[batch_ids, first_valid_idx[none_selected]] = True

        # 2. Noise scale gamma = sigmoid(z), z ~ N(mean, std^2), shared across sequence
        z = (
            torch.randn((b,), device=device, dtype=torch.float32) * self.logit_normal_std
            + self.logit_normal_mean
        )
        gamma = torch.sigmoid(z)  # (B,)

        eps = torch.randn((b, t, d), device=device, dtype=torch.float32)
        gamma_exp = gamma[:, None, None].to(dtype)
        noisy = gamma_exp * clean_emb + (1.0 - gamma_exp) * (self.noise_scale * eps.to(dtype))

        x = torch.where(selected[:, :, None], noisy, clean_emb)
        return {
            "x": x,
            "selected": selected,
            "valid_mask": valid_mask,
            "mask_ratio": mask_ratio,
            "gamma": gamma,
            "z_logit": z,
        }

    def compute_energy(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (seq_energy: (B,), token_energy: (B, T))."""
        return self.ebm(x, valid_mask=valid_mask)

    def denoise(
        self,
        x: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        create_graph: Optional[bool] = None,
        detach_x_skip: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute energy and one-step gradient denoising e_hat = sg(x) - (1/alpha) * grad_x E(x).

        x enters E(x) WITHOUT being detached (so x still receives gradients through grad_x E(x)
        when create_graph=True). When detach_x_skip=True (default), only the additive skip term
        x in e_hat = x.detach() - (1/alpha) * grad_x is detached so the MSE passes gradients
        through grad_x E(x) rather than a direct skip path on x.
        """
        if create_graph is None:
            create_graph = self.training and torch.is_grad_enabled()
        if detach_x_skip is None:
            detach_x_skip = self.detach_x_skip

        # Always enable grad locally so validation (under torch.no_grad) can still take grad_x E(x)
        with torch.enable_grad():
            if create_graph:
                x_in = x if x.requires_grad else x.requires_grad_(True)
            else:
                x_in = x.detach().requires_grad_(True)

            seq_energy, token_energy = self.compute_energy(x_in, valid_mask=valid_mask)
            grad_x = torch.autograd.grad(
                outputs=seq_energy.sum(),
                inputs=x_in,
                create_graph=create_graph,
                retain_graph=create_graph,
            )[0]
            x_skip = x_in.detach() if detach_x_skip else x_in
            e_hat = x_skip - (1.0 / self.alpha) * grad_x

        if not create_graph:
            e_hat = e_hat.detach()
            grad_x = grad_x.detach()
            seq_energy = seq_energy.detach()
            token_energy = token_energy.detach()

        return {
            "e_hat": e_hat,
            "grad_x": grad_x,
            "seq_energy": seq_energy,
            "token_energy": token_energy,
        }

    def predict_actions(self, emb_seq: torch.Tensor) -> torch.Tensor:
        """Run IDM on all consecutive state pairs in emb_seq (B, T, D) -> (B, T-1, action_dim)."""
        return self.idm(emb_seq[:, :-1], emb_seq[:, 1:])

    def compute_losses(
        self,
        batch: Dict[str, torch.Tensor],
        sigreg: Optional[nn.Module] = None,
        sigreg_weight: float = 0.09,
        sigreg_target: str = "raw",  # "raw" | "std"
        idm_weight: float = 1.0,
        compute_diagnostics: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute full training/validation losses and research diagnostics."""
        raw_emb, clean_emb = self.encode_pixels(batch["pixels"])  # (B, T, D)
        b, t, d = clean_emb.shape
        valid_mask = batch.get("valid_mask", None)

        # --- 1. Corruption & Denoising EBM loss (NON-DETACHED target by default) ---
        corrupt = self.sample_corruption(clean_emb, valid_mask=valid_mask)
        x = corrupt["x"]
        selected = corrupt["selected"]  # (B, T)
        gamma = corrupt["gamma"]  # (B,)

        is_train_graph = self.training and torch.is_grad_enabled()
        denoise_out = self.denoise(x, valid_mask=corrupt["valid_mask"], create_graph=is_train_graph)
        e_hat = denoise_out["e_hat"]
        grad_x = denoise_out["grad_x"]

        target = clean_emb.detach() if self.detach_target else clean_emb
        sq_err = (e_hat - target).pow(2)  # (B, T, D)
        # Average only over selected tokens and embedding dimensions, no gamma weighting
        denoise_loss = sq_err[selected].mean()

        # --- 2. SIGReg regularizer on clean embeddings ---
        if sigreg is not None and sigreg_weight > 0.0:
            sigreg_in = raw_emb if sigreg_target == "raw" else clean_emb
            sigreg_loss = sigreg(sigreg_in.transpose(0, 1))
        else:
            sigreg_loss = clean_emb.new_tensor(0.0)

        # --- 3. Inverse Dynamics Model loss on detached consecutive embeddings ---
        idm_emb = clean_emb.detach() if self.detach_idm_input else clean_emb
        pred_actions = self.predict_actions(idm_emb)  # (B, T-1, A)

        raw_actions = batch["action"][:, :-1].float()  # (B, T-1, A) transitions t -> t+1
        valid_act = ~torch.isnan(raw_actions).any(dim=-1)  # (B, T-1)
        if valid_mask is not None:
            vm = valid_mask.to(torch.bool)
            valid_act = valid_act & vm[:, :-1] & vm[:, 1:]

        if valid_act.any():
            clean_actions = torch.nan_to_num(raw_actions, nan=0.0)
            idm_loss = F.mse_loss(pred_actions[valid_act], clean_actions[valid_act])
            with torch.no_grad():
                idm_cos = F.cosine_similarity(
                    pred_actions[valid_act].detach(), clean_actions[valid_act], dim=-1
                ).mean()
        else:
            idm_loss = clean_emb.new_tensor(0.0)
            idm_cos = clean_emb.new_tensor(0.0)

        total_loss = denoise_loss + sigreg_weight * sigreg_loss + idm_weight * idm_loss

        out: Dict[str, torch.Tensor] = {
            "loss": total_loss,
            "denoise_loss": denoise_loss,
            "sigreg_loss": sigreg_loss,
            "idm_loss": idm_loss,
            "raw_emb": raw_emb,
            "emb": clean_emb,
            "e_hat": e_hat,
            "pred_actions": pred_actions,
        }

        # --- 4. Research diagnostics (detached, zero grad overhead) ---
        if compute_diagnostics:
            with torch.no_grad():
                out["diag/idm_cos_sim"] = idm_cos

                # Baseline identity MSE: how noisy x is before denoising
                id_sq_err = (x.detach() - clean_emb.detach()).pow(2)
                baseline_mse = id_sq_err[selected].mean()
                out["diag/baseline_identity_mse"] = baseline_mse
                out["diag/denoise_ratio"] = denoise_loss.detach() / (baseline_mse + 1e-8)

                # Gamma-stratified denoising MSE
                tok_mse = sq_err.detach().mean(dim=-1)  # (B, T)
                gamma_bt = gamma[:, None].expand(b, t)
                for name, lo, hi in [
                    ("low_gamma", 0.0, 0.25),
                    ("mid_gamma", 0.25, 0.6),
                    ("high_gamma", 0.6, 1.01),
                ]:
                    m = selected & (gamma_bt >= lo) & (gamma_bt < hi)
                    if m.any():
                        out[f"diag/denoise_mse_{name}"] = tok_mse[m].mean()

                # Corruption schedule stats
                out["diag/mask_ratio_mean"] = corrupt["mask_ratio"].mean()
                out["diag/selected_frac"] = selected.float().mean()
                out["diag/gamma_mean"] = gamma.mean()
                out["diag/gamma_median"] = gamma.median()

                # Energy landscape: noisy vs clean (and grad norms)
                noisy_e = denoise_out["seq_energy"].detach()
                clean_e, _ = self.compute_energy(clean_emb.detach(), valid_mask=corrupt["valid_mask"])
                out["diag/energy_noisy_mean"] = noisy_e.mean()
                out["diag/energy_noisy_std"] = noisy_e.std(unbiased=False)
                out["diag/energy_clean_mean"] = clean_e.mean()
                out["diag/energy_clean_std"] = clean_e.std(unbiased=False)
                out["diag/energy_gap_noisy_minus_clean"] = (noisy_e - clean_e).mean()

                grad_norms = grad_x.detach().float().norm(dim=-1)  # (B, T)
                out["diag/energy_grad_norm_mean"] = grad_norms[selected].mean()
                out["diag/energy_grad_norm_max"] = grad_norms[selected].max()

                # Representation geometry & collapse monitors
                raw_flat = raw_emb.detach().float().reshape(-1, d)
                std_flat = clean_emb.detach().float().reshape(-1, d)
                out["diag/raw_emb_norm_mean"] = raw_flat.norm(dim=-1).mean()
                out["diag/raw_emb_batch_std"] = raw_flat.std(dim=0, unbiased=False).mean()
                out["diag/std_emb_batch_std"] = std_flat.std(dim=0, unbiased=False).mean()

                if t > 1:
                    temp_diff = (clean_emb.detach()[:, 1:] - clean_emb.detach()[:, :-1]).float()
                    out["diag/temporal_diff_norm"] = temp_diff.norm(dim=-1).mean()

                if sigreg is not None and sigreg_target == "raw":
                    out["diag/sigreg_on_std"] = sigreg(clean_emb.detach().transpose(0, 1))

        return out


@torch.no_grad()
def compute_effective_rank(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute effective rank (exp of spectral entropy of covariance matrix) to monitor collapse."""
    x = embeddings.detach().float().reshape(-1, embeddings.shape[-1])
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    cov = (x.T @ x) / (x.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    total = eigvals.sum()
    if total <= eps:
        return embeddings.new_tensor(0.0)
    p = eigvals / total
    entropy = -(p * torch.log(p.clamp(min=eps))).sum()
    return torch.exp(entropy)
