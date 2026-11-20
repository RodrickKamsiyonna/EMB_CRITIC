# EBM World Model (on LeWorldModel)

Action-free Energy-Based World Model built on top of `le-wm` (`stable-worldmodel` + `stable-pretraining`) with a from-scratch **ResNet-18** visual encoder, a **causal RoPE Transformer EBM**, **SIGReg** regularization, and a detached **Inverse Dynamics Model (IDM)** for open-loop and closed-loop latent planning.

## Architecture

1. **Visual Encoder (`ResNet18Encoder` in `ebm_modules.py`)**
   - From-scratch `ResNet-18` (`pretrained=False`) + MLP projector (`512 -> 2048 -> D`, `BatchNorm1d`, `GELU`).
   - Clean token embeddings are centered and divided by their own std across feature dimensions (`standardize_tokens`) before corruption.
2. **Corruption & Denoising Objective (`EBMWorldModel`)**
   - Masking ratio per sequence: $r \sim U(0, 1)$, each valid token selected independently with probability $r$ (if none selected, the first valid token is selected).
   - Noise scale per sequence (shared across selected tokens): $z \sim \mathcal{N}(-1.5, 0.8^2)$, $\gamma = \sigma(z)$ (median $\approx 0.182$), $x = \gamma e + (1 - \gamma)\cdot 2\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. Unselected tokens retain clean standardized embeddings.
   - Noisy inputs enter the Transformer **without** an input RMSNorm; internal block pre-RMSNorm and final RMSNorm are retained.
   - **Causal RoPE Transformer (`EBMTransformer`):** RoPE, QK-RMSNorm, `ReLU²` MLP, causal self-attention, no vocab embedding / LM head / value-embedding table.
     - Default (`model=ebm_tiny`): 6 layers, `embed_dim=192`, 3 heads × `head_dim=64` (~15M total params, matching LeWM's tiny scale for fast single-A100 training).
     - Large preset (`model=ebm_large`): 15 layers, `embed_dim=1664`, 13 heads × `head_dim=128`, `max_seq_len=1024`.
   - **Energy Head:** 2-layer MLP mapping each final hidden state to a $D$-dim vector; its L2 norm gives positive scalar token energy, summed over valid tokens to produce sequence energy $E(x)$.
   - **Denoising MSE:** $\hat{e} = x_{\text{det}} - \frac{1}{\alpha}\nabla_{x_{\text{det}}} E(x_{\text{det}})$ ($\alpha=1$, with noisy input $x$ **detached** from the encoder), plain MSE against **non-detached** clean standardized embeddings averaged only over selected tokens and embedding dimensions (no $\gamma$ weighting).
3. **Regularization & Inverse Dynamics Model**
   - **SIGReg** (from LeWM) applied to clean embeddings (`weight=0.09`).
   - **IDM (`InverseDynamicsModel`):** trained on **detached** consecutive clean standardized embeddings $(e_t, e_{t+1})$ to predict the `frameskip * action_dim` action chunk between frames.

## Planning (`planner.py`)

- **Open-loop (`config/eval/pusht_ebm_openloop.yaml`):** Fix observed frame(s) $e_{\text{obs}}$ at the start and goal embedding $e_{\text{goal}}$ at the end, initialize intermediate horizon states $z_1, \dots, z_{H-1}$ with random noise, minimize $E([e_{\text{obs}}, z_1, \dots, z_{H-1}, e_{\text{goal}}])$ updating only the noise tokens, then decode consecutive state pairs via IDM and apply all actions.
- **Closed-loop (`config/eval/pusht_ebm_closedloop.yaml`):** Same energy minimization, accept the first `receding_horizon` macro-steps, execute them in the world, append the new observed frame(s) (retaining past states up to `history_size`), and replan until the full horizon finishes.

## Commands

```bash
# 1. Install LeWM dependencies (on GPU/A100 runtime)
pip install "stable-worldmodel[train,env]"

# 2. Run unit & integration verification tests (works on CPU too)
python tests/test_ebm_wm.py

# 3. Train on PushT (default LeWM tiny width: 6L, d=192, frameskip=5, T=4)
python train_ebm.py data=pusht

# 3b. Train with full 15L x 1664d preset
python train_ebm.py data=pusht model=ebm_large embed_dim=1664 loader.batch_size=16

# 4. Evaluate Open-Loop & Closed-Loop on PushT
python eval_ebm.py --config-name=pusht_ebm_openloop.yaml policy=ebm_wm
python eval_ebm.py --config-name=pusht_ebm_closedloop.yaml policy=ebm_wm
```

## Research Logs (WandB + `metrics.jsonl`)

Every run logs to WandB (when `wandb.enabled=true`) and persists epoch summaries to `metrics.jsonl`:
- **Losses:** `loss`, `denoise_loss`, `sigreg_loss`, `idm_loss`, `diag/idm_cos_sim`
- **Denoising quality:** `diag/baseline_identity_mse`, `diag/denoise_ratio`, gamma-stratified MSE (`low_gamma`, `mid_gamma`, `high_gamma`)
- **Energy landscape:** `diag/energy_clean_mean`, `diag/energy_noisy_mean`, `diag/energy_gap_noisy_minus_clean`, `diag/energy_grad_norm_mean`, `val_probe/energy_gap_rev_minus_real`, `val_probe/energy_gap_shuf_minus_real`
- **Collapse & geometry:** `val_probe/effective_rank`, `diag/raw_emb_batch_std`, `diag/std_emb_batch_std`, `diag/temporal_diff_norm`
- **Periodic validation inpainting probe:** `val_inpaint/latent_mse`, `val_inpaint/latent_cos`, `val_inpaint/action_mse`, `val_inpaint/energy_drop`
