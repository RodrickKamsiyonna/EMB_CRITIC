"""Open-loop and Closed-loop (Receding Horizon) Energy-Minimization Latent Planner.

Planning procedure:
- Encode observed history frame(s) -> e_obs (fixed) and goal frame -> e_goal (fixed).
- Initialize intermediate horizon tokens z_1, ..., z_{H-1} with random Gaussian noise.
- Form sequence X = [e_obs, z_1, ..., z_{H-1}, e_goal].
- Minimize sequence energy E(X) updating ONLY the random noise tokens z_1, ..., z_{H-1}.
- Select the lowest-energy trajectory across parallel random restarts (num_samples).
- Decode actions for every consecutive pair along [e_curr, z*_1, ..., z*_{H-1}, e_goal]
  using the Inverse Dynamics Model (IDM).
- Open-loop: execute all H planned macro-step action blocks without replanning.
- Closed-loop: accept the first `receding_horizon` action blocks, execute them in the
  world, append the new observed state(s) to the history buffer (retaining past states
  up to `history_size`), and repeat until the full horizon/budget finishes.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from einops import rearrange

from ebm_modules import EBMWorldModel, standardize_tokens


class EBMLatentPlanner:
    """Gradient-based latent trajectory optimizer for EBMWorldModel + IDM."""

    def __init__(
        self,
        model: EBMWorldModel,
        n_steps: int = 50,
        lr: float = 0.05,
        num_samples: int = 16,
        init_noise_std: float = 1.0,
        optimizer_type: str = "adam",  # "adam" | "sgd" | "langevin"
        grad_clip_norm: Optional[float] = 10.0,
        cosine_lr: bool = True,
        langevin_noise_scale: float = 0.05,
        project_std_after_step: bool = False,
        final_denoise_steps: int = 0,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.num_samples = int(num_samples)
        self.init_noise_std = float(init_noise_std)
        self.optimizer_type = optimizer_type.lower()
        self.grad_clip_norm = grad_clip_norm
        self.cosine_lr = bool(cosine_lr)
        self.langevin_noise_scale = float(langevin_noise_scale)
        self.project_std_after_step = bool(project_std_after_step)
        self.final_denoise_steps = int(final_denoise_steps)
        self.seed = seed

    def _make_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self.seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed)
        return gen

    def plan_latents(
        self,
        obs_emb: torch.Tensor,
        goal_emb: torch.Tensor,
        horizon: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Any]:
        """Minimize sequence energy over intermediate noise tokens between obs_emb and goal_emb.

        Args:
            obs_emb: Fixed observed prefix embeddings, shape (B, H_obs, D).
            goal_emb: Fixed goal embedding, shape (B, 1, D) or (B, D).
            horizon: Number of macro-step transitions (frameskip steps each) from
                the latest observed state obs_emb[:, -1] to goal_emb.
                Requires H - 1 intermediate optimizable latent states.

        Returns:
            dict with:
              - actions: (B, horizon, action_dim) predicted by IDM
              - best_traj: (B, horizon + 1, D) = [e_curr, z*_1, ..., z*_{H-1}, e_goal]
              - best_full_seq: (B, H_obs + horizon, D) = [e_obs, z*_1, ..., z*_{H-1}, e_goal]
              - best_energy: (B,)
              - metrics: dict of scalar research diagnostics
        """
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        if goal_emb.ndim == 2:
            goal_emb = goal_emb.unsqueeze(1)

        model_dtype = next(self.model.ebm.parameters()).dtype
        obs_fixed = obs_emb.detach().to(dtype=model_dtype)
        goal_fixed = goal_emb[:, -1:].detach().to(dtype=model_dtype)  # (B, 1, D)
        b, h_obs, d = obs_fixed.shape
        device = obs_fixed.device
        dtype = obs_fixed.dtype

        n_mid = horizon - 1

        # Trivial 1-step case: directly decode IDM(e_curr, e_goal)
        if n_mid == 0:
            full_seq = torch.cat([obs_fixed, goal_fixed], dim=1)  # (B, H_obs + 1, D)
            with torch.no_grad():
                seq_energy, _ = self.model.compute_energy(full_seq)
                best_traj = torch.cat([obs_fixed[:, -1:], goal_fixed], dim=1)  # (B, 2, D)
                actions = self.model.predict_actions(best_traj)  # (B, 1, A)
            return {
                "actions": actions,
                "best_traj": best_traj,
                "best_full_seq": full_seq,
                "best_energy": seq_energy,
                "metrics": {
                    "plan/energy_init": float(seq_energy.mean().item()),
                    "plan/energy_final": float(seq_energy.mean().item()),
                    "plan/energy_drop": 0.0,
                    "plan/best_energy": float(seq_energy.mean().item()),
                    "plan/median_energy": float(seq_energy.mean().item()),
                    "plan/grad_norm_init": 0.0,
                    "plan/grad_norm_final": 0.0,
                    "plan/opt_z_mean": 0.0,
                    "plan/opt_z_std": 0.0,
                    "plan/opt_z_norm": 0.0,
                },
            }

        s = self.num_samples
        gen = generator if generator is not None else self._make_generator(device)

        # Expand fixed prefix and goal suffix across S parallel random restarts
        obs_exp = obs_fixed.unsqueeze(1).expand(b, s, h_obs, d).reshape(b * s, h_obs, d)
        goal_exp = goal_fixed.unsqueeze(1).expand(b, s, 1, d).reshape(b * s, 1, d)

        # Initialize ONLY the intermediate horizon tokens with random Gaussian noise
        z = torch.randn((b * s, n_mid, d), generator=gen, device=device, dtype=dtype) * self.init_noise_std
        z = z.detach().requires_grad_(True)

        # Freeze model weights temporarily so autograd only computes dE/dz
        was_training = self.model.training
        prev_req_grad = [p.requires_grad for p in self.model.parameters()]
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        try:
            if self.optimizer_type == "adam":
                opt = torch.optim.Adam([z], lr=self.lr)
            elif self.optimizer_type in ("sgd", "langevin"):
                opt = None
            else:
                raise ValueError(f"Unsupported optimizer_type='{self.optimizer_type}'")

            energy_init_val = None
            grad_norm_init_val = None
            grad_norm_last_val = 0.0
            energy_curve: List[float] = []

            with torch.enable_grad():
                for step in range(self.n_steps):
                    frac = step / max(1, self.n_steps - 1)
                    step_lr = (
                        self.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
                        if self.cosine_lr
                        else self.lr
                    )
                    if opt is not None:
                        for pg in opt.param_groups:
                            pg["lr"] = max(step_lr, self.lr * 0.05)
                        opt.zero_grad(set_to_none=True)
                    elif z.grad is not None:
                        z.grad.zero_()

                    seq = torch.cat([obs_exp, z, goal_exp], dim=1)  # (B*S, H_obs + H, D)
                    seq_energy, _ = self.model.compute_energy(seq)  # (B*S,)
                    total_energy = seq_energy.sum()
                    total_energy.backward()

                    grad = z.grad
                    grad_norm = grad.detach().float().norm(dim=-1).mean().item()
                    if step == 0:
                        energy_init_val = float(seq_energy.detach().mean().item())
                        grad_norm_init_val = float(grad_norm)
                    grad_norm_last_val = float(grad_norm)
                    energy_curve.append(float(seq_energy.detach().mean().item()))

                    if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_([z], max_norm=self.grad_clip_norm)

                    if opt is not None:
                        opt.step()
                    else:
                        with torch.no_grad():
                            z.sub_(step_lr * z.grad)
                            if self.optimizer_type == "langevin":
                                noise_std = self.langevin_noise_scale * math.sqrt(max(step_lr, 1e-8)) * (1.0 - frac)
                                if noise_std > 0:
                                    z.add_(torch.randn_like(z) * noise_std)

                    if self.project_std_after_step:
                        with torch.no_grad():
                            z.copy_(standardize_tokens(z, eps=self.model.std_eps))

                # Optional direct denoising refinement steps: z <- z - (1/alpha) * grad_z E(X)
                for _ in range(self.final_denoise_steps):
                    if z.grad is not None:
                        z.grad.zero_()
                    seq = torch.cat([obs_exp, z, goal_exp], dim=1)
                    seq_energy, _ = self.model.compute_energy(seq)
                    seq_energy.sum().backward()
                    with torch.no_grad():
                        z.sub_((1.0 / self.model.alpha) * z.grad)

            # Final energy evaluation and best-sample selection
            with torch.no_grad():
                final_seq = torch.cat([obs_exp, z.detach(), goal_exp], dim=1)
                final_energy, _ = self.model.compute_energy(final_seq)  # (B*S,)
                final_energy_bs = final_energy.view(b, s)  # (B, S)
                best_idx = final_energy_bs.argmin(dim=1)  # (B,)
                best_energy = final_energy_bs.gather(1, best_idx[:, None]).squeeze(1)  # (B,)

                z_bs = z.detach().view(b, s, n_mid, d)
                batch_ids = torch.arange(b, device=device)
                best_z = z_bs[batch_ids, best_idx]  # (B, H - 1, D)

                best_full_seq = torch.cat([obs_fixed, best_z, goal_fixed], dim=1)  # (B, H_obs + H, D)
                best_traj = torch.cat([obs_fixed[:, -1:], best_z, goal_fixed], dim=1)  # (B, H + 1, D)

                # Decode actions for all consecutive pairs using IDM
                actions = self.model.predict_actions(best_traj)  # (B, H, A)

                z_f = best_z.float()
                metrics = {
                    "plan/energy_init": float(energy_init_val if energy_init_val is not None else best_energy.mean().item()),
                    "plan/energy_final": float(final_energy.mean().item()),
                    "plan/energy_drop": float(
                        (energy_init_val - final_energy.mean().item())
                        if energy_init_val is not None
                        else 0.0
                    ),
                    "plan/best_energy": float(best_energy.mean().item()),
                    "plan/median_energy": float(final_energy_bs.median(dim=1).values.mean().item()),
                    "plan/grad_norm_init": float(grad_norm_init_val or 0.0),
                    "plan/grad_norm_final": float(grad_norm_last_val),
                    "plan/opt_z_mean": float(z_f.mean().item()),
                    "plan/opt_z_std": float(z_f.std(dim=-1, unbiased=False).mean().item()),
                    "plan/opt_z_norm": float(z_f.norm(dim=-1).mean().item()),
                    "plan/energy_curve": energy_curve,
                }
        finally:
            for p, rg in zip(self.model.parameters(), prev_req_grad):
                p.requires_grad_(rg)
            if was_training:
                self.model.train()

        return {
            "actions": actions,
            "best_traj": best_traj,
            "best_full_seq": best_full_seq,
            "best_energy": best_energy,
            "metrics": metrics,
        }

    def plan_from_pixels(
        self,
        obs_pixels: torch.Tensor,
        goal_pixels: torch.Tensor,
        horizon: int,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, Any]:
        """Encode observed history pixels and goal pixel frame, then plan."""
        if obs_pixels.ndim == 4:
            obs_pixels = obs_pixels.unsqueeze(1)  # (B, 1, C, H, W)
        if goal_pixels.ndim == 4:
            goal_pixels = goal_pixels.unsqueeze(1)  # (B, 1, C, H, W)

        with torch.no_grad():
            _, obs_emb = self.model.encode_pixels(obs_pixels)
            _, goal_emb = self.model.encode_pixels(goal_pixels)

        return self.plan_latents(
            obs_emb=obs_emb,
            goal_emb=goal_emb,
            horizon=horizon,
            generator=generator,
        )

    def validate_inpainting(
        self,
        clean_emb: torch.Tensor,
        true_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Validation probe: given real sequence [e_0, e_1, ..., e_{T-1}], replace middle
        states e_1..e_{T-2} with noise, minimize energy, and compare against ground truth.
        """
        b, t, d = clean_emb.shape
        if t < 3:
            return {}
        model_dtype = next(self.model.ebm.parameters()).dtype
        clean_emb = clean_emb.detach().to(dtype=model_dtype)
        obs_emb = clean_emb[:, :1]
        goal_emb = clean_emb[:, -1:]
        gt_mid = clean_emb[:, 1:-1]

        plan_out = self.plan_latents(obs_emb=obs_emb, goal_emb=goal_emb, horizon=t - 1)
        pred_mid = plan_out["best_traj"][:, 1:-1]

        with torch.no_grad():
            gt_energy, _ = self.model.compute_energy(clean_emb.detach())
            latent_mse = (pred_mid - gt_mid).pow(2).mean().item()
            latent_cos = torch.nn.functional.cosine_similarity(pred_mid, gt_mid, dim=-1).mean().item()

            res = {
                "val_inpaint/latent_mse": float(latent_mse),
                "val_inpaint/latent_cos": float(latent_cos),
                "val_inpaint/gt_energy": float(gt_energy.mean().item()),
                "val_inpaint/plan_best_energy": float(plan_out["best_energy"].mean().item()),
                "val_inpaint/energy_drop": float(plan_out["metrics"]["plan/energy_drop"]),
                "val_inpaint/opt_z_std": float(plan_out["metrics"]["plan/opt_z_std"]),
            }

            if true_actions is not None:
                act_tgt = true_actions[:, :-1].float()
                valid = ~torch.isnan(act_tgt).any(dim=-1)
                if valid.any():
                    act_clean = torch.nan_to_num(act_tgt, nan=0.0)
                    pred_act = plan_out["actions"]
                    res["val_inpaint/action_mse"] = float(
                        torch.nn.functional.mse_loss(pred_act[valid], act_clean[valid]).item()
                    )
                    res["val_inpaint/action_cos"] = float(
                        torch.nn.functional.cosine_similarity(
                            pred_act[valid], act_clean[valid], dim=-1
                        ).mean().item()
                    )
        return res


class EBMPlanSolver:
    """Solver adapter compatible with stable_worldmodel.policy.WorldModelPolicy.

    Supports both:
      - Open-loop planning: set plan_config.receding_horizon == plan_config.horizon
      - Closed-loop MPC: set plan_config.receding_horizon < plan_config.horizon
        (e.g. receding_horizon=1, keeping up to history_size observed frames).
    """

    def __init__(
        self,
        model: EBMWorldModel,
        n_steps: int = 50,
        lr: float = 0.05,
        num_samples: int = 16,
        init_noise_std: float = 1.0,
        optimizer_type: str = "adam",
        grad_clip_norm: Optional[float] = 10.0,
        cosine_lr: bool = True,
        langevin_noise_scale: float = 0.05,
        project_std_after_step: bool = False,
        final_denoise_steps: int = 0,
        history_size: Optional[int] = 64,
        shrink_horizon: bool = False,
        device: str = "cuda",
        seed: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.history_size = int(history_size) if history_size is not None else None
        self.shrink_horizon = bool(shrink_horizon)
        self._steps_taken = 0
        self.planner = EBMLatentPlanner(
            model=model,
            n_steps=n_steps,
            lr=lr,
            num_samples=num_samples,
            init_noise_std=init_noise_std,
            optimizer_type=optimizer_type,
            grad_clip_norm=grad_clip_norm,
            cosine_lr=cosine_lr,
            langevin_noise_scale=langevin_noise_scale,
            project_std_after_step=project_std_after_step,
            final_denoise_steps=final_denoise_steps,
            seed=seed,
        )
        self._config = None
        self.config = None
        self._action_space = None
        self._n_envs: Optional[int] = None
        self._action_dim: Optional[int] = None
        self.last_metrics: Dict[str, Any] = {}

    def configure(self, *, action_space=None, n_envs: int = 1, config=None, **kwargs) -> None:
        """Called by stable_worldmodel.policy.WorldModelPolicy before evaluation."""
        self._action_space = action_space
        self._n_envs = int(n_envs)
        self._config = config
        self.config = config
        self._steps_taken = 0
        if action_space is not None and hasattr(action_space, "shape") and action_space.shape:
            shape = action_space.shape
            single_shape = shape[1:] if len(shape) > 1 else shape
            self._action_dim = int(math.prod(single_shape))
        else:
            self._action_dim = None

    @property
    def n_envs(self) -> Optional[int]:
        """Number of parallel environments being planned for."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        if self._action_dim is None:
            return int(self.model.idm.action_dim)
        action_block = int(getattr(self._config, "action_block", 1)) if self._config is not None else 1
        return self._action_dim * action_block

    @property
    def horizon(self) -> int:
        """Planning horizon length in timesteps."""
        return int(getattr(self._config, "horizon", 5)) if self._config is not None else 5

    def __call__(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Make solver callable, forwarding to solve() (required by WorldModelPolicy)."""
        return self.solve(*args, **kwargs)

    def _normalize_pixels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert 4D/5D/6D pixel tensors into (B, T, C, H, W) on model device."""
        if tensor.ndim == 4:
            tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 6:
            # Drop sample dim if caller pre-expanded (B, S, T, C, H, W)
            tensor = tensor[:, 0]
        elif tensor.ndim != 5:
            raise ValueError(f"Unsupported pixel tensor shape {tuple(tensor.shape)}")
        try:
            target_device = next(self.model.parameters()).device
        except StopIteration:
            target_device = self.device
        return tensor.to(target_device)

    def solve(self, info_dict: Dict[str, Any], init_action: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Solve for a horizon-length sequence of action blocks given current info_dict."""
        base_horizon = self.horizon
        receding = (
            int(getattr(self._config, "receding_horizon", base_horizon))
            if self._config is not None
            else base_horizon
        )

        obs_pixels = self._normalize_pixels(info_dict["pixels"])
        # Keep observed history (up to history_size, e.g. 64 frames for full-episode EBM)
        if self.history_size is not None and obs_pixels.size(1) > self.history_size:
            obs_pixels = obs_pixels[:, -self.history_size :]

        goal_pixels = self._normalize_pixels(info_dict["goal"])[:, -1:]

        plan_horizon = (
            max(1, base_horizon - self._steps_taken)
            if self.shrink_horizon
            else base_horizon
        )

        out = self.planner.plan_from_pixels(
            obs_pixels=obs_pixels,
            goal_pixels=goal_pixels,
            horizon=plan_horizon,
        )
        actions = out["actions"]
        # Pad actions back to base_horizon if caller expects fixed (B, base_horizon, A) shape
        if actions.size(1) < base_horizon:
            pad = actions[:, -1:].expand(-1, base_horizon - actions.size(1), -1)
            actions = torch.cat([actions, pad], dim=1)

        if self.shrink_horizon:
            self._steps_taken += min(receding, plan_horizon)

        self.last_metrics = out["metrics"]
        return {
            "actions": actions.detach().float().cpu(),
            "costs": out["best_energy"].detach().float().cpu().tolist(),
            "best_traj": out["best_traj"].detach(),
            "best_energy": out["best_energy"].detach(),
            "metrics": out["metrics"],
        }


def run_ebm_rollout(
    step_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, bool, Dict[str, Any]]],
    init_obs_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    planner: EBMLatentPlanner,
    total_horizon: int,
    receding_horizon: int = 1,
    history_size: int = 3,
    shrink_horizon: bool = True,
    keep_first_state: bool = True,
) -> Dict[str, Any]:
    """Explicit open-loop or closed-loop (receding-horizon) rollout driver.

    Args:
        step_fn: Callable taking an action block of shape (B, action_block_dim) and returning
            (next_obs_pixels: (B, C, H, W), done: bool, step_info: dict).
            Note: each macro-step applies `frameskip` primitive env steps internally.
        init_obs_pixels: Initial observed frame(s), shape (B, C, H, W) or (B, T0, C, H, W).
        goal_pixels: Target goal frame, shape (B, C, H, W) or (B, 1, C, H, W).
        planner: Configured EBMLatentPlanner.
        total_horizon: Total number of macro-steps to execute.
        receding_horizon: Number of planned macro-steps to execute before replanning.
            - Open-loop: set `receding_horizon >= total_horizon`.
            - Closed-loop: set `receding_horizon < total_horizon` (e.g., 1).
        history_size: Maximum number of observed frames passed as fixed prefix to EBM.
        shrink_horizon: If True, shrink remaining plan horizon to `total_horizon - executed`
            so the goal stays aligned with the end of the episode.
        keep_first_state: If True, when history exceeds `history_size`, retain the first
            observed state plus the most recent `history_size - 1` states.
    """
    if init_obs_pixels.ndim == 4:
        history = [init_obs_pixels]
    elif init_obs_pixels.ndim == 5:
        history = [init_obs_pixels[:, i] for i in range(init_obs_pixels.size(1))]
    else:
        raise ValueError(f"Unexpected init_obs_pixels shape {tuple(init_obs_pixels.shape)}")

    if goal_pixels.ndim == 4:
        goal_pixels = goal_pixels.unsqueeze(1)

    executed_actions: List[torch.Tensor] = []
    plan_logs: List[Dict[str, Any]] = []
    step_infos: List[Dict[str, Any]] = []

    steps_done = 0
    done = False

    while steps_done < total_horizon and not done:
        remaining = total_horizon - steps_done
        plan_h = remaining if shrink_horizon else total_horizon

        if len(history) <= history_size:
            ctx_frames = history
        elif keep_first_state and history_size >= 2:
            ctx_frames = [history[0]] + history[-(history_size - 1) :]
        else:
            ctx_frames = history[-history_size:]

        obs_ctx = torch.stack(ctx_frames, dim=1)  # (B, H_ctx, C, H, W)
        plan_out = planner.plan_from_pixels(
            obs_pixels=obs_ctx,
            goal_pixels=goal_pixels,
            horizon=plan_h,
        )
        plan_logs.append(
            {
                "replan_at_step": steps_done,
                "plan_horizon": plan_h,
                "ctx_len": len(ctx_frames),
                **{k: v for k, v in plan_out["metrics"].items() if k != "plan/energy_curve"},
            }
        )

        n_accept = min(receding_horizon, remaining, plan_out["actions"].size(1))
        for k in range(n_accept):
            act_block = plan_out["actions"][:, k]  # (B, frameskip * action_dim)
            next_obs, done, info = step_fn(act_block)
            executed_actions.append(act_block.detach())
            step_infos.append(info)
            history.append(next_obs)
            steps_done += 1
            if done or steps_done >= total_horizon:
                break

    return {
        "actions": torch.stack(executed_actions, dim=1) if executed_actions else None,
        "obs_history": torch.stack(history, dim=1),
        "plan_logs": plan_logs,
        "step_infos": step_infos,
        "steps_executed": steps_done,
        "done": done,
    }
