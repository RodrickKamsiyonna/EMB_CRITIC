"""Evaluation script for EBMWorldModel (Open-loop and Closed-loop Receding-Horizon Planning)."""

from __future__ import annotations

import json
import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


class EBMWorldModelPolicy(swm.policy.WorldModelPolicy):
    """WorldModelPolicy subclass that buffers strided per-env pixel history for causal EBM MPC.

    In stable_worldmodel==0.1.1, WorldModelPolicy does not maintain a multi-frame observation
    HistoryBuffer (each replan only receives the single latest frame T=1). This subclass records
    observations at macro-step (`action_block`) boundaries so closed-loop EBM planning conditions
    on the real observed trajectory `[e_0, e_1, ..., e_t]` up to `solver.history_size`.
    """

    def set_env(self, env) -> None:
        super().set_env(env)
        n_envs = getattr(env, "num_envs", 1)
        self._ebm_history: list[list[torch.Tensor]] = [[] for _ in range(n_envs)]
        self._ebm_step_count: list[int] = [0 for _ in range(n_envs)]
        self._pending_flush: np.ndarray | None = None

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        needs_flush = info_dict.get("_needs_flush")
        if needs_flush is not None and hasattr(self, "_ebm_history"):
            self._pending_flush = np.asarray(needs_flush, dtype=bool)
            for i in range(len(self._ebm_history)):
                if self._pending_flush[i]:
                    self._ebm_history[i].clear()
                    self._ebm_step_count[i] = 0
        else:
            self._pending_flush = None
        try:
            return super().get_action(info_dict, **kwargs)
        finally:
            self._pending_flush = None

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        out = super()._prepare_info(info_dict)
        # If a newer stable_worldmodel already manages HistoryBuffer, defer to it.
        if (
            getattr(self, "_history_buffer", None) is not None
            or not hasattr(self, "_ebm_history")
            or "pixels" not in out
        ):
            return out

        pixels = out["pixels"]
        if not torch.is_tensor(pixels):
            return out
        latest = pixels[:, -1] if pixels.ndim == 5 else pixels  # (n_envs, C, H, W)
        n_envs = latest.shape[0]

        terminated = out.get("terminated")
        dead = (
            np.asarray(terminated, dtype=bool)
            if terminated is not None
            else np.zeros(n_envs, dtype=bool)
        )

        stride = max(1, int(self.cfg.action_block))
        max_hist = getattr(self.solver, "history_size", None)

        for i in range(n_envs):
            if dead[i]:
                continue
            if self._ebm_step_count[i] % stride == 0:
                hist = self._ebm_history[i]
                hist.append(latest[i].detach().cpu())
                if max_hist is not None and len(hist) > max_hist:
                    if max_hist >= 2:
                        hist = [hist[0]] + hist[-(max_hist - 1) :]
                    else:
                        hist = hist[-max_hist:]
                    self._ebm_history[i] = hist
            self._ebm_step_count[i] += 1

        needs_replan = any(
            (
                len(self._action_buffer[i]) == 0
                or (self._pending_flush is not None and bool(self._pending_flush[i]))
            )
            and not dead[i]
            for i in range(n_envs)
        )
        if not needs_replan:
            return out

        target_len = max(
            (len(self._ebm_history[i]) for i in range(n_envs) if not dead[i]),
            default=1,
        )
        if target_len <= 1:
            return out

        stacked = []
        for i in range(n_envs):
            hist = self._ebm_history[i]
            if not hist:
                frames = [latest[i].detach().cpu()] * target_len
            elif len(hist) < target_len:
                frames = [hist[0]] * (target_len - len(hist)) + hist
            elif len(hist) > target_len:
                frames = [hist[0]] + hist[-(target_len - 1) :] if target_len >= 2 else hist[-1:]
            else:
                frames = hist
            stacked.append(torch.stack(frames, dim=0))
        out["pixels"] = torch.stack(stacked, dim=0)  # (n_envs, target_len, C, H, W)
        return out


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht_ebm_closedloop")
def run(cfg: DictConfig):
    """Run open-loop or closed-loop EBM latent planning evaluation."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon * action_block must be <= eval_budget"

    mode = (
        "open_loop"
        if cfg.plan_config.receding_horizon >= cfg.plan_config.horizon
        else "closed_loop"
    )
    print(
        f"[EBM-Eval] mode={mode}, horizon={cfg.plan_config.horizon}, "
        f"receding_horizon={cfg.plan_config.receding_horizon}, "
        f"action_block={cfg.plan_config.action_block}, eval_budget={cfg.eval.eval_budget}"
    )

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    device = cfg.solver.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    policy_name = cfg.get("policy", "random")
    solver = None

    if policy_name != "random":
        model = swm.wm.utils.load_pretrained(policy_name)
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = EBMWorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), policy_name).parent
        if policy_name != "random"
        else Path(__file__).parent
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = np.sort(
        valid_indices[g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)]
    )

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    solver_summary = {}
    if solver is not None and hasattr(solver, "get_diagnostics_summary"):
        solver_summary = solver.get_diagnostics_summary()
    elif solver is not None and hasattr(solver, "last_metrics"):
        solver_summary = {
            k: v for k, v in solver.last_metrics.items() if k != "plan/energy_curve"
        }

    print("[EBM-Eval] Environment metrics:", metrics)
    if solver_summary:
        per_replan = solver_summary.get("per_replan", [])
        if per_replan:
            print("[EBM-Eval] Per-replan diagnostics:")
            for r in per_replan:
                print(
                    f"  call={r['replan_call']:2d} | H={r['plan_horizon']} | ctx={r['ctx_len']:2d} | "
                    f"envs={r['n_envs_active']:2d} | "
                    f"E: {r['plan/energy_init']:.2f}->{r['plan/energy_final']:.2f} "
                    f"(drop={r['plan/energy_drop']:.2f}, best={r['plan/best_energy']:.2f}) | "
                    f"grad: {r['plan/grad_norm_init']:.3f}->{r['plan/grad_norm_final']:.3f} | "
                    f"z_std={r['plan/opt_z_std']:.3f} z_norm={r['plan/opt_z_norm']:.2f}"
                )
        summary_view = {k: v for k, v in solver_summary.items() if k != "per_replan"}
        print("[EBM-Eval] Planner summary:", json.dumps(summary_view, indent=2))

    out_file = results_path / cfg.output.filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", encoding="utf-8") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"mode: {mode}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"planner_diagnostics: {json.dumps(solver_summary)}\n")
        f.write(f"evaluation_time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    run()
