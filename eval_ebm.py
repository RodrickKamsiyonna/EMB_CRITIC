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
        policy = swm.policy.WorldModelPolicy(
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

    solver_metrics = {}
    if solver is not None and hasattr(solver, "last_metrics"):
        solver_metrics = {
            k: v for k, v in solver.last_metrics.items() if k != "plan/energy_curve"
        }

    print("[EBM-Eval] Environment metrics:", metrics)
    if solver_metrics:
        print("[EBM-Eval] Last planner diagnostics:", json.dumps(solver_metrics, indent=2))

    out_file = results_path / cfg.output.filename
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", encoding="utf-8") as f:
        f.write("\n==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n==== RESULTS ====\n")
        f.write(f"mode: {mode}\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"planner_diagnostics: {json.dumps(solver_metrics)}\n")
        f.write(f"evaluation_time: {end_time - start_time:.2f} seconds\n")


if __name__ == "__main__":
    run()
