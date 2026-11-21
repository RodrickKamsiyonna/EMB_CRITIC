"""Real end-to-end test of train_ebm.py in full-episode mode (variable-length HDF5 + WandB + auto-resume)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def make_synthetic_variable_pusht_h5(h5_path: Path, n_episodes: int = 24):
    """Create variable-length episodes (35..80 raw steps -> 7..16 frames at frameskip=5)."""
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    ep_lens = rng.integers(35, 81, size=(n_episodes,), dtype=np.int32)
    ep_offsets = np.zeros((n_episodes,), dtype=np.int64)
    ep_offsets[1:] = np.cumsum(ep_lens[:-1])
    total = int(ep_lens.sum())

    pixels = rng.integers(0, 256, size=(total, 64, 64, 3), dtype=np.uint8)
    action = rng.standard_normal((total, 2)).astype(np.float32)
    proprio = rng.standard_normal((total, 4)).astype(np.float32)
    state = rng.standard_normal((total, 5)).astype(np.float32)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=ep_lens)
        f.create_dataset("ep_offset", data=ep_offsets)
        f.create_dataset("pixels", data=pixels)
        f.create_dataset("action", data=action)
        f.create_dataset("proprio", data=proprio)
        f.create_dataset("state", data=state)


def main():
    repo_dir = Path(__file__).resolve().parents[1]
    tmp_root = Path("/tmp/test_ebm_e2e")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    data_root = tmp_root / "data"
    run_home = tmp_root / "runs"
    h5_path = data_root / "datasets" / "pusht_tiny_test.h5"
    make_synthetic_variable_pusht_h5(h5_path, n_episodes=24)

    env = os.environ.copy()
    env["LOCAL_DATASET_DIR"] = str(data_root)
    env["STABLEWM_HOME"] = str(run_home)
    env["HYDRA_FULL_ERROR"] = "1"
    env["WANDB_MODE"] = "offline"
    env["WANDB_DIR"] = str(tmp_root)

    # 24 episodes -> 22 train (5 batches of 4), 2 val
    base_cmd = [
        sys.executable,
        str(repo_dir / "train_ebm.py"),
        "data=pusht",
        "data_mode=episode",
        "max_ep_frames=64",
        f"data.dataset.name={h5_path}",
        "model=ebm_tiny",
        "output_model_name=e2e_test_run",
        "subdir=e2e_test_run",
        "training.auto_resume=true",
        "logging.save_every_n_steps=2",
        "logging.val_inpaint_every_n_epochs=1",
        "logging.val_inpaint_steps=3",
        "logging.val_inpaint_restarts=2",
        "logging.val_inpaint_samples=2",
        "logging.val_probe_samples=4",
        "img_size=64",
        "embed_dim=64",
        "model.encoder.projector_hidden_dim=128",
        "model.ebm.n_layer=2",
        "model.ebm.n_head=2",
        "model.ebm.head_dim=32",
        "model.ebm.mlp_dim=128",
        "model.ebm.energy_mlp_dim=64",
        "model.idm.hidden_dim=64",
        "loss.sigreg.kwargs.num_proj=64",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.precision=32-true",
        "trainer.log_every_n_steps=1",
        "loader.batch_size=4",
        "loader.num_workers=0",
        "loader.persistent_workers=false",
        "loader.prefetch_factor=null",
        "loader.pin_memory=false",
        "wandb.enabled=true",
        "wandb.config.project=emb_critic",
        "wandb.config.name=e2e_episode_test",
    ]

    print("=== Phase 1: Mid-epoch stop (max_steps=2, save_every_n_steps=2) ===")
    subprocess.run(base_cmd + ["trainer.max_epochs=2", "+trainer.max_steps=2"], cwd=repo_dir, env=env, check=True)

    run_dir = run_home / "checkpoints" / "e2e_test_run"
    assert (run_dir / "last.ckpt").exists(), f"Missing last.ckpt in {run_dir}"
    ckpts = list(run_dir.glob("*.ckpt"))
    assert len(ckpts) == 1 and ckpts[0].name == "last.ckpt", f"Expected only last.ckpt, got {ckpts}"
    weight_files = list(run_home.rglob("weights_latest.pt"))
    assert len(weight_files) >= 1, f"Missing weights_latest.pt under {run_home}"
    print("Phase 1 (mid-epoch stop at step 2) OK!")

    print("=== Phase 2a: Auto-resume mid-epoch -> finish epoch 1 ===")
    subprocess.run(base_cmd + ["trainer.max_epochs=1"], cwd=repo_dir, env=env, check=True)
    lines1 = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    ep_lines1 = [r for r in lines1 if "event" not in r]
    assert len(ep_lines1) == 1 and ep_lines1[0]["epoch"] == 1, f"Unexpected epoch 1 records: {ep_lines1}"
    # Verify explicit mse_loss and idm_mse_loss keys are logged!
    for req_key in ["train/mse_loss", "val/mse_loss", "train/idm_mse_loss", "val/idm_mse_loss", "val_probe/effective_rank", "val_inpaint/latent_mse"]:
        assert req_key in ep_lines1[0], f"Missing expected log key '{req_key}' in {sorted(ep_lines1[0].keys())}"
    print("Phase 2a (mid-epoch auto-resume -> epoch 1) OK! global_step:", ep_lines1[0]["global_step"],
          "val/mse_loss:", round(ep_lines1[0]["val/mse_loss"], 4))

    print("=== Phase 2b: Auto-resume across epoch boundary -> finish epoch 2 ===")
    subprocess.run(base_cmd + ["trainer.max_epochs=2"], cwd=repo_dir, env=env, check=True)
    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    epoch_lines = [r for r in lines if "event" not in r]
    assert len(epoch_lines) >= 2 and epoch_lines[-1]["epoch"] == 2, f"Expected epoch 2 after resume, got {epoch_lines}"
    print("Phase 2b (cross-epoch auto-resume -> epoch 2) OK! Epochs logged:", [r["epoch"] for r in epoch_lines])

    print("=== Phase 3: Verify swm.wm.utils.load_pretrained (eval_ebm.py path) ===")
    os.environ["STABLEWM_HOME"] = str(run_home)
    sys.path.insert(0, str(repo_dir))
    import stable_worldmodel as swm

    loaded = swm.wm.utils.load_pretrained("e2e_test_run/weights_latest.pt")
    loaded.eval()
    print("Phase 3 load_pretrained OK! Loaded type:", type(loaded).__name__)
    print("ALL FULL-EPISODE E2E TRAIN + RESUME + WANDB + LOAD TESTS PASSED!")


if __name__ == "__main__":
    main()
