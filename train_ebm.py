"""Training script for EBM World Model + ResNet-18 + IDM (based on LeWM).

Supports:
- Per-step checkpointing every `logging.save_every_n_steps` steps (default: 1000),
  keeping ONLY the latest checkpoint (`last.ckpt` + `weights_latest.pt`).
- Automatic resume from `run_dir/last.ckpt` (including seamless WandB run resume).
- Exact mid-epoch dataset resume via `ResumableDistributedSampler` + `ResumableDatasetCallback`
  (works identically on single GPU and multi-GPU DDP).
"""

from __future__ import annotations

import json
import os
import uuid
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from ebm_modules import compute_effective_rank
from module import SIGReg
from planner import EBMLatentPlanner
from resumable_sampler import ResumableDistributedSampler
from utils import get_column_normalizer, get_img_preprocessor


def ebm_forward(self, batch, stage, cfg):
    """Forward step for EBMWorldModel: denoising EBM + SIGReg + detached IDM + research logs."""
    output = self.model.compute_losses(
        batch=batch,
        sigreg=self.sigreg,
        sigreg_weight=cfg.loss.sigreg.weight,
        sigreg_target=cfg.loss.sigreg.target,
        idm_weight=cfg.loss.idm.weight,
        compute_diagnostics=cfg.logging.compute_diagnostics,
    )

    log_payload = {}
    for k, v in output.items():
        if torch.is_tensor(v) and v.ndim == 0:
            log_payload[f"{stage}/{k}"] = v.detach()

    on_step = stage == "train"
    self.log_dict(log_payload, on_step=on_step, on_epoch=True, sync_dist=True)

    # Stash a small validation sample for epoch-end research probes (erank, shuffle gap, inpainting)
    if stage == "val" and hasattr(self, "_val_probe_cache"):
        cache = self._val_probe_cache
        if cache["n_batches"] < cfg.logging.val_probe_max_batches:
            cache["embs"].append(output["emb"].detach().cpu())
            cache["actions"].append(batch["action"].detach().cpu())
            cache["n_batches"] += 1

    return output


class ResumableDatasetCallback(Callback):
    """Persists exact mid-epoch batch progress in Lightning checkpoints and restores the sampler.

    MUST be placed BEFORE ModelCheckpoint in `trainer.callbacks` so `batches_seen_in_epoch`
    is incremented for the just-completed batch before the checkpoint is written.
    """

    def __init__(self, train_sampler: ResumableDistributedSampler):
        super().__init__()
        self.train_sampler = train_sampler
        self.batches_seen_in_epoch: int = 0
        self.last_epoch: int = 0
        self.resumed_epoch: int | None = None

    def on_train_epoch_start(self, trainer, pl_module):
        cur_epoch = int(trainer.current_epoch)
        self.train_sampler.set_epoch(cur_epoch)

        if self.resumed_epoch is not None and cur_epoch == self.resumed_epoch:
            # Arm exact mid-epoch batch offset; keep batches_seen_in_epoch accumulating from here
            self.train_sampler.set_resume(cur_epoch, self.batches_seen_in_epoch)
            if trainer.is_global_zero:
                print(
                    f"[EBM-WM] Resuming dataset mid-epoch: epoch={cur_epoch}, "
                    f"skipping {self.batches_seen_in_epoch} already-completed batches/rank."
                )
            self.resumed_epoch = None
        else:
            self.batches_seen_in_epoch = 0
            self.train_sampler.set_resume(None, 0)
        self.last_epoch = cur_epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batches_seen_in_epoch += 1
        self.last_epoch = int(trainer.current_epoch)

    def state_dict(self):
        return {
            "epoch": int(self.last_epoch),
            "batches_seen_in_epoch": int(self.batches_seen_in_epoch),
        }

    def load_state_dict(self, state_dict):
        self.last_epoch = int(state_dict.get("epoch", 0))
        self.batches_seen_in_epoch = int(state_dict.get("batches_seen_in_epoch", 0))
        self.resumed_epoch = self.last_epoch
        self.train_sampler.set_epoch(self.last_epoch)
        self.train_sampler.set_resume(self.resumed_epoch, self.batches_seen_in_epoch)


class LatestStepWeightsCallback(Callback):
    """Saves ONLY the latest inference weights (`weights_latest.pt`) every N train steps."""

    def __init__(self, run_name: str, cfg, save_every_n_steps: int = 1000):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.save_every_n_steps = max(1, int(save_every_n_steps))
        self._last_saved_step: int = -1

    def _save_latest(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_saved_step:
            return
        from stable_worldmodel.wm.utils import save_pretrained

        save_pretrained(
            pl_module.model,
            run_name=self.run_name,
            config=self.cfg,
            filename="weights_latest.pt",
        )
        self._last_saved_step = step

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step > 0 and step % self.save_every_n_steps == 0:
            self._save_latest(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self._save_latest(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        self._save_latest(trainer, pl_module)


class EBMResearchCallback(Callback):
    """Research logging callback: JSONL persistence, effective rank, temporal shuffle gap,
    and periodic trajectory-inpainting planning probes on validation data.
    """

    def __init__(self, run_dir: Path, cfg):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.metrics_file = self.run_dir / "metrics.jsonl"

    def on_fit_start(self, trainer, pl_module):
        pl_module._val_probe_cache = {"embs": [], "actions": [], "n_batches": 0}
        if trainer.is_global_zero:
            model = pl_module.model
            counts = {
                "params/encoder": sum(p.numel() for p in model.encoder.parameters()),
                "params/ebm": sum(p.numel() for p in model.ebm.parameters()),
                "params/idm": sum(p.numel() for p in model.idm.parameters()),
                "params/total": sum(p.numel() for p in model.parameters()),
            }
            print("[EBM-WM] Parameter counts:", json.dumps(counts, indent=2))
            if trainer.logger is not None:
                trainer.logger.log_metrics(counts, step=trainer.global_step)
            self._append_jsonl(
                {
                    "event": "fit_start",
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    **counts,
                }
            )

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module._val_probe_cache = {"embs": [], "actions": [], "n_batches": 0}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        cache = getattr(pl_module, "_val_probe_cache", None)
        extra_metrics = {}

        if cache and len(cache["embs"]) > 0:
            device = pl_module.device
            model = pl_module.model
            all_embs = torch.cat(cache["embs"], dim=0)
            all_acts = torch.cat(cache["actions"], dim=0)

            extra_metrics["val_probe/effective_rank"] = float(compute_effective_rank(all_embs).item())

            probe_n = min(all_embs.size(0), int(self.cfg.logging.val_probe_samples))
            sample_embs = all_embs[:probe_n].to(device)
            sample_acts = all_acts[:probe_n].to(device)

            was_training = model.training
            model.eval()
            with torch.no_grad():
                e_real, _ = model.compute_energy(sample_embs)
                e_rev, _ = model.compute_energy(torch.flip(sample_embs, dims=[1]))
                shuf = sample_embs.clone()
                if probe_n > 1:
                    for t_idx in range(1, shuf.size(1)):
                        perm = torch.roll(torch.arange(probe_n, device=device), shifts=t_idx)
                        shuf[:, t_idx] = sample_embs[perm, t_idx]
                e_shuf, _ = model.compute_energy(shuf)

                extra_metrics["val_probe/energy_real"] = float(e_real.mean().item())
                extra_metrics["val_probe/energy_reversed"] = float(e_rev.mean().item())
                extra_metrics["val_probe/energy_shuffled"] = float(e_shuf.mean().item())
                extra_metrics["val_probe/energy_gap_rev_minus_real"] = float((e_rev - e_real).mean().item())
                extra_metrics["val_probe/energy_gap_shuf_minus_real"] = float((e_shuf - e_real).mean().item())

            every_n = int(self.cfg.logging.val_inpaint_every_n_epochs)
            epoch_num = trainer.current_epoch + 1
            if every_n > 0 and (epoch_num % every_n == 0 or epoch_num == trainer.max_epochs):
                inpaint_n = min(probe_n, int(self.cfg.logging.val_inpaint_samples))
                planner = EBMLatentPlanner(
                    model=model,
                    n_steps=int(self.cfg.logging.val_inpaint_steps),
                    lr=float(self.cfg.logging.val_inpaint_lr),
                    num_samples=int(self.cfg.logging.val_inpaint_restarts),
                    init_noise_std=float(self.cfg.logging.val_inpaint_noise_std),
                    seed=int(self.cfg.seed),
                )
                inpaint_metrics = planner.validate_inpainting(
                    clean_emb=sample_embs[:inpaint_n],
                    true_actions=sample_acts[:inpaint_n],
                )
                extra_metrics.update(inpaint_metrics)

            if was_training:
                model.train()

        if extra_metrics:
            pl_module.log_dict(extra_metrics, on_epoch=True, sync_dist=False)

        record = {"epoch": trainer.current_epoch + 1, "global_step": trainer.global_step}
        for k, v in trainer.callback_metrics.items():
            if torch.is_tensor(v) and v.ndim == 0:
                record[k] = float(v.detach().cpu().item())
        record.update(extra_metrics)
        self._append_jsonl(record)

    def _append_jsonl(self, record: dict):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _get_or_create_wandb_id(run_dir: Path, auto_resume: bool) -> str:
    """Persist WandB run ID in run_dir so resumed sessions append to the same WandB run."""
    id_file = run_dir / "wandb_run_id.txt"
    if auto_resume and id_file.exists():
        saved = id_file.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    new_id = uuid.uuid4().hex[:12]
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(new_id, encoding="utf-8")
    return new_id


@hydra.main(version_base=None, config_path="./config/train", config_name="ebm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.idm.action_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    # Deterministic, DDP-aware resumable samplers so mid-epoch resume continues at the exact batch
    train_sampler = ResumableDistributedSampler(
        dataset_len=len(train_set),
        batch_size=cfg.loader.batch_size,
        seed=cfg.seed,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = ResumableDistributedSampler(
        dataset_len=len(val_set),
        batch_size=cfg.loader.batch_size,
        seed=cfg.seed,
        shuffle=False,
        drop_last=False,
    )

    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
    train = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        **loader_kwargs,
    )
    val = torch.utils.data.DataLoader(
        val_set,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    pl_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(ebm_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##  checkpoint & resume ##
    ##########################

    run_id = cfg.get("subdir") or cfg.output_model_name
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    auto_resume = bool(cfg.training.auto_resume)
    last_ckpt = run_dir / "last.ckpt"
    resume_ckpt = str(last_ckpt) if (auto_resume and last_ckpt.exists()) else None
    if resume_ckpt:
        print(f"[EBM-WM] Auto-resuming training from {resume_ckpt}")
    else:
        print(f"[EBM-WM] Starting fresh training run in {run_dir}")

    logger = None
    if cfg.wandb.enabled:
        wandb_cfg = OmegaConf.to_container(cfg.wandb.config, resolve=True)
        wandb_id = _get_or_create_wandb_id(run_dir, auto_resume=auto_resume)
        logger = WandbLogger(id=wandb_id, resume="allow", **wandb_cfg)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    save_every_n_steps = int(cfg.logging.save_every_n_steps)

    # IMPORTANT: ResumableDatasetCallback MUST come BEFORE ModelCheckpoint
    resumable_cb = ResumableDatasetCallback(train_sampler=train_sampler)
    step_ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        every_n_train_steps=save_every_n_steps,
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    latest_weights_cb = LatestStepWeightsCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        save_every_n_steps=save_every_n_steps,
    )
    research_cb = EBMResearchCallback(run_dir=run_dir, cfg=cfg)

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs["use_distributed_sampler"] = False

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[resumable_cb, step_ckpt_cb, latest_weights_cb, research_cb],
        num_sanity_val_steps=0 if resume_ckpt else 1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=pl_module,
        data=data_module,
        ckpt_path=resume_ckpt,
    )

    manager()
    return


if __name__ == "__main__":
    run()
