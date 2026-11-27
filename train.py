import os
import uuid
from functools import partial
from pathlib import Path
from typing import Optional

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from module import SIGReg
from resumable_sampler import ResumableDistributedSampler
from utils import get_column_normalizer, get_img_preprocessor


class ResumableDatasetCallback(Callback):
    """Persists exact mid-epoch batch progress in checkpoints and restores the sampler.

    MUST be placed BEFORE ModelCheckpoint in `trainer.callbacks` so `batches_seen_in_epoch`
    is updated for the just-completed batch before `last.ckpt` is written.
    """

    def __init__(self, train_sampler: ResumableDistributedSampler):
        super().__init__()
        self.train_sampler = train_sampler
        self.batches_seen_in_epoch: int = 0
        self.last_epoch: int = 0
        self.resumed_epoch: Optional[int] = None

    def on_train_epoch_start(self, trainer, pl_module):
        cur_epoch = int(trainer.current_epoch)
        self.train_sampler.set_epoch(cur_epoch)
        full_batches = self.train_sampler.num_batches_per_rank()

        if (
            self.resumed_epoch is not None
            and cur_epoch == self.resumed_epoch
            and 0 < self.batches_seen_in_epoch <= full_batches
        ):
            self.train_sampler.set_resume(cur_epoch, self.batches_seen_in_epoch)
            if trainer.is_global_zero:
                print(
                    f"[LeWM] Resuming dataset mid-epoch: epoch={cur_epoch}, "
                    f"skipping {self.batches_seen_in_epoch}/{full_batches} completed batches/rank."
                )
            self.resumed_epoch = None
        else:
            self.batches_seen_in_epoch = 0
            self.train_sampler.set_resume(None, 0)
            self.resumed_epoch = None
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


class LatestStepCheckpointCallback(Callback):
    """Saves ONLY the latest inference weights (`weights_latest.pt`) every N train steps
    (never on intermediate epochs), and ensures final `last.ckpt` + `weights_latest.pt`
    are saved when training finishes.
    """

    def __init__(
        self,
        run_name: str,
        run_dir: Path,
        cfg,
        save_every_n_steps: int = 500,
    ):
        super().__init__()
        self.run_name = run_name
        self.run_dir = Path(run_dir)
        self.cfg = cfg
        self.save_every_n_steps = max(1, int(save_every_n_steps))
        self._last_saved_step: int = -1

    def _save_latest_weights(self, trainer, pl_module):
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_saved_step:
            return
        if trainer.is_global_zero:
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
            self._save_latest_weights(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        step = int(trainer.global_step)
        if step > 0 and step != self._last_saved_step:
            trainer.save_checkpoint(str(self.run_dir / "last.ckpt"))
            self._save_latest_weights(trainer, pl_module)


def _get_or_create_wandb_id(run_dir: Path, auto_resume: bool) -> str:
    """Persist WandB run ID in run_dir so resumed sessions append to the same run."""
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


def lejepa_forward(self, batch, stage, cfg):
    """Encode observations into beliefs, unroll multi-step predictions, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    belief = output["belief"]  # (B, T, D)
    frame_emb = output["frame_emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    if self.model.belief_encoder is not None:
        # Multi-step belief rollout: from every b_t, autoregressively unroll the
        # predictor to predict all future beliefs b_{t+k} ahead of it (up to
        # cfg.rollout_steps if set, otherwise all T - 1 steps ahead).
        T = belief.size(1)
        max_rollout_steps = cfg.get("rollout_steps", None)
        num_rollout_steps = (
            T - 1 if max_rollout_steps is None else min(max_rollout_steps, T - 1)
        )
        assert num_rollout_steps >= 1, f"Need at least 2 timesteps, got T={T}"

        sq_errors = []
        pred = belief[:, :-1]  # starting beliefs b_0 .. b_{T-2}
        for k in range(1, num_rollout_steps + 1):
            state_in = pred if k == 1 else pred[:, :-1]
            act_in = act_emb[:, k - 1 : T - 1]
            pred = self.model.predict(state_in, act_in)
            tgt = belief[:, k:]
            sq_errors.append((pred - tgt).pow(2))

        output["pred_loss"] = torch.cat(sq_errors, dim=1).mean()
    else:
        ctx_belief = belief[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_belief = belief[:, n_preds:]
        pred_belief = self.model.predict(ctx_belief, ctx_act)
        output["pred_loss"] = (pred_belief - tgt_belief).pow(2).mean()

    # Per-timestep belief SIGReg (independent across batch at each temporal position)
    output["sigreg_loss"] = self.sigreg(belief.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    # Optional frame-level SIGReg (disabled by default)
    frame_sigreg_cfg = cfg.loss.get("frame_sigreg", None)
    if (
        frame_sigreg_cfg is not None
        and frame_sigreg_cfg.get("enabled", False)
        and frame_sigreg_cfg.get("weight", 0.0) > 0
    ):
        output["frame_sigreg_loss"] = self.sigreg(frame_emb.transpose(0, 1))
        output["loss"] = (
            output["loss"]
            + frame_sigreg_cfg.get("weight", 0.0) * output["frame_sigreg_loss"]
        )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    if cache_dir is not None and not Path(dataset_name).is_absolute():
        direct_path = Path(cache_dir) / dataset_name
        if direct_path.exists():
            dataset_name = str(direct_path)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    loader_kwargs = OmegaConf.to_container(cfg.loader, resolve=True)
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

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    grad_clip_val = trainer_kwargs.pop("gradient_clip_val", None)
    trainer_kwargs["use_distributed_sampler"] = False

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
            "gradient_clip_val": grad_clip_val,
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    pl_module = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##  checkpoint & resume ##
    ##########################

    run_id = cfg.get("subdir") or cfg.output_model_name
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = cfg.get("training", None) or {}
    auto_resume = bool(training_cfg.get("auto_resume", True))
    save_every_n_steps = int(training_cfg.get("save_every_n_steps", 500))

    last_ckpt = run_dir / "last.ckpt"
    resume_ckpt = str(last_ckpt) if (auto_resume and last_ckpt.exists()) else None
    if resume_ckpt:
        print(f"[LeWM] Auto-resuming training from {resume_ckpt}")
    else:
        print(f"[LeWM] Starting fresh training run in {run_dir}")

    logger = None
    if cfg.wandb.enabled:
        wandb_cfg = dict(OmegaConf.to_container(cfg.wandb.config, resolve=True) or {})
        wandb_cfg.pop("id", None)
        wandb_cfg.pop("resume", None)
        if wandb_cfg.get("entity") in (None, "", "null", "lewm"):
            wandb_cfg["entity"] = None
        wandb_id = _get_or_create_wandb_id(run_dir, auto_resume=auto_resume)
        logger = WandbLogger(id=wandb_id, resume="allow", **wandb_cfg)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        OmegaConf.save(cfg, f)

    # IMPORTANT: ResumableDatasetCallback MUST come BEFORE ModelCheckpoint so
    # batches_seen_in_epoch includes the just-completed batch when last.ckpt is saved.
    resumable_cb = ResumableDatasetCallback(train_sampler=train_sampler)
    step_ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir),
        every_n_train_steps=save_every_n_steps,
        save_top_k=0,
        save_last=True,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
    )
    latest_weights_cb = LatestStepCheckpointCallback(
        run_name=cfg.output_model_name,
        run_dir=run_dir,
        cfg=cfg.model,
        save_every_n_steps=save_every_n_steps,
    )

    pl.seed_everything(cfg.seed, workers=True)

    trainer = pl.Trainer(
        **trainer_kwargs,
        default_root_dir=str(run_dir),
        callbacks=[resumable_cb, step_ckpt_cb, latest_weights_cb],
        num_sanity_val_steps=0 if resume_ckpt else 1,
        logger=logger,
        enable_checkpointing=True,
    )
    trainer.gradient_clip_val_ = grad_clip_val
    trainer.gradient_clip_algorithm_ = "norm"

    trainer.fit(pl_module, datamodule=data_module, ckpt_path=resume_ckpt)
    return


if __name__ == "__main__":
    run()
