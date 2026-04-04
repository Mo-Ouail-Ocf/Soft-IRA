from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover
    SummaryWriter = None


def _to_scalar(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return value.detach().float().mean().item()
    return None


class RunLogger:
    def __init__(self, cfg, run_name: str, output_dir: str | Path) -> None:
        self.cfg = cfg
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = self.output_dir / "tensorboard"
        self.wandb_dir = self.output_dir / "wandb"
        self.writer = None
        if cfg.logging.tensorboard and SummaryWriter is not None:
            self.writer = SummaryWriter(self.tensorboard_dir.as_posix())
        elif cfg.logging.tensorboard and SummaryWriter is None:
            print("TensorBoard logging requested, but tensorboard is not installed in the active environment.")

        self.wandb_run = None
        if cfg.logging.wandb and wandb is not None:
            self.wandb_dir.mkdir(parents=True, exist_ok=True)
            self.wandb_run = wandb.init(
                project=cfg.project.name,
                entity=cfg.logging.wandb_entity or None,
                name=run_name,
                dir=self.wandb_dir.as_posix(),
                mode=cfg.logging.wandb_mode,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=False,
            )
        elif cfg.logging.wandb and wandb is None:
            print("W&B logging requested, but wandb is not installed in the active environment.")

        OmegaConf.save(cfg, self.output_dir / "resolved_config.yaml")

    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        sanitized: dict[str, float | int] = {}
        for key, value in metrics.items():
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            sanitized[key] = scalar
            if self.writer is not None:
                self.writer.add_scalar(key, scalar, step)
        if self.wandb_run is not None and sanitized:
            self.wandb_run.log(sanitized, step=step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
