from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """Stores only the latest and best checkpoints for a run."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.best_score = float("-inf")

    def _save(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.directory / filename
        torch.save(payload, path)
        return path

    def save_last(
        self,
        agent: Any,
        step: int,
        score: float | None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        payload = {
            "step": step,
            "score": score,
            "metadata": metadata or {},
            "agent": agent.state_dict(),
        }
        return self._save("last.pt", payload)

    def maybe_save_best(
        self,
        agent: Any,
        step: int,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if score <= self.best_score:
            return None
        self.best_score = score
        payload = {
            "step": step,
            "score": score,
            "metadata": metadata or {},
            "agent": agent.state_dict(),
        }
        return self._save("best.pt", payload)
