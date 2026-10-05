"""
checkpoint.py

Generic checkpoint/resume for a long chunked computation. This module
knows nothing about solar physics or geometry -- it persists a dict of
numpy arrays plus a progress marker (how many hours are already complete)
and a config hash, so simulation.py can resume a long run after an
interruption while refusing to silently resume into a checkpoint that
doesn't match the configuration currently being run (different city
model, different simulation mode, different time range, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utilities import get_logger

logger = get_logger()


@dataclass
class CheckpointState:
    last_completed_hour: int  # exclusive end index of hours completed so far
    arrays: dict[str, np.ndarray]


class CheckpointManager:
    """One checkpoint file per run, written atomically (write to a temp
    file, then rename) so a crash mid-save can never leave a corrupt
    checkpoint that silently resumes into garbage.
    """

    def __init__(self, checkpoint_dir: str | Path, config_hash: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_hash = config_hash
        self.path = self.checkpoint_dir / "checkpoint.npz"

    def load_if_valid(self) -> CheckpointState | None:
        if not self.path.exists():
            return None
        try:
            with np.load(self.path, allow_pickle=False) as npz:
                stored_hash = str(npz["config_hash"])
                if stored_hash != self.config_hash:
                    logger.warning(
                        f"Checkpoint at {self.path} was produced by a different configuration "
                        "(city model / mode / time range / AOD source changed); ignoring it and "
                        "starting fresh. Delete it manually, or set run.force_restart=true, if this "
                        "mismatch was intentional."
                    )
                    return None
                last_completed_hour = int(npz["last_completed_hour"])
                arrays = {k: npz[k] for k in npz.files if k not in ("config_hash", "last_completed_hour")}
        except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint must never crash the run
            logger.warning(f"Checkpoint at {self.path} could not be read ({exc}); starting fresh")
            return None

        logger.info(f"Resuming from checkpoint: {last_completed_hour:,} hour(s) already completed")
        return CheckpointState(last_completed_hour=last_completed_hour, arrays=arrays)

    def save(self, last_completed_hour: int, arrays: dict[str, np.ndarray]) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: np.savez_compressed silently appends ".npz" to the target filename if it
        # doesn't already end with ".npz" -- so the temp file name must already end in
        # ".npz", or the rename below would look for a file numpy never actually created.
        tmp_path = self.checkpoint_dir / f"{self.path.stem}.tmp.npz"
        np.savez_compressed(
            tmp_path,
            config_hash=np.array(self.config_hash),
            last_completed_hour=np.array(last_completed_hour),
            **arrays,
        )
        tmp_path.replace(self.path)  # atomic rename on the same filesystem
        logger.debug(f"Checkpoint saved at hour {last_completed_hour:,}")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
            logger.info(f"Cleared checkpoint at {self.path}")
