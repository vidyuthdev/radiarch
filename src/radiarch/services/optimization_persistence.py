"""On-disk persistence for Optimization Service outputs (Service 4).

Layout under ``{artifact_dir}/optimization/``::

    optimization/
      _index.json                    # cache_key → optimization_id
      {opt_id}/
        weights.npy                  # optimal fluence-weight vector (float32)
        dose.nii.gz                  # final dose volume (Gy, float32)
        meta.json                    # full OptimizationResult
        checkpoints/
          iter_{N}.npy               # weight snapshot at iteration N

Two differences from :mod:`dose_persistence`:

* Checkpoints are written **incrementally during the solver loop**, before
  the final result exists. So rather than the dose store's "build everything
  in a tempdir then ``os.replace`` the whole record" pattern, this store
  creates ``{opt_id}/`` up front (:meth:`prepare`), streams checkpoints into
  it, and at the end writes the final artifacts as individual atomic files
  (tempfile in the same dir → ``os.replace``). Whole-dir replace would clobber
  the checkpoints written mid-run.
* The cache index is updated *last*, only on a successful :meth:`save`, so a
  crashed run leaves an orphan dir (wasted space) but never a dangling cache
  entry.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import SimpleITK as sitk

from ..models.optimization import CheckpointInfo, OptimizationResult
from .dose_persistence import _IndexedStoreBase

WEIGHTS_FILENAME = "weights.npy"
DOSE_FILENAME = "dose.nii.gz"
META_FILENAME = "meta.json"
CHECKPOINT_DIRNAME = "checkpoints"


@dataclass
class OptimizationPaths:
    """On-disk paths for one optimization record."""

    root: Path
    weights: Path
    dose: Path
    meta: Path
    checkpoints: Path

    @classmethod
    def for_id(cls, base_dir: Path, opt_id: str) -> "OptimizationPaths":
        root = base_dir / opt_id
        return cls(
            root=root,
            weights=root / WEIGHTS_FILENAME,
            dose=root / DOSE_FILENAME,
            meta=root / META_FILENAME,
            checkpoints=root / CHECKPOINT_DIRNAME,
        )

    def checkpoint_path(self, iteration: int) -> Path:
        return self.checkpoints / f"iter_{iteration}.npy"


def _atomic_write_bytes(path: Path, writer) -> None:
    """Write to a unique temp file in the target dir, then ``os.replace``.

    ``writer`` is a callable receiving the temp ``Path``; it must fully
    materialize the file. The rename is atomic on POSIX so readers never see
    a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the real suffix at the end (".nii.gz", ".npy") so SimpleITK / numpy
    # can pick the right writer from the temp filename.
    tmp = path.with_name(f".tmp.{uuid.uuid4().hex}.{path.name}")
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    # np.save() appends ".npy" when given a path, which would break the atomic
    # rename — write through a file handle so the temp name is used verbatim.
    def _w(p: Path) -> None:
        with open(p, "wb") as fh:
            np.save(fh, np.asarray(arr, dtype=np.float32))
    _atomic_write_bytes(path, _w)


def _write_dose_volume(path: Path, dose: np.ndarray, spacing_mm: tuple) -> None:
    def _w(p: Path) -> None:
        img = sitk.GetImageFromArray(np.asarray(dose, dtype=np.float32))
        img.SetSpacing([float(s) for s in spacing_mm])
        sitk.WriteImage(img, str(p), useCompression=True)
    _atomic_write_bytes(path, _w)


class OptimizationStore(_IndexedStoreBase):
    """File-backed optimization persistence with a JSON cache index."""

    # -- lookups -------------------------------------------------------

    def lookup_by_cache_key(self, cache_key: str) -> Optional[OptimizationResult]:
        opt_id = self._load_index().get(cache_key)
        if not opt_id:
            return None
        return self.get_by_id(opt_id)

    def get_by_id(self, opt_id: str) -> Optional[OptimizationResult]:
        paths = OptimizationPaths.for_id(self.base_dir, opt_id)
        if not paths.meta.exists():
            return None
        try:
            data = json.loads(paths.meta.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return OptimizationResult.model_validate(data)

    # -- run lifecycle -------------------------------------------------

    def prepare(self, opt_id: str) -> OptimizationPaths:
        """Create the record dir + checkpoints subdir before the solver runs."""
        paths = OptimizationPaths.for_id(self.base_dir, opt_id)
        paths.checkpoints.mkdir(parents=True, exist_ok=True)
        return paths

    def write_checkpoint(
        self, opt_id: str, iteration: int, weights: np.ndarray, cost: float
    ) -> CheckpointInfo:
        """Atomically snapshot weights at ``iteration``; return its descriptor."""
        paths = OptimizationPaths.for_id(self.base_dir, opt_id)
        cp_path = paths.checkpoint_path(iteration)
        _save_npy(cp_path, weights)
        return CheckpointInfo(
            iteration=iteration, weights_uri=str(cp_path), cost=float(cost)
        )

    def prune_checkpoints(self, opt_id: str, keep: int) -> int:
        """Keep only the ``keep`` newest checkpoints; return count evicted."""
        paths = OptimizationPaths.for_id(self.base_dir, opt_id)
        if not paths.checkpoints.exists() or keep < 0:
            return 0
        cps = sorted(
            paths.checkpoints.glob("iter_*.npy"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        evict = cps[:-keep] if keep > 0 else cps
        for p in evict:
            p.unlink(missing_ok=True)
        return len(evict)

    def save(
        self,
        *,
        opt_id: str,
        cache_key: str,
        weights: np.ndarray,
        dose: np.ndarray,
        spacing_mm: tuple,
        result: OptimizationResult,
    ) -> OptimizationPaths:
        """Persist final weights + dose + meta; update the cache index last."""
        paths = self.prepare(opt_id)
        _save_npy(paths.weights, weights)
        _write_dose_volume(paths.dose, dose, spacing_mm)
        _atomic_write_bytes(
            paths.meta, lambda p: p.write_text(result.model_dump_json(indent=2))
        )
        index = self._load_index()
        index[cache_key] = opt_id
        self._save_index(index)
        return paths

    # -- maintenance ---------------------------------------------------

    def load_weights(self, opt_id: str) -> np.ndarray:
        paths = OptimizationPaths.for_id(self.base_dir, opt_id)
        if not paths.weights.exists():
            raise FileNotFoundError(f"weights missing for {opt_id}")
        return np.load(paths.weights)

    def delete_by_id(self, opt_id: str) -> bool:
        root = self.base_dir / opt_id
        if not root.exists():
            return False
        meta = root / META_FILENAME
        cache_key = None
        if meta.exists():
            try:
                cache_key = json.loads(meta.read_text()).get("cache_key")
            except (OSError, json.JSONDecodeError):
                pass
        if cache_key:
            index = self._load_index()
            if index.get(cache_key) == opt_id:
                index.pop(cache_key)
                self._save_index(index)
        shutil.rmtree(root, ignore_errors=True)
        return True

    def list_ids(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / META_FILENAME).exists()
        )


__all__ = [
    "WEIGHTS_FILENAME",
    "DOSE_FILENAME",
    "META_FILENAME",
    "CHECKPOINT_DIRNAME",
    "OptimizationPaths",
    "OptimizationStore",
]
