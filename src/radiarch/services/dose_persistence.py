"""On-disk persistence for Dose Service outputs.

Layout under ``{artifact_dir}/doses/``::

    doses/
      _index.json                    # cache_key → dose_id
      {dose_id}/
        dose.nii.gz                  # nominal dose grid (Gy, float32)
        meta.json                    # full DoseResult
        scenario_{hash}.nii.gz       # one per scenario in scenario_doses (optional)

And under ``{artifact_dir}/influence/``::

    influence/
      _index.json                    # cache_key → influence_id
      {influence_id}/
        dij.npz                      # sparse CSR matrix
        meta.json                    # full InfluenceResult

Atomicity follows the same belt-and-suspenders pattern as the geometry
and beam-model stores: write into a tempdir, ``os.replace`` into the
final location, update the index file last. The whole thing is safe to
restart at any point — orphan dirs without an index entry are wasted
space but never break a read.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import SimpleITK as sitk

from ..models.dose import DoseResult, InfluenceResult
from .dose_engines.protocol import InfluenceData


DOSE_FILENAME = "dose.nii.gz"
META_FILENAME = "meta.json"
INDEX_FILENAME = "_index.json"
INFLUENCE_FILENAME = "dij.npz"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@dataclass
class DosePaths:
    """On-disk paths for one dose record."""

    root: Path
    dose: Path
    meta: Path

    @classmethod
    def for_id(cls, base_dir: Path, dose_id: str) -> "DosePaths":
        root = base_dir / dose_id
        return cls(root=root, dose=root / DOSE_FILENAME, meta=root / META_FILENAME)

    def scenario_path(self, scenario_hash: str) -> Path:
        return self.root / f"scenario_{scenario_hash}.nii.gz"


@dataclass
class InfluencePaths:
    """On-disk paths for one influence record."""

    root: Path
    dij: Path
    meta: Path

    @classmethod
    def for_id(cls, base_dir: Path, influence_id: str) -> "InfluencePaths":
        root = base_dir / influence_id
        return cls(root=root, dij=root / INFLUENCE_FILENAME, meta=root / META_FILENAME)


# ---------------------------------------------------------------------------
# Helpers shared by both stores
# ---------------------------------------------------------------------------

class _IndexedStoreBase:
    """Common cache-index plumbing shared by dose + influence stores."""

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _index_path(self) -> Path:
        return self.base_dir / INDEX_FILENAME

    def _load_index(self) -> Dict[str, str]:
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_index(self, index: Dict[str, str]) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
        os.replace(tmp, self._index_path)


# ---------------------------------------------------------------------------
# Dose volume IO
# ---------------------------------------------------------------------------

def _write_dose_volume(path: Path, dose: np.ndarray, spacing_mm: tuple) -> None:
    """Write a (nz, ny, nx) float32 array to NIfTI with the geometry's spacing."""
    img = sitk.GetImageFromArray(dose.astype(np.float32))
    img.SetSpacing([float(s) for s in spacing_mm])
    sitk.WriteImage(img, str(path), useCompression=True)


def read_dose_volume(path: Path) -> np.ndarray:
    """Load a dose NIfTI back into a (nz, ny, nx) float32 array."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32)


# ---------------------------------------------------------------------------
# DoseStore
# ---------------------------------------------------------------------------

class DoseStore(_IndexedStoreBase):
    """File-backed dose persistence with a JSON cache index."""

    def lookup_by_cache_key(self, cache_key: str) -> Optional[DoseResult]:
        dose_id = self._load_index().get(cache_key)
        if not dose_id:
            return None
        return self.get_by_id(dose_id)

    def get_by_id(self, dose_id: str) -> Optional[DoseResult]:
        paths = DosePaths.for_id(self.base_dir, dose_id)
        if not paths.meta.exists():
            return None
        try:
            data = json.loads(paths.meta.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return DoseResult.model_validate(data)

    def save(
        self,
        *,
        dose_id: str,
        cache_key: str,
        nominal_dose: np.ndarray,
        spacing_mm: tuple,
        scenario_doses: Optional[Dict[str, np.ndarray]],
        result: DoseResult,
    ) -> DosePaths:
        """Write nominal dose + per-scenario doses + meta, atomically.

        ``scenario_doses`` maps scenario_hash → dose array. The meta JSON
        is expected to already list the scenarios in
        ``result.scenario_doses`` with matching hashes / URIs.
        """
        paths = DosePaths.for_id(self.base_dir, dose_id)
        with tempfile.TemporaryDirectory(
            dir=self.base_dir,
            prefix=f".{dose_id}.tmp.",
        ) as tmp:
            tmp_path = Path(tmp)
            _write_dose_volume(tmp_path / DOSE_FILENAME, nominal_dose, spacing_mm)
            if scenario_doses:
                for h, arr in scenario_doses.items():
                    _write_dose_volume(
                        tmp_path / f"scenario_{h}.nii.gz", arr, spacing_mm
                    )
            (tmp_path / META_FILENAME).write_text(result.model_dump_json(indent=2))

            if paths.root.exists():
                shutil.rmtree(paths.root)
            os.replace(tmp_path, paths.root)
            os.makedirs(tmp_path, exist_ok=True)  # quiet the cleanup

        index = self._load_index()
        index[cache_key] = dose_id
        self._save_index(index)
        return paths

    def delete_by_id(self, dose_id: str) -> bool:
        root = self.base_dir / dose_id
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
            if index.get(cache_key) == dose_id:
                index.pop(cache_key)
                self._save_index(index)
        shutil.rmtree(root, ignore_errors=True)
        return True

    def list_ids(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / META_FILENAME).exists()
        )


# ---------------------------------------------------------------------------
# InfluenceStore
# ---------------------------------------------------------------------------

class InfluenceStore(_IndexedStoreBase):
    """File-backed influence-matrix persistence with a JSON cache index."""

    def lookup_by_cache_key(self, cache_key: str) -> Optional[InfluenceResult]:
        influence_id = self._load_index().get(cache_key)
        if not influence_id:
            return None
        return self.get_by_id(influence_id)

    def get_by_id(self, influence_id: str) -> Optional[InfluenceResult]:
        paths = InfluencePaths.for_id(self.base_dir, influence_id)
        if not paths.meta.exists():
            return None
        try:
            data = json.loads(paths.meta.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return InfluenceResult.model_validate(data)

    def save(
        self,
        *,
        influence_id: str,
        cache_key: str,
        influence: InfluenceData,
        result: InfluenceResult,
    ) -> InfluencePaths:
        paths = InfluencePaths.for_id(self.base_dir, influence_id)
        with tempfile.TemporaryDirectory(
            dir=self.base_dir,
            prefix=f".{influence_id}.tmp.",
        ) as tmp:
            tmp_path = Path(tmp)
            np.savez_compressed(
                tmp_path / INFLUENCE_FILENAME,
                indptr=influence.indptr,
                indices=influence.indices,
                data=influence.data,
                shape=np.asarray([influence.n_voxels, influence.n_elements],
                                 dtype=np.int64),
            )
            (tmp_path / META_FILENAME).write_text(result.model_dump_json(indent=2))

            if paths.root.exists():
                shutil.rmtree(paths.root)
            os.replace(tmp_path, paths.root)
            os.makedirs(tmp_path, exist_ok=True)

        index = self._load_index()
        index[cache_key] = influence_id
        self._save_index(index)
        return paths

    def load_influence(self, influence_id: str) -> InfluenceData:
        """Round-trip the sparse Dij back from disk."""
        paths = InfluencePaths.for_id(self.base_dir, influence_id)
        if not paths.dij.exists():
            raise FileNotFoundError(f"dij missing for {influence_id}")
        with np.load(paths.dij) as npz:
            indptr = npz["indptr"]
            indices = npz["indices"]
            data = npz["data"]
            shape = npz["shape"]
        return InfluenceData(
            indptr=indptr,
            indices=indices,
            data=data,
            n_voxels=int(shape[0]),
            n_elements=int(shape[1]),
        )

    def delete_by_id(self, influence_id: str) -> bool:
        root = self.base_dir / influence_id
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
            if index.get(cache_key) == influence_id:
                index.pop(cache_key)
                self._save_index(index)
        shutil.rmtree(root, ignore_errors=True)
        return True

    def list_ids(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / META_FILENAME).exists()
        )


__all__ = [
    "DOSE_FILENAME",
    "META_FILENAME",
    "INDEX_FILENAME",
    "INFLUENCE_FILENAME",
    "DosePaths",
    "InfluencePaths",
    "DoseStore",
    "InfluenceStore",
    "read_dose_volume",
]
