"""On-disk persistence for Geometry Service outputs.

Layout on disk under ``{artifact_dir}/geometry/``::

    geometry/
      _index.json                 # cache_key → geometry_id lookup
      {geometry_id}/
        density.nii.gz
        masks.nii.gz
        meta.json                 # full GeometryResult minus the URIs

NIfTI I/O goes through SimpleITK (already a project dep). We write
density as float32 and masks as uint16; the target GridSpec's
axis-aligned affine is mapped to ITK's ``SetSpacing`` + ``SetOrigin`` +
``SetDirection`` (identity direction only in v1 — rotational affines
are rejected upstream in the resampler).

Atomicity
---------
Outputs are written under ``{geometry_id}/.tmp/`` first, then renamed to
their final names in a single step. A crash mid-write leaves the .tmp
directory behind (cleaned up on the next cache-miss for the same key)
but never poisons the cache with half-written files.

The cache index is a plain JSON file. For v1 (synchronous mode) this is
simpler than introducing an Alembic migration and a Geometry table; the
async-mode PR will migrate reads to the DB.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..models.geometry import GeometryResult, GridSpec


DENSITY_FILENAME = "density.nii.gz"
MASKS_FILENAME = "masks.nii.gz"
CT_FILENAME = "ct.nii.gz"
META_FILENAME = "meta.json"
INDEX_FILENAME = "_index.json"


# ---------------------------------------------------------------------------
# GridSpec ↔ SimpleITK
# ---------------------------------------------------------------------------

def _apply_gridspec_to_itk(image, grid: GridSpec) -> None:
    """Stamp an ``sitk.Image`` with spacing/origin/direction from ``grid``.

    NB: v1 is axis-aligned, so the direction cosine matrix is the 3×3
    identity. Rotational affines are rejected in ``resample_to_grid``.
    """
    image.SetSpacing(tuple(float(s) for s in grid.spacing_mm))
    if grid.origin_mm is not None:
        image.SetOrigin(tuple(float(o) for o in grid.origin_mm))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))


def _write_nifti(volume: np.ndarray, grid: GridSpec, path: Path) -> None:
    """Write a 3D numpy array as a NIfTI file at ``path``.

    SimpleITK expects arrays in (z, y, x) order; we store volumes in
    (i, j, k) = (x, y, z) order (matches OpenTPS and DICOM), so we
    transpose on the way out and on the way back in (see ``_read_nifti``).
    """
    import SimpleITK as sitk  # local import → keeps module-load light

    # Transpose (i,j,k)=(x,y,z) → (z,y,x) for SimpleITK.
    arr = np.asarray(volume)
    image = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)))
    _apply_gridspec_to_itk(image, grid)
    # SimpleITK handles .nii.gz transparently based on extension.
    sitk.WriteImage(image, str(path))


def _read_nifti(path: Path) -> Tuple[np.ndarray, GridSpec]:
    """Read a NIfTI file; return ``(array_in_ijk_order, GridSpec)``.

    The inverse of :func:`_write_nifti`. Primarily used in tests /
    downstream services that want to round-trip the geometry outputs.
    """
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    arr = np.transpose(arr, (2, 1, 0))    # → (x, y, z)

    spacing = tuple(float(s) for s in image.GetSpacing())
    origin = tuple(float(o) for o in image.GetOrigin())
    size = tuple(int(s) for s in arr.shape)

    spec = GridSpec(spacing_mm=spacing, origin_mm=origin, size=size)
    spec.affine = spec.compute_affine()
    return arr, spec


# ---------------------------------------------------------------------------
# On-disk geometry store (synchronous-mode cache)
# ---------------------------------------------------------------------------

@dataclass
class GeometryPaths:
    """Convenience bundle of the final on-disk paths for one geometry.

    ``ct`` is the resampled HU volume — present for all geometries built
    since the D6.1 change. Older cached geometries won't have the file
    on disk; readers should treat its absence as "no CT available".
    """

    root: Path
    density: Path
    masks: Path
    ct: Path
    meta: Path

    @classmethod
    def for_id(cls, base_dir: Path, geometry_id: str) -> "GeometryPaths":
        root = base_dir / geometry_id
        return cls(
            root=root,
            density=root / DENSITY_FILENAME,
            masks=root / MASKS_FILENAME,
            ct=root / CT_FILENAME,
            meta=root / META_FILENAME,
        )


class GeometryStore:
    """File-backed geometry persistence with a JSON cache index.

    Not thread-safe across processes (fine for synchronous mode — the
    async-mode store will layer a DB row with a unique index on cache_key
    to serialize concurrent builds).
    """

    def __init__(self, base_dir: str | os.PathLike[str]) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ---- cache index --------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self.base_dir / INDEX_FILENAME

    def _load_index(self) -> Dict[str, str]:
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text())
        except (OSError, json.JSONDecodeError):
            # A corrupt index shouldn't wedge the whole service; treat
            # it as empty and overwrite on the next successful write.
            return {}

    def _save_index(self, index: Dict[str, str]) -> None:
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(index, indent=2, sort_keys=True))
        os.replace(tmp, self._index_path)

    def lookup_by_cache_key(self, cache_key: str) -> Optional[GeometryResult]:
        """Return the cached ``GeometryResult`` for ``cache_key``, or None."""
        index = self._load_index()
        geometry_id = index.get(cache_key)
        if not geometry_id:
            return None
        return self.get_by_id(geometry_id)

    def get_by_id(self, geometry_id: str) -> Optional[GeometryResult]:
        paths = GeometryPaths.for_id(self.base_dir, geometry_id)
        if not paths.meta.exists():
            return None
        try:
            data = json.loads(paths.meta.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        return GeometryResult.model_validate(data)

    # ---- writes -------------------------------------------------------

    def save(
        self,
        *,
        geometry_id: str,
        cache_key: str,
        density: np.ndarray,
        masks: np.ndarray,
        result: GeometryResult,
        ct: Optional[np.ndarray] = None,
    ) -> GeometryPaths:
        """Write density + masks (+ optional CT) + meta atomically, then update the cache index.

        ``ct`` is the CT volume in Hounsfield Units on the target grid
        (same shape as ``density``). When provided, it's written as
        ``ct.nii.gz`` and the caller is responsible for setting
        ``result.ct_grid_uri`` to the matching path. Omitted CTs are not
        an error — the analytic engine doesn't need one — but engines
        like MCsquare that depend on a real CT will then refuse to run.
        """
        paths = GeometryPaths.for_id(self.base_dir, geometry_id)
        # Use a sibling .tmp dir so the final rename is on the same fs.
        with tempfile.TemporaryDirectory(dir=self.base_dir, prefix=f".{geometry_id}.tmp.") as tmp:
            tmp_path = Path(tmp)
            tmp_density = tmp_path / DENSITY_FILENAME
            tmp_masks = tmp_path / MASKS_FILENAME
            tmp_ct = tmp_path / CT_FILENAME
            tmp_meta = tmp_path / META_FILENAME

            _write_nifti(density.astype(np.float32, copy=False), result.grid_spec, tmp_density)
            _write_nifti(masks.astype(np.uint16, copy=False), result.grid_spec, tmp_masks)
            if ct is not None:
                # Store CT as int16 — it's HU, which fits nicely and halves
                # the file size vs float32.
                _write_nifti(ct.astype(np.int16, copy=False), result.grid_spec, tmp_ct)
            tmp_meta.write_text(result.model_dump_json(indent=2))

            # Atomic replace: if paths.root already exists (retry of a
            # failed build), nuke it first.
            if paths.root.exists():
                shutil.rmtree(paths.root)
            os.replace(tmp_path, paths.root)
            # ``tmp_path`` has moved to ``paths.root``; replace the context
            # manager's now-invalid reference with a fresh tempdir so the
            # __exit__ cleanup becomes a no-op.
            os.makedirs(tmp_path, exist_ok=True)

        # Update the cache index last — a crash between file-write and
        # index-update leaves orphan geometry dirs (harmless) instead of
        # dangling index entries (harmful).
        index = self._load_index()
        index[cache_key] = geometry_id
        self._save_index(index)
        return paths

    # ---- deletes ------------------------------------------------------

    def delete_by_id(self, geometry_id: str) -> bool:
        """Remove a geometry directory and scrub its cache_key from the index.

        Returns True when a geometry was actually deleted, False when the
        id was unknown. Safe to call against partial state (missing meta,
        missing files, stale index entry) — each step is defensive.
        """
        root = self.base_dir / geometry_id
        if not root.exists():
            return False

        # Scrub the cache index first. If we're about to delete files,
        # the index entry should disappear atomically enough that a
        # racing reader never sees "cache hit → files gone".
        cache_key = self._read_cache_key(root)
        if cache_key is not None:
            index = self._load_index()
            if index.get(cache_key) == geometry_id:
                index.pop(cache_key)
                self._save_index(index)

        shutil.rmtree(root, ignore_errors=True)
        return True

    def _read_cache_key(self, root: Path) -> Optional[str]:
        meta = root / META_FILENAME
        if not meta.exists():
            return None
        try:
            return json.loads(meta.read_text()).get("cache_key")
        except (OSError, json.JSONDecodeError):
            return None

    # ---- debugging helpers -------------------------------------------

    def list_ids(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / META_FILENAME).exists()
        )


__all__ = [
    "DENSITY_FILENAME",
    "MASKS_FILENAME",
    "CT_FILENAME",
    "META_FILENAME",
    "INDEX_FILENAME",
    "GeometryPaths",
    "GeometryStore",
]
