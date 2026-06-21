"""On-disk persistence for BAO Service outputs (Service 5).

Layout under ``{artifact_dir}/bao/``::

    bao/
      _index.json            # cache_key → bao_id
      {bao_id}/
        meta.json            # full BAOResult

BAO results are pure metadata (selected angles + scores), so this store is just
the cache-index plumbing shared with the dose/optimization stores plus an atomic
``meta.json`` write — no large binary artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from ..models.bao import BAOResult
from .dose_persistence import _IndexedStoreBase

META_FILENAME = "meta.json"


class BAOStore(_IndexedStoreBase):
    """File-backed BAO persistence with a JSON cache index."""

    def lookup_by_cache_key(self, cache_key: str) -> Optional[BAOResult]:
        bao_id = self._load_index().get(cache_key)
        if not bao_id:
            return None
        return self.get_by_id(bao_id)

    def get_by_id(self, bao_id: str) -> Optional[BAOResult]:
        meta = self.base_dir / bao_id / META_FILENAME
        if not meta.exists():
            return None
        try:
            return BAOResult.model_validate(json.loads(meta.read_text()))
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, *, bao_id: str, cache_key: str, result: BAOResult) -> Path:
        root = self.base_dir / bao_id
        root.mkdir(parents=True, exist_ok=True)
        meta = root / META_FILENAME
        tmp = meta.with_name(f".tmp.{uuid.uuid4().hex}.{META_FILENAME}")
        tmp.write_text(result.model_dump_json(indent=2))
        os.replace(tmp, meta)
        index = self._load_index()
        index[cache_key] = bao_id
        self._save_index(index)
        return root

    def delete_by_id(self, bao_id: str) -> bool:
        root = self.base_dir / bao_id
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
            if index.get(cache_key) == bao_id:
                index.pop(cache_key)
                self._save_index(index)
        shutil.rmtree(root, ignore_errors=True)
        return True

    def list_ids(self) -> List[str]:
        if not self.base_dir.exists():
            return []
        return sorted(
            p.name for p in self.base_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".") and (p / META_FILENAME).exists()
        )


__all__ = ["BAOStore", "META_FILENAME"]
