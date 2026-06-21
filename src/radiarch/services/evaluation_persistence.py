"""On-disk persistence for Evaluation Service outputs (Service 6).

Layout under ``{artifact_dir}/evaluation/``::

    evaluation/
      _index.json            # cache_key → evaluation_id
      {evaluation_id}/
        meta.json            # full EvaluationResult (DVH curves + indices + gamma)

Evaluation results are JSON-only (DVH curves are modest arrays serialized inline),
so this is the cache-index plumbing plus an atomic ``meta.json`` write.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from ..models.evaluation import EvaluationResult
from .dose_persistence import _IndexedStoreBase

META_FILENAME = "meta.json"


class EvaluationStore(_IndexedStoreBase):
    """File-backed evaluation persistence with a JSON cache index."""

    def lookup_by_cache_key(self, cache_key: str) -> Optional[EvaluationResult]:
        eid = self._load_index().get(cache_key)
        if not eid:
            return None
        return self.get_by_id(eid)

    def get_by_id(self, evaluation_id: str) -> Optional[EvaluationResult]:
        meta = self.base_dir / evaluation_id / META_FILENAME
        if not meta.exists():
            return None
        try:
            return EvaluationResult.model_validate(json.loads(meta.read_text()))
        except (OSError, json.JSONDecodeError):
            return None

    def save(self, *, evaluation_id: str, cache_key: str,
             result: EvaluationResult) -> Path:
        root = self.base_dir / evaluation_id
        root.mkdir(parents=True, exist_ok=True)
        meta = root / META_FILENAME
        tmp = meta.with_name(f".tmp.{uuid.uuid4().hex}.{META_FILENAME}")
        tmp.write_text(result.model_dump_json(indent=2))
        os.replace(tmp, meta)
        index = self._load_index()
        index[cache_key] = evaluation_id
        self._save_index(index)
        return root

    def delete_by_id(self, evaluation_id: str) -> bool:
        root = self.base_dir / evaluation_id
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
            if index.get(cache_key) == evaluation_id:
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


__all__ = ["EvaluationStore", "META_FILENAME"]
