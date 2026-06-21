"""Unit tests for :class:`OptimizationStore` (Service 4, O11 persistence).

Covers the atomic save/lookup round-trip, checkpoint streaming + pruning, the
cache index, and deletion — independent of the solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from radiarch.models.optimization import ConvergenceInfo, OptimizationResult
from radiarch.services.optimization_persistence import OptimizationStore


def _result(opt_id: str, cache_key: str, store: OptimizationStore) -> OptimizationResult:
    paths = store.prepare(opt_id)
    return OptimizationResult(
        optimization_id=opt_id,
        cache_key=cache_key,
        weights_ref_uri=str(paths.weights),
        dose_ref_uri=str(paths.dose),
        convergence=ConvergenceInfo(success=True, iterations=3, final_cost=1.5,
                                    cost_history=[5.0, 3.0, 1.5]),
        compute_time_s=0.1,
        geometry_id="g-1", beam_model_id="bm-1",
        engine_name="analytic", engine_version="0.1.0",
    )


def test_save_and_get_roundtrip(tmp_path):
    store = OptimizationStore(tmp_path)
    weights = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    dose = np.ones((2, 3, 3), dtype=np.float32)
    result = _result("opt-1", "ck-1", store)
    store.save(opt_id="opt-1", cache_key="ck-1", weights=weights, dose=dose,
               spacing_mm=(2.0, 2.0, 3.0), result=result)

    got = store.get_by_id("opt-1")
    assert got is not None
    assert got.optimization_id == "opt-1"
    np.testing.assert_allclose(store.load_weights("opt-1"), weights)


def test_cache_lookup(tmp_path):
    store = OptimizationStore(tmp_path)
    store.save(opt_id="opt-1", cache_key="ck-1",
               weights=np.ones(2, dtype=np.float32),
               dose=np.ones((1, 2, 2), dtype=np.float32),
               spacing_mm=(1, 1, 1), result=_result("opt-1", "ck-1", store))
    assert store.lookup_by_cache_key("ck-1").optimization_id == "opt-1"
    assert store.lookup_by_cache_key("missing") is None


def test_checkpoint_write_and_prune(tmp_path):
    store = OptimizationStore(tmp_path)
    store.prepare("opt-1")
    for it in (2, 4, 6, 8):
        cp = store.write_checkpoint("opt-1", it, np.full(3, it, dtype=np.float32), cost=float(it))
        assert np.allclose(np.load(cp.weights_uri), it)
    # Keep newest 2 → evict iter_2, iter_4.
    evicted = store.prune_checkpoints("opt-1", keep=2)
    assert evicted == 2
    remaining = sorted(p.name for p in (tmp_path / "opt-1" / "checkpoints").glob("iter_*.npy"))
    assert remaining == ["iter_6.npy", "iter_8.npy"]


def test_delete_removes_record_and_index(tmp_path):
    store = OptimizationStore(tmp_path)
    store.save(opt_id="opt-1", cache_key="ck-1",
               weights=np.ones(2, dtype=np.float32),
               dose=np.ones((1, 2, 2), dtype=np.float32),
               spacing_mm=(1, 1, 1), result=_result("opt-1", "ck-1", store))
    assert store.delete_by_id("opt-1") is True
    assert store.get_by_id("opt-1") is None
    assert store.lookup_by_cache_key("ck-1") is None
    assert store.delete_by_id("opt-1") is False


def test_list_ids_excludes_incomplete(tmp_path):
    store = OptimizationStore(tmp_path)
    store.save(opt_id="opt-1", cache_key="ck-1",
               weights=np.ones(2, dtype=np.float32),
               dose=np.ones((1, 2, 2), dtype=np.float32),
               spacing_mm=(1, 1, 1), result=_result("opt-1", "ck-1", store))
    store.prepare("opt-incomplete")  # dir without meta.json
    assert store.list_ids() == ["opt-1"]
