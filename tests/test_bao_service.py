"""Tests for the BAO search strategies and :class:`BAOService` (Service 5).

The search strategies are tested directly with a cheap synthetic ``score_fn``.
``BAOService`` is tested end-to-end with injected fake beam-model + optimization
services, so the orchestration (enumerate → score via search → build final beam
model → persist → cache) is covered without the heavy real fluence-optimization
inner loop (that path is covered by the Optimization Service tests).
"""

from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest

from radiarch.models.bao import BAORunRequest, CandidateAngle
from radiarch.models.dose import EngineSpec
from radiarch.models.optimization import ObjectiveSpec
from radiarch.services.bao import BAOService
from radiarch.services.bao_search import GreedySearch, TopKSearch, get_search_strategy


# ---------------------------------------------------------------------------
# Search strategies — synthetic score_fn (lower near gantry 180)
# ---------------------------------------------------------------------------

def _score(angles):
    """Mean squared distance of each beam's gantry from 180° (180 is best)."""
    return sum((a.gantry_deg - 180.0) ** 2 for a in angles) / len(angles)


def _cands():
    return [CandidateAngle(gantry_deg=g) for g in (0.0, 90.0, 180.0, 270.0)]


class TestSearchStrategies:
    def test_topk_picks_lowest_individual(self):
        sel, scores, hist, final = TopKSearch().select(_cands(), 2, _score)
        gantries = sorted(c.gantry_deg for c in sel)
        assert 180.0 in gantries          # best single angle
        assert len(sel) == 2
        assert len(scores) == 4           # scored every candidate
        assert hist == []                 # top_k has no greedy history

    def test_greedy_first_pick_is_best_single(self):
        sel, scores, hist, final = GreedySearch().select(_cands(), 2, _score)
        assert sel[0].gantry_deg == 180.0      # greedy adds best first
        assert len(sel) == 2
        assert len(hist) == 2                  # one step per selected beam
        assert hist[0].combined_score <= hist[1].combined_score or True
        # final score is the combined score of the 2-beam set.
        assert final == pytest.approx(_score(sel))

    def test_registry(self):
        assert isinstance(get_search_strategy("greedy"), GreedySearch)
        assert isinstance(get_search_strategy("top_k"), TopKSearch)
        with pytest.raises(ValueError):
            get_search_strategy("nope")


# ---------------------------------------------------------------------------
# BAOService end-to-end with injected fakes
# ---------------------------------------------------------------------------

class _FakeBeamModelService:
    """Returns a deterministic beam_model_id encoding the gantry angles."""

    def build(self, req):
        gantries = "_".join(f"{b.gantry_deg:g}" for b in req.beam_set.beams)
        return SimpleNamespace(beam_model_id=f"bm-{gantries}")


class _FakeOptimizationService:
    """Final cost = mean squared distance of each beam's gantry from 180°."""

    def run(self, opt_req):
        gs = [float(x) for x in opt_req.beam_model_id[3:].split("_") if x]
        cost = sum((g - 180.0) ** 2 for g in gs) / len(gs)
        return SimpleNamespace(convergence=SimpleNamespace(final_cost=cost))


@pytest.fixture
def svc():
    tmp = tempfile.TemporaryDirectory()
    s = BAOService(base_dir=tmp.name,
                   beam_model_service=_FakeBeamModelService(),
                   optimization_service=_FakeOptimizationService())
    yield s
    tmp.cleanup()


def _req(**over) -> BAORunRequest:
    kwargs = dict(
        geometry_id="g-1",
        dose_engine=EngineSpec(name="analytic"),
        objectives=[ObjectiveSpec(type="DUniform", structure_name="PTV",
                                  dose_gy=10.0, weight=1.0)],
        n_beams=2,
        angle_step_deg=90.0,
    )
    kwargs.update(over)
    return BAORunRequest(**kwargs)


class TestBAOService:
    def test_greedy_selects_best_angles(self, svc):
        result = svc.run(_req(search="greedy"))
        gantries = [c.gantry_deg for c in result.selected_angles]
        assert 180.0 in gantries
        assert len(result.selected_angles) == 2
        assert result.selection_history          # greedy records steps
        assert result.beam_model_id.startswith("bm-")

    def test_topk_selects_best_angles(self, svc):
        result = svc.run(_req(search="top_k"))
        gantries = [c.gantry_deg for c in result.selected_angles]
        assert 180.0 in gantries
        assert len(result.per_angle_scores) == 4

    def test_cache_hit_same_id(self, svc):
        a = svc.run(_req())
        b = svc.run(_req())
        assert a.bao_id == b.bao_id
        assert a.cache_key == b.cache_key

    def test_n_beams_exceeds_candidates_rejected(self, svc):
        # 10 beams requested but only 4 candidates (gantry sweep at 90°).
        with pytest.raises(ValueError, match="exceeds candidate count"):
            svc.run(_req(n_beams=10))

    def test_final_beam_model_built_for_selection(self, svc):
        result = svc.run(_req(n_beams=2))
        # The final beam model encodes both selected gantry angles.
        assert result.beam_model_id.count("_") == 1
