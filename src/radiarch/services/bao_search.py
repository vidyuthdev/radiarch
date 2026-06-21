"""Beam-angle subset-selection strategies for the BAO Service (Service 5).

A *search strategy* picks the best ``n_beams`` subset from a candidate angle
set, given a black-box ``score_fn`` that scores an angle *set* (lower is
better). Decoupling the search from the scoring keeps both unit-testable: the
search strategies can be exercised with a cheap synthetic ``score_fn``, and the
real scorer (build a beam model → fluence-optimize → composite cost) lives in
:class:`BAOService`.

Two strategies for v0.1:

* ``top_k`` — score every candidate individually, keep the ``n_beams`` lowest.
  Cheap (one score per candidate) but ignores beam-to-beam interplay.
* ``greedy`` — forward selection: repeatedly add the candidate whose addition
  most lowers the *combined*-set score. Captures complementarity between beams
  (e.g. avoid two near-parallel fields) at ``O(n_beams · n_candidates)`` scores.
"""

from __future__ import annotations

from typing import Callable, List, Protocol, Tuple, runtime_checkable

from loguru import logger

from ..models.bao import AngleScore, BAOSelectionStep, CandidateAngle

# Scores an angle *set*; lower is better. The BAO service supplies the concrete
# implementation (build beam model → optimize → composite cost).
ScoreFn = Callable[[List[CandidateAngle]], float]


@runtime_checkable
class SearchStrategy(Protocol):
    """Selects ``n_beams`` from ``candidates`` using ``score_fn``."""

    name: str

    def select(
        self,
        candidates: List[CandidateAngle],
        n_beams: int,
        score_fn: ScoreFn,
    ) -> Tuple[List[CandidateAngle], List[AngleScore], List[BAOSelectionStep], float]:
        """Return ``(selected, per_angle_scores, history, final_score)``."""
        ...


class TopKSearch:
    """Score each candidate alone; keep the ``n_beams`` best (lowest score)."""

    name: str = "top_k"

    def select(self, candidates, n_beams, score_fn):
        scores: List[AngleScore] = []
        for c in candidates:
            s = float(score_fn([c]))
            scores.append(AngleScore(gantry_deg=c.gantry_deg,
                                     couch_deg=c.couch_deg, score=s))
            logger.debug("top_k: {} -> {:.6g}", c.key(), s)
        ranked = sorted(zip(candidates, scores), key=lambda t: t[1].score)
        selected = [c for c, _ in ranked[:n_beams]]
        # Final score = the combined score of the selected set (one extra eval),
        # so it's comparable to greedy's final_score.
        final = float(score_fn(selected)) if selected else float("inf")
        return selected, scores, [], final


class GreedySearch:
    """Forward selection: add the candidate that most lowers the combined score."""

    name: str = "greedy"

    def select(self, candidates, n_beams, score_fn):
        remaining = list(candidates)
        selected: List[CandidateAngle] = []
        history: List[BAOSelectionStep] = []
        per_angle: List[AngleScore] = []
        final_score = float("inf")

        n = min(n_beams, len(remaining))
        for step in range(n):
            best_c = None
            best_score = float("inf")
            for c in remaining:
                trial = selected + [c]
                s = float(score_fn(trial))
                if step == 0:
                    # First round doubles as the per-candidate score table.
                    per_angle.append(AngleScore(gantry_deg=c.gantry_deg,
                                                couch_deg=c.couch_deg, score=s))
                if s < best_score:
                    best_score = s
                    best_c = c
            if best_c is None:
                break
            selected.append(best_c)
            remaining.remove(best_c)
            final_score = best_score
            history.append(BAOSelectionStep(
                step=step, added_gantry_deg=best_c.gantry_deg,
                added_couch_deg=best_c.couch_deg, combined_score=best_score,
            ))
            logger.debug("greedy step {}: +{} -> {:.6g}",
                         step, best_c.key(), best_score)
        return selected, per_angle, history, final_score


_STRATEGIES = {"top_k": TopKSearch, "greedy": GreedySearch}


def get_search_strategy(name: str) -> SearchStrategy:
    """Instantiate the search strategy named by ``name`` (422 on unknown)."""
    try:
        cls = _STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown BAO search strategy {name!r}; available: {sorted(_STRATEGIES)}"
        ) from exc
    return cls()


__all__ = [
    "ScoreFn",
    "SearchStrategy",
    "TopKSearch",
    "GreedySearch",
    "get_search_strategy",
]
