"""Phase 3B — Candidate triplet ranker.

Combines preference scores and group proximity scores into a single ranking
for a (booking, candidate triplet list) pair.

Scoring model:
    total_score = W_PREF * preference_score + W_GROUP * group_score

Candidate unit
    Each candidate is a Triplet (one Sirvoy 'Rom').  Scoring is anchored on
    the triplet's first spot: section/row checks, length feasibility, terrain,
    and group-proximity distances all use the first spot as the representative
    position.

The group_score term has two modes:
  - Normal (later cluster members): score_group_proximity() — 1/(1+mean_distance)
    to already-assigned co-members.
  - First-in-cluster seeding: compute_seed_score() — weighted match fraction
    against merged co-member preferences from GroupContext.  Activates only
    when group_spots is empty/None AND a GroupContext is provided.

preferred_spot_ids matching
    A triplet "matches" a preferred_spot_id if ANY of its spot_ids appears in
    the list (not just the first spot).  Internally, if any triplet spot matches,
    the first_spot_id is added to the effective preferred list so that
    score_spot() (which only sees the first spot) registers the hit.

Weights are declared as module-level constants so they are visible, explicit,
and easy to tune without hunting for magic numbers.

Hard constraint violations (wrong section, vehicle type restriction, vehicle
too long for anchor spot) discard the candidate entirely before any weight
arithmetic is applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from saronsdal.allocation.group_scorer import (
    GroupContext,
    GroupScore,
    compute_seed_score,
    score_group_proximity,
)
from saronsdal.allocation.preference_scorer import SpotScore, score_spot
from saronsdal.models.normalized import Booking, Spot
from saronsdal.models.triplet import Triplet
from saronsdal.spatial.topology_loader import Topology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring weights — explicit constants, not magic numbers
# ---------------------------------------------------------------------------

W_PREF: float = 0.7   # weight for preference score  (proximity, terrain, …)
W_GROUP: float = 0.3  # weight for group proximity score
TOP_N: int = 10       # default number of top candidates returned by rank_candidates


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class RankedCandidate:
    """A scored, ranked candidate (triplet, booking) pair."""
    triplet: Triplet
    total_score: float        # W_PREF * preference_score + W_GROUP * group_score
    preference_score: float   # raw output of score_spot() on the anchor spot
    group_score: float        # raw output of score_group_proximity() or compute_seed_score()
    spot_score: SpotScore     # full preference breakdown for inspection / logging
    group_detail: GroupScore  # full group proximity breakdown

    @property
    def spot(self) -> Optional[Spot]:
        """First (anchor) spot of the triplet.

        Backward-compatible accessor: existing code that reads ``r.spot.spot_id``
        continues to work after the single-spot → triplet refactor.
        """
        return self.triplet.first_spot


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rank_candidates(
    booking: Booking,
    candidates: List[Triplet],
    topo: Topology,
    prefs=None,                                          # StructuredPreferences | None
    group_spots: Optional[List[Tuple[str, str]]] = None,
    preferred_spot_ids: Optional[List[str]] = None,
    preferred_rows: Optional[List[str]] = None,
    group_context: Optional[GroupContext] = None,
    top_n: int = TOP_N,
) -> List[RankedCandidate]:
    """Score and rank candidate triplets for a booking.

    Steps:
        1. For each triplet, determine the effective preferred_spot_ids:
           a triplet "matches" if ANY of its spot_ids is in the list.
           The first_spot_id is added to the effective list so that
           score_spot() (which only sees the anchor spot) detects the hit.
        2. Score the anchor spot with score_spot() (preference signals +
           hard constraint check including vehicle length feasibility).
        3. Discard candidates that have any hard violation.
        4. Score remaining candidates with the appropriate group signal:
             a. group_spots non-empty → score_group_proximity() (normal mode)
             b. group_spots empty AND group_context provided → compute_seed_score()
                (first-in-cluster seeding mode)
             c. neither → GroupScore(score=0.0, …)
        5. Compute total_score = W_PREF * pref + W_GROUP * group.
        6. Sort descending by total_score; tie-break by (section, first_spot_id)
           for full determinism.
        7. Return the top_n results (all results when top_n <= 0).

    Args:
        booking:            The booking being placed.
        candidates:         Currently available Triplets to evaluate.
        topo:               Loaded Topology for distance calculations.
        prefs:              StructuredPreferences from Phase 2.5 (optional).
        group_spots:        (section, spot_id) of already-assigned group
                            members; None / empty = no group yet.
        preferred_spot_ids: Explicit spot IDs from booking request + Phase 2.5
                            subsection suggestions (pre-merged by caller).
                            Matched against all spot_ids in each triplet.
        preferred_rows:     Row letters from booking request + Phase 2.5
                            subsection suggestions (pre-merged by caller).
        group_context:      Merged co-member preferences for first-in-cluster
                            seeding.  Ignored when group_spots is non-empty.
        top_n:              Cap on returned list size (0 = unlimited).

    Returns:
        List of RankedCandidate sorted descending by total_score.
        Empty list when all candidates have hard violations or candidates is
        empty.
    """
    ranked: List[RankedCandidate] = []

    # Determine group scoring mode once, outside the per-triplet loop
    use_seeding = bool(not group_spots and group_context is not None)

    for triplet in candidates:
        anchor = triplet.first_spot
        if anchor is None:
            # Should not happen for is_allocatable triplets, but guard anyway
            continue

        # Step 1: effective preferred_spot_ids for this triplet.
        # A triplet "matches" if any of its spot_ids was explicitly requested.
        # If so, add first_spot_id to the list so score_spot() sees the hit.
        eff_preferred_ids = preferred_spot_ids
        if preferred_spot_ids and any(
            sid in preferred_spot_ids for sid in triplet.spot_ids
        ):
            if triplet.first_spot_id not in preferred_spot_ids:
                eff_preferred_ids = list(preferred_spot_ids) + [triplet.first_spot_id]

        # Step 2 & 3: score anchor spot + hard constraint check
        spot_sc = score_spot(
            spot=anchor,
            booking=booking,
            topo=topo,
            prefs=prefs,
            preferred_spot_ids=eff_preferred_ids,
            preferred_rows=preferred_rows,
        )
        if spot_sc.violations:
            continue  # hard violation — discard this triplet

        # Step 4: group score — proximity (normal) or context seeding (first-in-cluster)
        if use_seeding:
            grp_sc = compute_seed_score(anchor, group_context)
        else:
            grp_sc = score_group_proximity(anchor, topo, group_spots)

        # Step 5: combined score
        total = W_PREF * spot_sc.total_score + W_GROUP * grp_sc.score

        ranked.append(RankedCandidate(
            triplet=triplet,
            total_score=round(total, 4),
            preference_score=spot_sc.total_score,
            group_score=grp_sc.score,
            spot_score=spot_sc,
            group_detail=grp_sc,
        ))

    # Step 6: sort descending; tie-break by (section, first_spot_id) for determinism
    ranked.sort(
        key=lambda r: (-r.total_score, r.triplet.section, r.triplet.first_spot_id)
    )

    if top_n > 0:
        ranked = ranked[:top_n]

    logger.debug(
        "Booking %s: %d/%d triplets valid, top score=%.4f",
        booking.booking_no,
        len(ranked),
        len(candidates),
        ranked[0].total_score if ranked else 0.0,
    )
    return ranked
