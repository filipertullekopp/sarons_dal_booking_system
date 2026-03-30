"""Phase 3B — Group proximity scorer.

Computes how close a candidate spot is to already-assigned group members.

Formula:
    group_score = 1 / (1 + mean_distance_to_assigned_group_members)

Score interpretation:
    1.0 — directly adjacent (distance = 0)
    0.5 — mean distance = 1 cell
    0.0 — no group, or all members on a different grid

Only members that have already been assigned (earlier in the greedy pass)
contribute to the score.  Future placements are NOT assumed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from saronsdal.models.normalized import Spot
from saronsdal.spatial.distance_engine import (
    CROSS_GRID_DISTANCE,
    spot_to_spot_distance,
)
from saronsdal.spatial.topology_loader import Topology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GroupScore:
    """Group proximity scoring result for one (booking, candidate spot) pair."""
    score: float              # 0.0–1.0
    mean_distance: float      # mean grid-cell distance to contributing members
    member_count: int         # number of same-grid members that contributed
    notes: List[str] = field(default_factory=list)


@dataclass
class GroupContext:
    """Merged preferences from a booking's social cluster, for first-in-cluster seeding.

    When the first member of a social cluster is placed and no co-members are
    yet assigned, GroupContext provides a weighted directional pull toward the
    region preferred by the rest of the cluster.  It replaces the zero group_score
    that the first-in-cluster booking would otherwise receive.

    Weights reflect how many distinct linked co-members expressed each preference.
    A section supported by three members gets a stronger pull than one supported
    by a single member.  Preferences are deduplicated per member (one vote per
    member per dimension regardless of how many times they repeated the preference).

    contributor_sources maps each contributing booking_no to:
        "cluster"  — linked via Phase 2 resolved_groups
        "near_ref" — linked via Phase 2.5 reference_resolutions
        "both"     — appears in both link sources
    """
    section_weights: Dict[str, int]     # section → supporter count
    row_weights: Dict[str, int]         # row letter → supporter count
    spot_id_weights: Dict[str, int]     # spot_id → supporter count
    contributing_booking_nos: List[str] # co-members that contributed ≥1 preference
    contributor_sources: Dict[str, str] # booking_no → "cluster"|"near_ref"|"both"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score_group_proximity(
    spot: Spot,
    topo: Topology,
    group_spots: Optional[List[Tuple[str, str]]],
) -> GroupScore:
    """Score a candidate spot's proximity to already-assigned group members.

    Args:
        spot:        Candidate spot being evaluated.
        topo:        Loaded Topology for distance look-ups.
        group_spots: (section, spot_id) pairs of group members already
                     assigned in the current greedy pass.  None or empty
                     list means no group members have been placed yet.

    Returns:
        GroupScore.  score=0.0 when there is no group or all placed members
        are on a different topology grid.
    """
    if not group_spots:
        return GroupScore(
            score=0.0,
            mean_distance=CROSS_GRID_DISTANCE,
            member_count=0,
            notes=["no group members assigned yet"],
        )

    same_grid_dists: List[float] = []
    cross_grid_count = 0

    for g_sec, g_sid in group_spots:
        try:
            d = spot_to_spot_distance(topo, spot.section, spot.spot_id, g_sec, g_sid)
        except KeyError:
            # Either this candidate or the group member is absent from topology
            cross_grid_count += 1
            continue

        if d >= CROSS_GRID_DISTANCE:
            cross_grid_count += 1
        else:
            same_grid_dists.append(d)

    if not same_grid_dists:
        return GroupScore(
            score=0.0,
            mean_distance=CROSS_GRID_DISTANCE,
            member_count=0,
            notes=[
                f"all {len(group_spots)} group member(s) are on a different grid"
                if cross_grid_count
                else "group members not found in topology"
            ],
        )

    mean_d = sum(same_grid_dists) / len(same_grid_dists)
    score = 1.0 / (1.0 + mean_d)

    return GroupScore(
        score=round(score, 4),
        mean_distance=round(mean_d, 2),
        member_count=len(same_grid_dists),
        notes=[
            f"mean distance to {len(same_grid_dists)} group member(s): {mean_d:.1f} cells"
            + (f" ({cross_grid_count} cross-grid member(s) excluded)" if cross_grid_count else "")
        ],
    )


# ---------------------------------------------------------------------------
# Group-context seeding (first-in-cluster)
# ---------------------------------------------------------------------------


def compute_seed_score(spot: Spot, group_context: GroupContext) -> GroupScore:
    """Score a candidate spot using pre-computed group context (first-in-cluster seeding).

    Used when a booking is the first member of its social cluster to be placed,
    so no co-members have an assigned spot yet.  Instead of returning a zero
    group score, this function computes a soft directional score from the merged
    preferences of all linked co-members.

    Scoring:
        Each non-empty preference dimension contributes one component in [0,1]:

            section_component = section_weights.get(spot.section, 0)
                                 / sum(section_weights.values())

            row_component     = row_weights.get(spot.row, 0)
                                 / sum(row_weights.values())

            spot_component    = spot_id_weights.get(spot.spot_id, 0)
                                 / sum(spot_id_weights.values())

        The final seed score is the mean of all non-empty dimension components.
        Dimensions with no weight data are excluded from the mean rather than
        contributing a zero, so a context with only section data is not penalised
        for the absence of row or spot preferences.

    Sentinel fields on the returned GroupScore:
        mean_distance = 0.0   — distinguishes seeded results from proximity results
                                (proximity always uses CROSS_GRID_DISTANCE when 0.0)
        member_count  = 0     — no assigned members were used (seeded, not proximity)

    Args:
        spot:          Candidate spot being evaluated.
        group_context: Merged co-member preferences built before the allocation loop.

    Returns:
        GroupScore with seeded score and human-readable notes for each dimension.
    """
    components: List[float] = []
    notes: List[str] = [
        f"seeded from {len(group_context.contributing_booking_nos)} co-member(s) "
        f"({', '.join(sorted(group_context.contributor_sources.values(), reverse=True)[:3])})"
        if group_context.contributing_booking_nos else
        "seeded: context has no contributing members"
    ]

    # Section dimension
    if group_context.section_weights:
        total_sec = sum(group_context.section_weights.values())
        match_sec = group_context.section_weights.get(spot.section, 0)
        components.append(match_sec / total_sec)
        if match_sec:
            notes.append(
                f"seeded section '{spot.section}': {match_sec}/{total_sec} vote(s)"
            )
        else:
            top = sorted(
                group_context.section_weights, key=lambda s: -group_context.section_weights[s]
            )[:2]
            notes.append(
                f"seeded section: no match (group prefers {top})"
            )

    # Row dimension
    if group_context.row_weights:
        total_row = sum(group_context.row_weights.values())
        match_row = group_context.row_weights.get(spot.row, 0)
        components.append(match_row / total_row)
        if match_row:
            notes.append(
                f"seeded row '{spot.row}': {match_row}/{total_row} vote(s)"
            )

    # Spot-ID dimension
    if group_context.spot_id_weights:
        total_sid = sum(group_context.spot_id_weights.values())
        match_sid = group_context.spot_id_weights.get(spot.spot_id, 0)
        components.append(match_sid / total_sid)
        if match_sid:
            notes.append(
                f"seeded spot '{spot.spot_id}': {match_sid}/{total_sid} vote(s)"
            )

    seed_score = sum(components) / len(components) if components else 0.0

    return GroupScore(
        score=round(seed_score, 4),
        mean_distance=0.0,   # sentinel: seeded result, not proximity-based
        member_count=0,
        notes=notes,
    )
