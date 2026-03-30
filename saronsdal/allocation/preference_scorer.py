"""Phase 3 — Preference scorer.

Scores a candidate (Spot, Booking) pair as a float in [0.0, 1.0].
Higher is better.  Hard constraint violations set total_score = 0.0.

Score components:
  requested_spot      — spot was specifically requested by booking
  preferred_row       — spot is in a preferred row/subsection
  group_proximity     — close to already-assigned group members
  near_toilet         — Phase 2.5: near_toilet preference
  near_bibelskolen    — Phase 2.5: near Internatet section
  near_hall           — Phase 2.5: near Bedehuset section
  near_forest         — Phase 2.5: Furutoppen/Furulunden proxy
  avoid_river         — Phase 2.5: maximize distance to river
  avoid_noise         — Phase 2.5: maximize distance from roads
  flat_ground         — spot.hilliness = 0
  quiet_spot          — Phase 2.5: distance from road
  extra_space         — Phase 2.5: spot.length_m preference
  accessibility       — Phase 2.5: flat + near road
  inferred_section    — Phase 2.5: Gemini-inferred section match

Hard constraints (violations → total_score = 0.0):
  wrong_section            — booking.request.preferred_sections set and spot not in it
  no_motorhome             — spot.no_motorhome and booking is motorhome
  no_caravan_nor_motorhome — spot.no_caravan_nor_motorhome and booking is caravan/motorhome
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from saronsdal.models.normalized import Booking, Spot
from saronsdal.spatial.distance_engine import (
    CROSS_GRID_DISTANCE,
    spot_to_landmark_distance,
)
from saronsdal.spatial.topology_loader import Topology

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Normalisation constants
# ---------------------------------------------------------------------------

#: Distances beyond this are treated as "very far" (scores 1.0 for avoidance,
#: 0.0 for proximity).  Based on typical maximum grid span (~50 cells).
_DIST_MAX: float = 50.0

#: Hilliness cap for flat_ground scoring (hilliness >= this → score 0.0).
_HILL_MAX: int = 5

#: When scoring accessibility, how close to a road is "ideal" (cells).
_ACCESS_ROAD_IDEAL: float = 3.0

#: Spot length considered "long enough" for extra_space scoring (metres).
_EXTRA_SPACE_LEN_MAX: float = 15.0

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SpotScore:
    """Scoring result for one (booking, spot) candidate pair."""
    section: str
    spot_id: str
    total_score: float                  # 0.0–1.0; 0.0 means hard violation
    components: Dict[str, float]        # named sub-scores, each 0.0–1.0
    violations: List[str]               # hard constraint names that triggered
    notes: List[str]                    # human-readable explanations


# ---------------------------------------------------------------------------
# Score primitives
# ---------------------------------------------------------------------------


def _near(distance: float, max_dist: float = _DIST_MAX) -> float:
    """Score for "be near X": 1.0 at distance 0, 0.0 at distance >= max_dist."""
    if distance >= CROSS_GRID_DISTANCE:
        return 0.0
    return max(0.0, 1.0 - distance / max_dist)


def _far(distance: float, max_dist: float = _DIST_MAX) -> float:
    """Score for "be far from X": 0.0 at distance 0, 1.0 at distance >= max_dist."""
    if distance >= CROSS_GRID_DISTANCE:
        return 1.0   # different grid = effectively very far = good for avoidance
    return min(1.0, distance / max_dist)


def _landmark_dist(
    topo: Topology,
    section: str,
    spot_id: str,
    lm_type: str,
) -> float:
    try:
        return spot_to_landmark_distance(topo, section, spot_id, lm_type)
    except KeyError:
        return CROSS_GRID_DISTANCE


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


def score_spot(
    spot: Spot,
    booking: Booking,
    topo: Topology,
    prefs=None,                                         # StructuredPreferences | None
    preferred_spot_ids: Optional[List[str]] = None,
    preferred_rows: Optional[List[str]] = None,
) -> SpotScore:
    """Score a candidate spot for a booking.

    Args:
        spot:               Candidate spot (from spots.csv).
        booking:            The booking being placed.
        topo:               Loaded Topology for distance calculations.
        prefs:              StructuredPreferences from Phase 2.5 (optional).
        group_spots:        (section, spot_id) of already-assigned group members.
        preferred_spot_ids: Explicit spot IDs the booking requested.
        preferred_rows:     Row letters the booking preferred (e.g. ['D', 'E']).

    Returns:
        SpotScore.  If hard constraints are violated, total_score = 0.0.
    """
    section = spot.section
    spot_id = spot.spot_id
    components: Dict[str, float] = {}
    violations: List[str] = []
    notes: List[str] = []

    # ------------------------------------------------------------------
    # Hard constraints
    # ------------------------------------------------------------------
    vehicle = booking.vehicle
    if spot.no_caravan_nor_motorhome and vehicle.vehicle_type in ("caravan", "motorhome"):
        violations.append("no_caravan_nor_motorhome")
    if spot.no_motorhome and vehicle.vehicle_type == "motorhome":
        violations.append("no_motorhome")

    # Hard length feasibility: vehicle body must fit within this spot.
    # When scoring triplet candidates, spot is always the first (anchor) spot.
    # Fortelt/awning width does NOT contribute to body length — do not add it.
    if (
        vehicle.body_length_m is not None
        and vehicle.body_length_m > 0
        and spot.length_m > 0
        and vehicle.body_length_m > spot.length_m
    ):
        violations.append("vehicle_too_long_for_spot")

    request = booking.request
    if request.preferred_sections and section not in request.preferred_sections:
        violations.append("wrong_section")

    if violations:
        return SpotScore(
            section=section,
            spot_id=spot_id,
            total_score=0.0,
            components={},
            violations=violations,
            notes=[f"Hard violation: {', '.join(violations)}"],
        )

    # ------------------------------------------------------------------
    # Soft scores
    # ------------------------------------------------------------------

    # 1. Specific spot ID requested
    if preferred_spot_ids:
        if spot_id in preferred_spot_ids:
            components["requested_spot"] = 1.0
            notes.append(f"Exact spot {spot_id} was requested")
        else:
            components["requested_spot"] = 0.0

    # 2. Row preference
    if preferred_rows:
        if spot.row in preferred_rows:
            components["preferred_row"] = 1.0
        else:
            components["preferred_row"] = 0.1
            notes.append(f"Spot row {spot.row!r} not in preferred {preferred_rows}")

    # 3. Phase 2.5 structured preferences
    if prefs is not None:
        _apply_structured_prefs(prefs, spot, section, spot_id, topo,
                                components, notes)

    # 4. Baseline: flat ground (always scored, low weight)
    if "flat_ground" not in components:
        components["flat_ground_base"] = max(0.0, 1.0 - spot.hilliness / _HILL_MAX)

    # ------------------------------------------------------------------
    # Total score: simple mean of all components (equal weight)
    # ------------------------------------------------------------------
    if not components:
        total = 0.5   # neutral when no preferences specified
    else:
        total = sum(components.values()) / len(components)

    return SpotScore(
        section=section,
        spot_id=spot_id,
        total_score=round(total, 4),
        components=components,
        violations=[],
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Structured preference application
# ---------------------------------------------------------------------------


def _apply_structured_prefs(
    prefs,           # StructuredPreferences
    spot: Spot,
    section: str,
    spot_id: str,
    topo: Topology,
    components: Dict[str, float],
    notes: List[str],
) -> None:
    """Apply all non-None/non-False StructuredPreferences to the component dict."""

    if prefs.near_toilet:
        d = _landmark_dist(topo, section, spot_id, "toilet")
        components["near_toilet"] = _near(d)
        notes.append(f"near_toilet: dist={d:.1f}")

    if prefs.near_bibelskolen:
        # Bibelskolen is the Internatet section — favour that section
        if section == "Internatet":
            components["near_bibelskolen"] = 1.0
        else:
            components["near_bibelskolen"] = 0.0
            notes.append("near_bibelskolen: spot not in Internatet section")

    if prefs.near_hall:
        # The main hall is the Bedehuset — favour that section
        if section == "Bedehuset":
            components["near_hall"] = 1.0
        else:
            components["near_hall"] = 0.0
            notes.append("near_hall: spot not in Bedehuset section")

    if prefs.near_forest:
        # Furutoppen and Furulunden are the forested sections
        if section in ("Furutoppen", "Furulunden"):
            components["near_forest"] = 1.0
        else:
            components["near_forest"] = 0.3
            notes.append(f"near_forest: {section} is not a forest section")

    if prefs.avoid_river:
        d = _landmark_dist(topo, section, spot_id, "river")
        components["avoid_river"] = _far(d)
        notes.append(f"avoid_river: dist={d:.1f}")

    if prefs.avoid_noise:
        # Proxy: distance from any road
        d_road = _landmark_dist(topo, section, spot_id, "road")
        d_main = _landmark_dist(topo, section, spot_id, "main_road")
        d = min(d_road, d_main)
        components["avoid_noise"] = _far(d)
        notes.append(f"avoid_noise: nearest road dist={d:.1f}")

    if prefs.flat_ground:
        components["flat_ground"] = max(0.0, 1.0 - spot.hilliness / _HILL_MAX)

    if prefs.quiet_spot:
        d_road = min(
            _landmark_dist(topo, section, spot_id, "road"),
            _landmark_dist(topo, section, spot_id, "main_road"),
        )
        components["quiet_spot"] = _far(d_road)
        notes.append(f"quiet_spot: nearest road dist={d_road:.1f}")

    if prefs.terrain_pref:
        # Terrain direction (uphill/downhill) not in spot metadata — neutral
        components["terrain_pref"] = 0.5
        notes.append(f"terrain_pref={prefs.terrain_pref!r}: no terrain-direction data")

    if prefs.extra_space:
        if spot.length_m and spot.length_m > 0:
            components["extra_space"] = min(1.0, spot.length_m / _EXTRA_SPACE_LEN_MAX)
        else:
            components["extra_space"] = 0.5
            notes.append("extra_space: spot length unknown")

    if prefs.accessibility:
        hill = max(0.0, 1.0 - spot.hilliness / _HILL_MAX)
        d_road = _landmark_dist(topo, section, spot_id, "road")
        road_prox = _near(d_road, max_dist=_ACCESS_ROAD_IDEAL * 3)
        components["accessibility"] = (hill + road_prox) / 2
        notes.append(f"accessibility: hilliness={spot.hilliness}, road dist={d_road:.1f}")

    if prefs.drainage_concern:
        # No drainage data in metadata; neutral score
        components["drainage_concern"] = 0.5
        notes.append("drainage_concern: no drainage data available")

    if prefs.same_as_last_year:
        # Cannot score without historical assignment data
        notes.append("same_as_last_year: no historical data — cannot score")

    if prefs.inferred_section:
        components["inferred_section"] = 1.0 if section == prefs.inferred_section else 0.0
        notes.append(f"inferred_section={prefs.inferred_section!r}")
