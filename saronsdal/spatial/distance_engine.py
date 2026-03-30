"""Phase 3 — Distance engine.

Straight-line (Euclidean) distance between spots and landmarks using the
coordinates stored in a Topology object.

Distance rules (from the spec):
  - Every cell = 1 unit of space.
  - EMPTY cells count as distance.
  - No pathfinding, no obstacle avoidance — pure Euclidean geometry.
  - Spots on different grids cannot be compared meaningfully; those calls
    return CROSS_GRID_DISTANCE (a large sentinel value).

Typical grid spans:
  Elvebredden   ~8 columns × ~78 rows
  Furulunden    ~12 columns × ~46 rows
  Fjellterrassen ~3 columns × ~69 rows
  Furutoppen    ~2 columns × ~33 rows
  Bibelskolen   ~5 columns × ~34 rows
  Bedehuset     ~6 columns × ~50 rows
  Vårdalen/EG   ~45 columns × ~70 rows
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from saronsdal.spatial.topology_loader import GridCoord, Topology

#: Returned when spots are on different grids (no shared coordinate space).
CROSS_GRID_DISTANCE: float = 10_000.0


# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------


def euclidean(c1: GridCoord, c2: GridCoord) -> float:
    """Euclidean distance between two GridCoords.

    Returns CROSS_GRID_DISTANCE when they are in different grids.
    """
    if c1.grid != c2.grid:
        return CROSS_GRID_DISTANCE
    return math.sqrt((c2.x - c1.x) ** 2 + (c2.y - c1.y) ** 2)


# ---------------------------------------------------------------------------
# Spot ↔ spot
# ---------------------------------------------------------------------------


def spot_to_spot_distance(
    topo: Topology,
    section1: str, spot_id1: str,
    section2: str, spot_id2: str,
) -> float:
    """Euclidean grid distance between two spots.

    Args:
        topo:                Loaded Topology.
        section1, spot_id1:  First spot key.
        section2, spot_id2:  Second spot key.

    Returns:
        Euclidean distance in grid cells, or CROSS_GRID_DISTANCE when
        the spots are on different grids.

    Raises:
        KeyError: if either spot key is absent from the topology.
    """
    c1 = topo.get_spot_coord(section1, spot_id1)
    c2 = topo.get_spot_coord(section2, spot_id2)
    if c1 is None:
        raise KeyError(f"Spot not in topology: ({section1!r}, {spot_id1!r})")
    if c2 is None:
        raise KeyError(f"Spot not in topology: ({section2!r}, {spot_id2!r})")
    return euclidean(c1, c2)


# ---------------------------------------------------------------------------
# Spot ↔ landmark
# ---------------------------------------------------------------------------


def spot_to_landmark_distance(
    topo: Topology,
    section: str, spot_id: str,
    landmark_type: str,
) -> float:
    """Minimum Euclidean distance from a spot to any landmark of the given type.

    Only landmarks on the same grid as the spot are considered.  If no
    same-grid landmark of that type exists, CROSS_GRID_DISTANCE is returned.

    Args:
        topo:          Loaded Topology.
        section, spot_id: Spot key.
        landmark_type: Canonical landmark type (e.g. "toilet", "river").

    Raises:
        KeyError: if the spot is absent from the topology.
    """
    c = topo.get_spot_coord(section, spot_id)
    if c is None:
        raise KeyError(f"Spot not in topology: ({section!r}, {spot_id!r})")

    candidates = [
        lm for lm in topo.get_landmark_coords(landmark_type)
        if lm.grid == c.grid
    ]
    if not candidates:
        return CROSS_GRID_DISTANCE

    return min(euclidean(c, lm) for lm in candidates)


def nearest_landmark_coord(
    topo: Topology,
    section: str, spot_id: str,
    landmark_type: str,
) -> Optional[GridCoord]:
    """Return the GridCoord of the nearest same-grid landmark, or None."""
    c = topo.get_spot_coord(section, spot_id)
    if c is None:
        return None
    candidates = [
        lm for lm in topo.get_landmark_coords(landmark_type)
        if lm.grid == c.grid
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda lm: euclidean(c, lm))


# ---------------------------------------------------------------------------
# Group helpers
# ---------------------------------------------------------------------------


def pairwise_distances(
    topo: Topology,
    spot_keys: List[Tuple[str, str]],
) -> List[float]:
    """All pairwise Euclidean distances among a list of (section, spot_id) keys.

    Skips pairs where a spot is missing from the topology.
    Returns an empty list when fewer than 2 valid spots are provided.
    """
    coords = []
    for section, spot_id in spot_keys:
        c = topo.get_spot_coord(section, spot_id)
        if c is not None:
            coords.append(c)

    if len(coords) < 2:
        return []

    dists = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dists.append(euclidean(coords[i], coords[j]))
    return dists


def mean_group_distance(
    topo: Topology,
    spot_keys: List[Tuple[str, str]],
) -> float:
    """Mean pairwise distance among a group of spots.

    Returns CROSS_GRID_DISTANCE when fewer than 2 spots are in the topology.
    """
    dists = pairwise_distances(topo, spot_keys)
    if not dists:
        return CROSS_GRID_DISTANCE
    return sum(dists) / len(dists)
