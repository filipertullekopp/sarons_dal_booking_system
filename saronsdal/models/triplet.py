"""Triplet model — one Sirvoy 'Rom' (room) as an allocation unit.

A Triplet represents exactly one entry from sirvoy_room_ids.csv.  In the
current Sarons Dal campsite every standard camping 'Rom' spans exactly
three consecutive 3m-wide spots in the same row (e.g. "Furulunden A16-18").

The first spot of the triplet is the physically significant one:
  - its length_m is the feasibility gate for vehicle body length
  - it serves as the topology anchor for group proximity calculations
  - it carries equipment restriction flags (no_motorhome, etc.)

Non-standard entries (2-spot specials, single spots, tent-only ranges of
4+ spots) are parsed but flagged; they are excluded from the active
allocation pool via the is_allocatable property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from saronsdal.models.normalized import Spot


@dataclass
class Triplet:
    """One Sirvoy 'Rom' row, fully resolved against spots.csv data.

    Fields
    ------
    room_id
        Exact string from the 'Rom' column of sirvoy_room_ids.csv.
        This is the authoritative identifier used in the final output.
        Example: ``"Furulunden A16-18 (5m)"``
    section
        Canonical section name parsed from room_id.  Example: ``"Furulunden"``
    row
        Row letter parsed from room_id.  Example: ``"A"``
    spot_ids
        Ordered list of individual spot IDs contained in this room.
        For standard triplets this is exactly three entries, e.g.
        ``["A16", "A17", "A18"]``.
    first_spot
        The Spot object for the first (anchor) spot, looked up from
        spots.csv data.  ``None`` when the first spot is absent from the
        loaded spot data — flagged as ``"first_spot_missing"``.
    review_flags
        Human-readable flags describing why this room is non-standard or
        unresolvable.  Common values:
            ``"non_triplet_spot_count_N"`` — room has N spots (not 3)
            ``"first_spot_missing"``        — first spot not in spots.csv
    """

    room_id: str
    section: str
    row: str
    spot_ids: List[str]
    first_spot: Optional[Spot]
    review_flags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived accessors
    # ------------------------------------------------------------------

    @property
    def first_spot_id(self) -> str:
        """Spot ID of the anchor (first) spot, e.g. ``"A16"``."""
        return self.spot_ids[0]

    @property
    def first_spot_length_m(self) -> float:
        """Length of the anchor spot in metres.

        Used as the hard feasibility threshold for vehicle body length.
        Returns ``0.0`` when ``first_spot`` is None (unknown length).
        """
        return self.first_spot.length_m if self.first_spot else 0.0

    @property
    def is_allocatable(self) -> bool:
        """True only for standard 3-spot rooms with a resolved first spot.

        Non-standard rooms (2-spot, 4-spot, single-spot, missing anchor)
        are kept in the parsed catalogue but excluded from the candidate
        pool at allocation time.
        """
        return len(self.spot_ids) == 3 and self.first_spot is not None
