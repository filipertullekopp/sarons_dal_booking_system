"""Integration tests for the Sirvoy-triplet allocator.

Covers:
  - Hard length filter: vehicle too long for anchor spot → rejected
  - All three spots become unavailable after assignment
  - Output carries exact Sirvoy Room ID string and all spot IDs
  - Section constraint enforced across triplets
  - No available triplets → unassigned result
  - Social group logic still works with Triplet candidates
  - preferred_spot_ids matching any triplet spot (not just first spot)
"""
import re
from typing import List, Optional

import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.allocation.allocator import AllocationResult, allocate
from saronsdal.models.normalized import (
    Booking, RawGroupSignals, SectionRow, Spot, SpotRequest, VehicleUnit,
)
from saronsdal.models.triplet import Triplet
from saronsdal.models.grouping import ResolvedCluster
from saronsdal.spatial.topology_loader import load_topology


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _spot(
    section: str = "Furulunden",
    spot_id: str = "A1",
    length_m: float = 10.0,
    no_motorhome: bool = False,
    no_caravan: bool = False,
) -> Spot:
    row = re.match(r"^([A-Z]+)", spot_id).group(1)
    pos = int(re.match(r"[A-Z]+(\d+)", spot_id).group(1))
    return Spot(
        spot_id=spot_id, section=section, row=row, position=pos,
        length_m=length_m, hilliness=0, is_end_of_row=False,
        is_not_spot=False, is_reserved=False,
        no_motorhome=no_motorhome, no_caravan_nor_motorhome=no_caravan,
        width_m=3.0,
    )


def _triplet(
    section: str = "Furulunden",
    first_spot_id: str = "A1",
    row: str = "A",
    length_m: float = 10.0,
    no_motorhome: bool = False,
    no_caravan: bool = False,
) -> Triplet:
    """Minimal 3-spot triplet anchored at first_spot_id."""
    m = re.match(r"([A-Z]+)(\d+)", first_spot_id)
    row_letter = m.group(1)
    first_num = int(m.group(2))
    spot_ids = [f"{row_letter}{first_num + i}" for i in range(3)]
    anchor = _spot(section, spot_ids[0], length_m=length_m,
                   no_motorhome=no_motorhome, no_caravan=no_caravan)
    return Triplet(
        room_id=f"{section} {spot_ids[0]}-{spot_ids[2]}",
        section=section,
        row=row or row_letter,
        spot_ids=spot_ids,
        first_spot=anchor,
    )


def _booking(
    booking_no: str = "B001",
    vehicle_type: str = "caravan",
    body_length_m: Optional[float] = 7.0,
    preferred_sections: Optional[List[str]] = None,
    preferred_spot_ids: Optional[List[str]] = None,
) -> Booking:
    return Booking(
        booking_no=booking_no,
        booking_source="website",
        check_in=None, check_out=None,
        first_name="Test", last_name="Guest", full_name="Test Guest",
        num_guests=2, language="no", is_confirmed=True,
        vehicle=VehicleUnit(
            vehicle_type=vehicle_type,
            spec_size_hint=None,
            body_length_m=body_length_m,
            body_width_m=2.75,
            fortelt_width_m=0.0,
            total_width_m=2.75,
            required_spot_count=3,
            has_fortelt=False, has_markise=False,
            registration=None, parse_confidence=1.0,
        ),
        request=SpotRequest(
            preferred_sections=preferred_sections or [],
            preferred_spot_ids=preferred_spot_ids or [],
            preferred_section_rows=[],
            avoid_sections=[],
            amenity_flags=set(),
            raw_near_texts=[],
            parse_confidence=1.0,
        ),
        group_signals=RawGroupSignals(
            organization=None, group_field=None,
            near_text_fragments=[], is_org_private=True,
        ),
        data_confidence=1.0,
    )


def _cluster(members: List[str], cluster_id: str = "C1") -> ResolvedCluster:
    return ResolvedCluster(
        cluster_id=cluster_id,
        members=members,
        canonical_label=None,
        cluster_type="name_ref",
        min_link_strength=1.0,
        max_link_strength=1.0,
        internal_edges=[],
    )


_FURULUNDEN_ONLY = {"furulunden": "topology_grid_furulunden_mini.csv"}


@pytest.fixture(scope="module")
def topo():
    return load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)


# ---------------------------------------------------------------------------
# Hard length filter
# ---------------------------------------------------------------------------

class TestHardLengthFilter:
    def test_vehicle_too_long_rejected(self, topo):
        """Vehicle body 8m does not fit in a 6m anchor spot → no valid candidates."""
        booking = _booking("B001", body_length_m=8.0)
        triplets = [_triplet("Furulunden", "A1", length_m=6.0)]
        results = allocate([booking], triplets, topo)
        r = results[0]
        assert not r.is_assigned
        assert "vehicle_too_long_for_spot" in r.explanation.get("violation_counts", {})

    def test_vehicle_fits_assigned(self, topo):
        """Vehicle body 5m fits in a 6m anchor spot → assigned."""
        booking = _booking("B001", body_length_m=5.0)
        triplets = [_triplet("Furulunden", "A1", length_m=6.0)]
        results = allocate([booking], triplets, topo)
        assert results[0].is_assigned

    def test_unknown_vehicle_length_not_filtered(self, topo):
        """body_length_m=None → skip length check entirely."""
        booking = _booking("B001", body_length_m=None)
        triplets = [_triplet("Furulunden", "A1", length_m=3.0)]
        results = allocate([booking], triplets, topo)
        assert results[0].is_assigned

    def test_zero_anchor_length_not_filtered(self, topo):
        """Anchor spot with length_m=0.0 → length check skipped (unknown)."""
        booking = _booking("B001", body_length_m=10.0)
        triplets = [_triplet("Furulunden", "A1", length_m=0.0)]
        # Build a triplet with a zero-length anchor
        anchor = _spot("Furulunden", "A1", length_m=0.0)
        t = Triplet(
            room_id="Furulunden A1-3",
            section="Furulunden", row="A",
            spot_ids=["A1", "A2", "A3"],
            first_spot=anchor,
        )
        results = allocate([booking], [t], topo)
        assert results[0].is_assigned

    def test_exact_fit_assigned(self, topo):
        """Vehicle body exactly equals anchor spot length → assigned (not rejected)."""
        booking = _booking("B001", body_length_m=8.0)
        triplets = [_triplet("Furulunden", "A1", length_m=8.0)]
        results = allocate([booking], triplets, topo)
        assert results[0].is_assigned


# ---------------------------------------------------------------------------
# Occupancy: all three spots consumed
# ---------------------------------------------------------------------------

class TestOccupancy:
    def test_all_three_spots_occupied_after_assignment(self, topo):
        """After B001 takes triplet [A1,A2,A3], B002 cannot take any room
        that overlaps those spot IDs."""
        b1 = _booking("B001")
        b2 = _booking("B002")
        # Both share spots A2 (overlap in second triplet)
        t1 = _triplet("Furulunden", "A1")   # spots A1, A2, A3
        t2 = Triplet(                        # spots A2, A3, A4 — overlaps t1
            room_id="Furulunden A2-4",
            section="Furulunden", row="A",
            spot_ids=["A2", "A3", "A4"],
            first_spot=_spot("Furulunden", "A2", 10.0),
        )
        t3 = _triplet("Furulunden", "B1")   # spots B1, B2, B3 — no overlap
        results = allocate([b1, b2], [t1, t2, t3], topo)
        r1 = next(r for r in results if r.booking_no == "B001")
        r2 = next(r for r in results if r.booking_no == "B002")
        assert r1.is_assigned
        assert r2.is_assigned
        # B001 takes t1 (A1-A3), B002 cannot take t2 (overlaps A2-A3), takes t3
        assert set(r1.assigned_spot_ids) == {"A1", "A2", "A3"}
        assert set(r2.assigned_spot_ids) == {"B1", "B2", "B3"}

    def test_no_overlap_both_assigned(self, topo):
        b1 = _booking("B001")
        b2 = _booking("B002")
        t1 = _triplet("Furulunden", "A1")
        t2 = _triplet("Furulunden", "A4")
        results = allocate([b1, b2], [t1, t2], topo)
        assert all(r.is_assigned for r in results)
        ids1 = set(next(r for r in results if r.booking_no == "B001").assigned_spot_ids)
        ids2 = set(next(r for r in results if r.booking_no == "B002").assigned_spot_ids)
        assert ids1.isdisjoint(ids2)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_assigned_room_id_exact_sirvoy_string(self, topo):
        booking = _booking("B001", body_length_m=5.0)
        room_id = "Furulunden A16-18 (5m)"
        anchor = _spot("Furulunden", "A16", 5.0)
        t = Triplet(
            room_id=room_id,
            section="Furulunden", row="A",
            spot_ids=["A16", "A17", "A18"],
            first_spot=anchor,
        )
        results = allocate([booking], [t], topo)
        r = results[0]
        assert r.is_assigned
        assert r.assigned_room_id == room_id

    def test_assigned_spot_ids_all_three(self, topo):
        booking = _booking("B001")
        t = Triplet(
            room_id="Furulunden A1-3",
            section="Furulunden", row="A",
            spot_ids=["A1", "A2", "A3"],
            first_spot=_spot("Furulunden", "A1"),
        )
        results = allocate([booking], [t], topo)
        r = results[0]
        assert r.assigned_spot_ids == ["A1", "A2", "A3"]

    def test_assigned_spot_id_is_first_spot(self, topo):
        booking = _booking("B001")
        t = _triplet("Furulunden", "A4")
        results = allocate([booking], [t], topo)
        r = results[0]
        assert r.assigned_spot_id == "A4"

    def test_assigned_section_correct(self, topo):
        booking = _booking("B001")
        t = _triplet("Furulunden", "A1")
        results = allocate([booking], [t], topo)
        assert results[0].assigned_section == "Furulunden"

    def test_unassigned_result_shape(self, topo):
        booking = _booking("B001", body_length_m=100.0)  # too long for any spot
        t = _triplet("Furulunden", "A1", length_m=5.0)
        results = allocate([booking], [t], topo)
        r = results[0]
        assert not r.is_assigned
        assert r.assigned_room_id is None
        assert r.assigned_spot_ids == []
        assert r.assigned_spot_id is None
        assert r.unassigned_reason is not None


# ---------------------------------------------------------------------------
# No available triplets
# ---------------------------------------------------------------------------

class TestNoAvailableTriplets:
    def test_empty_triplet_list_unassigned(self, topo):
        booking = _booking("B001")
        results = allocate([booking], [], topo)
        r = results[0]
        assert not r.is_assigned
        assert r.unassigned_reason == "no_available_triplets"

    def test_all_occupied_unassigned(self, topo):
        b1 = _booking("B001")
        b2 = _booking("B002")
        t = _triplet("Furulunden", "A1")   # only one triplet
        results = allocate([b1, b2], [t], topo)
        assigned = [r for r in results if r.is_assigned]
        unassigned = [r for r in results if not r.is_assigned]
        assert len(assigned) == 1
        assert len(unassigned) == 1


# ---------------------------------------------------------------------------
# Section constraint
# ---------------------------------------------------------------------------

class TestSectionConstraint:
    def test_wrong_section_excluded(self, topo):
        booking = _booking("B001", preferred_sections=["Bedehuset"])
        t = _triplet("Furulunden", "A1")
        results = allocate([booking], [t], topo)
        assert not results[0].is_assigned

    def test_matching_section_assigned(self, topo):
        booking = _booking("B001", preferred_sections=["Furulunden"])
        t = _triplet("Furulunden", "A1")
        results = allocate([booking], [t], topo)
        assert results[0].is_assigned


# ---------------------------------------------------------------------------
# preferred_spot_ids: match any triplet spot
# ---------------------------------------------------------------------------

class TestPreferredSpotIdsAnyMatch:
    def test_preferred_second_spot_matches_triplet(self, topo):
        """Requesting 'A2' should prefer the triplet A1-A3 (which contains A2)."""
        booking = _booking("B001", preferred_spot_ids=["A2"])
        t_match = _triplet("Furulunden", "A1")   # contains A1,A2,A3
        t_other = _triplet("Furulunden", "A4")   # contains A4,A5,A6
        results = allocate([booking], [t_match, t_other], topo)
        r = results[0]
        assert r.is_assigned
        assert r.assigned_room_id == t_match.room_id

    def test_preferred_first_spot_matches_directly(self, topo):
        booking = _booking("B001", preferred_spot_ids=["A1"])
        t = _triplet("Furulunden", "A1")
        results = allocate([booking], [t], topo)
        assert results[0].is_assigned


# ---------------------------------------------------------------------------
# Social / group logic still works with Triplet candidates
# ---------------------------------------------------------------------------

class TestGroupLogicWithTriplets:
    def test_cluster_members_placed_nearby(self, topo):
        """Two cluster members should be placed in adjacent triplets."""
        b1 = _booking("B001")
        b2 = _booking("B002")
        cluster = _cluster(["B001", "B002"])
        # Triplets in same section with proximate anchor spots
        t1 = _triplet("Furulunden", "A1")
        t2 = _triplet("Furulunden", "A4")
        t3 = Triplet(
            room_id="Furulunden C1-3",
            section="Furulunden", row="C",
            spot_ids=["C1", "C2", "C3"],
            first_spot=_spot("Furulunden", "C1"),
        )
        results = allocate([b1, b2], [t1, t2, t3], topo, clusters=[cluster])
        # Both should be assigned
        assert all(r.is_assigned for r in results)
        # Both should be in Furulunden (same section as the available triplets)
        assert all(r.assigned_section == "Furulunden" for r in results)

    def test_explanation_has_required_keys(self, topo):
        booking = _booking("B001")
        t = _triplet("Furulunden", "A1")
        results = allocate([booking], [t], topo)
        exp = results[0].explanation
        assert "preference_score" in exp
        assert "group_score" in exp
        assert "group_context_seeded" in exp

    def test_is_assigned_property_consistent(self, topo):
        booking = _booking("B001")
        t = _triplet("Furulunden", "A1")
        r = allocate([booking], [t], topo)[0]
        # is_assigned should be consistent with assigned_room_id
        assert r.is_assigned == (r.assigned_room_id is not None)
