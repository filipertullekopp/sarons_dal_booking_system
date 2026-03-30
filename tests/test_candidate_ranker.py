"""Tests for saronsdal.allocation.candidate_ranker."""
import re
import pytest
from typing import List, Optional, Tuple

from tests.conftest import FIXTURES_DIR
from saronsdal.allocation.candidate_ranker import (
    RankedCandidate,
    TOP_N,
    W_GROUP,
    W_PREF,
    rank_candidates,
)
from saronsdal.models.normalized import (
    Booking,
    RawGroupSignals,
    Spot,
    SpotRequest,
    VehicleUnit,
)
from saronsdal.models.triplet import Triplet
from saronsdal.spatial.topology_loader import load_topology

# ---------------------------------------------------------------------------
# Fixture topology
# ---------------------------------------------------------------------------

_FURULUNDEN_ONLY = {"furulunden": "topology_grid_furulunden_mini.csv"}
_BOTH = {
    "furulunden": "topology_grid_furulunden_mini.csv",
    "bibelskolen": "topology_grid_bibelskolen_mini.csv",
}


@pytest.fixture(scope="module")
def topo():
    return load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)


@pytest.fixture(scope="module")
def topo_both():
    return load_topology(FIXTURES_DIR, filenames=_BOTH)


# ---------------------------------------------------------------------------
# Minimal object factories
# ---------------------------------------------------------------------------

def _spot(
    section: str = "Furulunden",
    spot_id: str = "A1",
    row: str = "A",
    hilliness: int = 0,
    no_motorhome: bool = False,
    no_caravan: bool = False,
    is_not_spot: bool = False,
    is_reserved: bool = False,
    length_m: float = 10.0,
) -> Spot:
    return Spot(
        spot_id=spot_id,
        section=section,
        row=row,
        position=1,
        length_m=length_m,
        hilliness=hilliness,
        is_end_of_row=False,
        is_not_spot=is_not_spot,
        is_reserved=is_reserved,
        no_motorhome=no_motorhome,
        no_caravan_nor_motorhome=no_caravan,
    )


def _triplet(
    section: str = "Furulunden",
    spot_id: str = "A1",
    row: str = "A",
    hilliness: int = 0,
    no_motorhome: bool = False,
    no_caravan: bool = False,
    length_m: float = 10.0,
) -> Triplet:
    """Minimal 3-spot Triplet with a real first_spot for ranking tests."""
    m = re.match(r"([A-Z]+)(\d+)", spot_id)
    row_letter = m.group(1)
    first_num = int(m.group(2))
    spot_ids = [f"{row_letter}{first_num + i}" for i in range(3)]
    anchor = _spot(section, spot_ids[0], row or row_letter, hilliness,
                   no_motorhome, no_caravan, length_m=length_m)
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
    preferred_sections: Optional[List[str]] = None,
    preferred_spot_ids: Optional[List[str]] = None,
) -> Booking:
    return Booking(
        booking_no=booking_no,
        booking_source="website",
        check_in=None,
        check_out=None,
        first_name="Test",
        last_name="Guest",
        full_name="Test Guest",
        num_guests=2,
        language="no",
        is_confirmed=True,
        vehicle=VehicleUnit(
            vehicle_type=vehicle_type,
            spec_size_hint=None,
            body_length_m=7.0,
            body_width_m=2.75,
            fortelt_width_m=0.0,
            total_width_m=2.75,
            required_spot_count=3,
            has_fortelt=False,
            has_markise=False,
            registration=None,
            parse_confidence=1.0,
        ),
        request=SpotRequest(
            preferred_sections=preferred_sections or [],
            preferred_spot_ids=preferred_spot_ids or [],
            avoid_sections=[],
            amenity_flags=set(),
            raw_near_texts=[],
            parse_confidence=1.0,
        ),
        group_signals=RawGroupSignals(
            organization=None,
            group_field=None,
            near_text_fragments=[],
            is_org_private=True,
        ),
        data_confidence=1.0,
    )


def _furulunden_triplets() -> List[Triplet]:
    """Representative triplets anchored in the mini Furulunden fixture grid."""
    return [
        _triplet("Furulunden", "A1", "A"),
        _triplet("Furulunden", "A4", "A"),
        _triplet("Furulunden", "B1", "B"),
        _triplet("Furulunden", "C1", "C"),
    ]


# ---------------------------------------------------------------------------
# Basic operation
# ---------------------------------------------------------------------------

class TestBasicOperation:
    def test_returns_list(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo)
        assert isinstance(ranked, list)

    def test_empty_candidates_returns_empty(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [], topo)
        assert ranked == []

    def test_returns_ranked_candidate_objects(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo)
        assert all(isinstance(r, RankedCandidate) for r in ranked)

    def test_result_count_capped_at_top_n(self, topo):
        # 4 candidates but top_n defaults to TOP_N (10) — all returned since 4 < 10
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo)
        assert len(ranked) <= TOP_N

    def test_custom_top_n_respected(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo, top_n=2)
        assert len(ranked) <= 2

    def test_top_n_zero_returns_all(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo, top_n=0)
        assert len(ranked) == 4


# ---------------------------------------------------------------------------
# Sorting / determinism
# ---------------------------------------------------------------------------

class TestSorting:
    def test_sorted_descending_by_total_score(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo)
        scores = [r.total_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_deterministic_same_input_same_output(self, topo):
        booking = _booking()
        r1 = rank_candidates(booking, _furulunden_triplets(), topo)
        r2 = rank_candidates(booking, _furulunden_triplets(), topo)
        assert [r.spot.spot_id for r in r1] == [r.spot.spot_id for r in r2]

    def test_tie_broken_by_section_then_first_spot_id(self, topo):
        # Two triplets with the same score: flat hilliness, no preferences
        triplets = [
            _triplet("Furulunden", "B1", "B", hilliness=0),
            _triplet("Furulunden", "A1", "A", hilliness=0),
        ]
        booking = _booking()
        ranked = rank_candidates(booking, triplets, topo, top_n=0)
        # Both have same preference score → tie-break by first_spot_id lexicographic
        spot_ids = [r.spot.spot_id for r in ranked]
        assert spot_ids == sorted(spot_ids)


# ---------------------------------------------------------------------------
# Hard violation filtering
# ---------------------------------------------------------------------------

class TestHardViolations:
    def test_no_motorhome_triplet_excluded_for_motorhome(self, topo):
        triplets = [_triplet("Furulunden", "A1", no_motorhome=True)]
        booking = _booking(vehicle_type="motorhome")
        ranked = rank_candidates(booking, triplets, topo)
        assert ranked == []

    def test_no_caravan_triplet_excluded_for_caravan(self, topo):
        triplets = [_triplet("Furulunden", "A1", no_caravan=True)]
        booking = _booking(vehicle_type="caravan")
        ranked = rank_candidates(booking, triplets, topo)
        assert ranked == []

    def test_wrong_section_excluded(self, topo):
        triplets = [_triplet("Furulunden", "A1")]
        booking = _booking(preferred_sections=["Bedehuset"])
        ranked = rank_candidates(booking, triplets, topo)
        assert ranked == []

    def test_right_section_included(self, topo):
        triplets = [_triplet("Furulunden", "A1")]
        booking = _booking(preferred_sections=["Furulunden"])
        ranked = rank_candidates(booking, triplets, topo)
        assert len(ranked) == 1

    def test_no_restriction_for_tent(self, topo):
        # no_motorhome flag only blocks motorhomes, not tents
        triplets = [_triplet("Furulunden", "A1", no_motorhome=True)]
        booking = _booking(vehicle_type="tent")
        ranked = rank_candidates(booking, triplets, topo)
        assert len(ranked) == 1

    def test_partial_violation_still_returns_valid_candidates(self, topo):
        triplets = [
            _triplet("Furulunden", "A1", no_motorhome=True),  # blocked
            _triplet("Furulunden", "A4"),                      # valid
        ]
        booking = _booking(vehicle_type="motorhome")
        ranked = rank_candidates(booking, triplets, topo)
        assert len(ranked) == 1
        assert ranked[0].spot.spot_id == "A4"


# ---------------------------------------------------------------------------
# Score combination
# ---------------------------------------------------------------------------

class TestScoreCombination:
    def test_total_score_matches_formula(self, topo):
        booking = _booking()
        triplets = [_triplet("Furulunden", "A1")]
        ranked = rank_candidates(booking, triplets, topo)
        assert len(ranked) == 1
        r = ranked[0]
        expected = W_PREF * r.preference_score + W_GROUP * r.group_score
        assert r.total_score == pytest.approx(expected, abs=1e-4)

    def test_no_group_group_score_is_zero(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [_triplet("Furulunden", "A1")], topo)
        assert ranked[0].group_score == pytest.approx(0.0)

    def test_total_score_without_group_equals_w_pref_times_pref(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [_triplet("Furulunden", "A1")], topo)
        r = ranked[0]
        assert r.total_score == pytest.approx(W_PREF * r.preference_score, abs=1e-4)

    def test_group_spot_raises_total_score(self, topo):
        # With a nearby group member, total_score should be higher than without
        booking = _booking()
        triplet = _triplet("Furulunden", "A1")

        without_group = rank_candidates(booking, [triplet], topo, group_spots=None)
        with_group = rank_candidates(
            booking, [triplet], topo, group_spots=[("Furulunden", "B1")]
        )
        assert with_group[0].total_score > without_group[0].total_score

    def test_total_score_bounded_zero_to_one(self, topo):
        booking = _booking()
        ranked = rank_candidates(
            booking, _furulunden_triplets(), topo,
            group_spots=[("Furulunden", "A4")],
        )
        for r in ranked:
            assert 0.0 <= r.total_score <= 1.0

    def test_preferred_spot_id_ranks_requested_triplet_first(self, topo):
        # A1 is preferred; A4 is not — triplet containing A1 should rank above A4
        triplets = [_triplet("Furulunden", "A4"), _triplet("Furulunden", "A1")]
        booking = _booking()
        ranked = rank_candidates(booking, triplets, topo, preferred_spot_ids=["A1"])
        assert ranked[0].spot.spot_id == "A1"
        assert ranked[0].preference_score > ranked[-1].preference_score

    def test_preferred_middle_spot_matches_triplet(self, topo):
        """Requesting 'A2' (middle of A1-A3 triplet) should prefer that triplet."""
        t_match = _triplet("Furulunden", "A1")   # contains A1, A2, A3
        t_other = _triplet("Furulunden", "A4")   # contains A4, A5, A6
        booking = _booking()
        ranked = rank_candidates(
            booking, [t_match, t_other], topo, preferred_spot_ids=["A2"]
        )
        assert ranked[0].spot.spot_id == "A1"  # first spot of the matched triplet


# ---------------------------------------------------------------------------
# Score breakdown fields
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_spot_score_field_present(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [_triplet("Furulunden", "A1")], topo)
        assert ranked[0].spot_score is not None

    def test_group_detail_field_present(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [_triplet("Furulunden", "A1")], topo)
        assert ranked[0].group_detail is not None

    def test_spot_score_section_matches(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [_triplet("Furulunden", "A1")], topo)
        assert ranked[0].spot_score.section == "Furulunden"

    def test_spot_score_spot_id_matches_anchor(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, [_triplet("Furulunden", "A1")], topo)
        assert ranked[0].spot_score.spot_id == "A1"

    def test_no_violations_in_ranked_results(self, topo):
        booking = _booking()
        ranked = rank_candidates(booking, _furulunden_triplets(), topo)
        for r in ranked:
            assert r.spot_score.violations == []

    def test_spot_property_returns_first_spot(self, topo):
        """RankedCandidate.spot is a backward-compat property returning first_spot."""
        booking = _booking()
        triplet = _triplet("Furulunden", "A1")
        ranked = rank_candidates(booking, [triplet], topo)
        assert ranked[0].spot is triplet.first_spot
        assert ranked[0].spot.spot_id == "A1"
