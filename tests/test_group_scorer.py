"""Tests for saronsdal.allocation.group_scorer."""
import math
import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.allocation.group_scorer import GroupScore, score_group_proximity
from saronsdal.models.normalized import Spot
from saronsdal.spatial.distance_engine import CROSS_GRID_DISTANCE
from saronsdal.spatial.topology_loader import load_topology

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FURULUNDEN_ONLY = {"furulunden": "topology_grid_furulunden_mini.csv"}
_BOTH = {
    "furulunden": "topology_grid_furulunden_mini.csv",
    "bibelskolen": "topology_grid_bibelskolen_mini.csv",
}


def _spot(section: str, spot_id: str, row: str = "A") -> Spot:
    """Minimal Spot factory for tests."""
    return Spot(
        spot_id=spot_id,
        section=section,
        row=row,
        position=1,
        length_m=10.0,
        hilliness=0,
        is_end_of_row=False,
        is_not_spot=False,
        is_reserved=False,
        no_motorhome=False,
        no_caravan_nor_motorhome=False,
    )


@pytest.fixture(scope="module")
def topo_furulunden():
    return load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)


@pytest.fixture(scope="module")
def topo_both():
    return load_topology(FIXTURES_DIR, filenames=_BOTH)


# ---------------------------------------------------------------------------
# No group — score = 0.0
# ---------------------------------------------------------------------------

class TestNoGroup:
    def test_none_group_spots(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, None)
        assert gs.score == pytest.approx(0.0)
        assert gs.member_count == 0

    def test_empty_group_spots(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [])
        assert gs.score == pytest.approx(0.0)
        assert gs.member_count == 0

    def test_no_group_mean_distance_is_sentinel(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, None)
        assert gs.mean_distance == CROSS_GRID_DISTANCE


# ---------------------------------------------------------------------------
# Single group member — same grid
# ---------------------------------------------------------------------------

class TestSingleMember:
    def test_adjacent_member_score_near_one(self, topo_furulunden):
        # A1 at (0,0), A2 at (1,0): distance=1, score=1/(1+1)=0.5
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "A2")])
        assert gs.score == pytest.approx(1 / 2, abs=1e-4)

    def test_same_spot_score_is_one(self, topo_furulunden):
        # distance=0, score=1/(1+0)=1.0
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "A1")])
        assert gs.score == pytest.approx(1.0, abs=1e-4)

    def test_distant_member_score_small(self, topo_furulunden):
        # A1 at (0,0), C3 at (3,4): distance=5, score=1/6≈0.1667
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "C3")])
        assert gs.score == pytest.approx(1 / 6, abs=1e-4)

    def test_member_count_is_one(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "A2")])
        assert gs.member_count == 1

    def test_formula_1_over_1_plus_distance(self, topo_furulunden):
        # A1(0,0) → B2(1,1): dist=sqrt(2), score=1/(1+sqrt(2))
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "B2")])
        expected = 1 / (1 + math.sqrt(2))
        assert gs.score == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Multiple group members — mean distance
# ---------------------------------------------------------------------------

class TestMultipleMembers:
    def test_mean_of_two_members(self, topo_furulunden):
        # A1(0,0) to A2(1,0)=1, to B1(0,1)=1 → mean=1, score=0.5
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(
            spot, topo_furulunden,
            [("Furulunden", "A2"), ("Furulunden", "B1")]
        )
        assert gs.score == pytest.approx(0.5, abs=1e-4)
        assert gs.member_count == 2

    def test_mean_of_three_members(self, topo_furulunden):
        # A1(0,0) to A2(1,0)=1, B1(0,1)=1, B2(1,1)=sqrt(2)
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(
            spot, topo_furulunden,
            [("Furulunden", "A2"), ("Furulunden", "B1"), ("Furulunden", "B2")]
        )
        mean_d = (1 + 1 + math.sqrt(2)) / 3
        assert gs.score == pytest.approx(1 / (1 + mean_d), abs=1e-4)
        assert gs.member_count == 3

    def test_mean_distance_matches_score(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(
            spot, topo_furulunden,
            [("Furulunden", "A2"), ("Furulunden", "C3")]
        )
        # Verify mean_distance and score are consistent
        assert gs.score == pytest.approx(1 / (1 + gs.mean_distance), abs=1e-4)


# ---------------------------------------------------------------------------
# Cross-grid group members
# ---------------------------------------------------------------------------

class TestCrossGrid:
    def test_cross_grid_member_excluded(self, topo_both):
        # Furulunden spot, Internatet group member → cross-grid → score=0
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_both, [("Internatet", "D1")])
        assert gs.score == pytest.approx(0.0)
        assert gs.member_count == 0

    def test_mix_same_and_cross_grid(self, topo_both):
        # A2(1,0) same-grid: dist=1; D1 cross-grid: excluded
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(
            spot, topo_both,
            [("Furulunden", "A2"), ("Internatet", "D1")]
        )
        # Only A2 contributes: dist=1, score=0.5
        assert gs.score == pytest.approx(0.5, abs=1e-4)
        assert gs.member_count == 1


# ---------------------------------------------------------------------------
# Missing spots in topology
# ---------------------------------------------------------------------------

class TestMissingSpots:
    def test_candidate_not_in_topology_still_works(self, topo_furulunden):
        # Spot Z99 is not in topology — KeyError handled gracefully
        spot = _spot("Furulunden", "Z99")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "A1")])
        # spot_to_spot_distance raises KeyError → score=0
        assert gs.score == pytest.approx(0.0)

    def test_group_member_not_in_topology(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "Z99")])
        assert gs.score == pytest.approx(0.0)
        assert gs.member_count == 0

    def test_all_members_missing_returns_zero(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(
            spot, topo_furulunden,
            [("Furulunden", "Z99"), ("Furulunden", "Z98")]
        )
        assert gs.score == pytest.approx(0.0)

    def test_partial_valid_members_contribute(self, topo_furulunden):
        # A2 valid (dist=1), Z99 missing → mean of [1] → score=0.5
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(
            spot, topo_furulunden,
            [("Furulunden", "A2"), ("Furulunden", "Z99")]
        )
        assert gs.score == pytest.approx(0.5, abs=1e-4)
        assert gs.member_count == 1


# ---------------------------------------------------------------------------
# Return type / notes
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_group_score_dataclass(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, None)
        assert isinstance(gs, GroupScore)

    def test_notes_non_empty(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", "A2")])
        assert len(gs.notes) > 0

    def test_no_group_notes_explain_absence(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        gs = score_group_proximity(spot, topo_furulunden, None)
        assert any("no group" in n.lower() for n in gs.notes)

    def test_score_in_zero_to_one(self, topo_furulunden):
        spot = _spot("Furulunden", "A1")
        for g_sid in ["A1", "A2", "B1", "C3"]:
            gs = score_group_proximity(spot, topo_furulunden, [("Furulunden", g_sid)])
            assert 0.0 <= gs.score <= 1.0
