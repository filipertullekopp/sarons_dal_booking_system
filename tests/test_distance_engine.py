"""Tests for saronsdal.spatial.distance_engine."""
import math
import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.spatial.topology_loader import GridCoord, load_topology
from saronsdal.spatial.distance_engine import (
    CROSS_GRID_DISTANCE,
    euclidean,
    mean_group_distance,
    nearest_landmark_coord,
    pairwise_distances,
    spot_to_landmark_distance,
    spot_to_spot_distance,
)

_FURULUNDEN_ONLY = {"furulunden": "topology_grid_furulunden_mini.csv"}
_BIBELSKOLEN_ONLY = {"bibelskolen": "topology_grid_bibelskolen_mini.csv"}
_BOTH = {
    "furulunden": "topology_grid_furulunden_mini.csv",
    "bibelskolen": "topology_grid_bibelskolen_mini.csv",
}


@pytest.fixture(scope="module")
def topo_furulunden():
    return load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)


@pytest.fixture(scope="module")
def topo_bibelskolen():
    return load_topology(FIXTURES_DIR, filenames=_BIBELSKOLEN_ONLY)


@pytest.fixture(scope="module")
def topo_both():
    return load_topology(FIXTURES_DIR, filenames=_BOTH)


# ---------------------------------------------------------------------------
# euclidean (unit-level)
# ---------------------------------------------------------------------------

class TestEuclidean:
    def test_same_point_is_zero(self):
        c = GridCoord(x=3, y=4, grid="g")
        assert euclidean(c, c) == pytest.approx(0.0)

    def test_horizontal_distance(self):
        a = GridCoord(x=0, y=0, grid="g")
        b = GridCoord(x=3, y=0, grid="g")
        assert euclidean(a, b) == pytest.approx(3.0)

    def test_vertical_distance(self):
        a = GridCoord(x=0, y=0, grid="g")
        b = GridCoord(x=0, y=4, grid="g")
        assert euclidean(a, b) == pytest.approx(4.0)

    def test_diagonal_345(self):
        a = GridCoord(x=0, y=0, grid="g")
        b = GridCoord(x=3, y=4, grid="g")
        assert euclidean(a, b) == pytest.approx(5.0)

    def test_cross_grid_returns_sentinel(self):
        a = GridCoord(x=0, y=0, grid="grid1")
        b = GridCoord(x=1, y=1, grid="grid2")
        assert euclidean(a, b) == CROSS_GRID_DISTANCE

    def test_symmetric(self):
        a = GridCoord(x=2, y=3, grid="g")
        b = GridCoord(x=5, y=7, grid="g")
        assert euclidean(a, b) == pytest.approx(euclidean(b, a))


# ---------------------------------------------------------------------------
# spot_to_spot_distance
# ---------------------------------------------------------------------------

class TestSpotToSpotDistance:
    def test_same_spot_is_zero(self, topo_furulunden):
        d = spot_to_spot_distance(topo_furulunden, "Furulunden", "A1", "Furulunden", "A1")
        assert d == pytest.approx(0.0)

    def test_adjacent_spots_horizontal(self, topo_furulunden):
        # A1 at (0,0), A2 at (1,0) → distance = 1
        d = spot_to_spot_distance(topo_furulunden, "Furulunden", "A1", "Furulunden", "A2")
        assert d == pytest.approx(1.0)

    def test_adjacent_spots_vertical(self, topo_furulunden):
        # A1 at (0,0), B1 at (0,1) → distance = 1
        d = spot_to_spot_distance(topo_furulunden, "Furulunden", "A1", "Furulunden", "B1")
        assert d == pytest.approx(1.0)

    def test_diagonal_distance(self, topo_furulunden):
        # A1 at (0,0), B2 at (1,1) → distance = sqrt(2)
        d = spot_to_spot_distance(topo_furulunden, "Furulunden", "A1", "Furulunden", "B2")
        assert d == pytest.approx(math.sqrt(2))

    def test_empty_cells_contribute_to_distance(self, topo_furulunden):
        # A1 at (0,0), C1 at (2,3). EMPTY cells between them still add to distance.
        # Distance = sqrt((2-0)^2 + (3-0)^2) = sqrt(4+9) = sqrt(13)
        d = spot_to_spot_distance(topo_furulunden, "Furulunden", "A1", "Furulunden", "C1")
        assert d == pytest.approx(math.sqrt(13))

    def test_missing_spot_raises_key_error(self, topo_furulunden):
        with pytest.raises(KeyError):
            spot_to_spot_distance(topo_furulunden, "Furulunden", "Z99", "Furulunden", "A1")

    def test_cross_grid_returns_cross_distance(self, topo_both):
        # Furulunden A1 and Internatet D1 are in different grids
        d = spot_to_spot_distance(topo_both, "Furulunden", "A1", "Internatet", "D1")
        assert d == CROSS_GRID_DISTANCE


# ---------------------------------------------------------------------------
# spot_to_landmark_distance
# ---------------------------------------------------------------------------

class TestSpotToLandmarkDistance:
    def test_a3_to_toilet(self, topo_furulunden):
        # A3 at (2,0), toilet at (4,0) → distance = 2
        d = spot_to_landmark_distance(topo_furulunden, "Furulunden", "A3", "toilet")
        assert d == pytest.approx(2.0)

    def test_a1_to_toilet(self, topo_furulunden):
        # A1 at (0,0), toilet at (4,0) → distance = 4
        d = spot_to_landmark_distance(topo_furulunden, "Furulunden", "A1", "toilet")
        assert d == pytest.approx(4.0)

    def test_b1_to_river(self, topo_furulunden):
        # B1 at (0,1), river at (3,2) → sqrt(9+1) = sqrt(10)
        d = spot_to_landmark_distance(topo_furulunden, "Furulunden", "B1", "river")
        assert d == pytest.approx(math.sqrt(10))

    def test_c2_to_road(self, topo_furulunden):
        # C2 at (2,4), road at (0,4) → distance = 2
        d = spot_to_landmark_distance(topo_furulunden, "Furulunden", "C2", "road")
        assert d == pytest.approx(2.0)

    def test_no_same_grid_landmark_returns_cross_distance(self, topo_furulunden):
        # No "shower" landmark in furulunden grid
        d = spot_to_landmark_distance(topo_furulunden, "Furulunden", "A1", "shower")
        assert d == CROSS_GRID_DISTANCE

    def test_missing_spot_raises_key_error(self, topo_furulunden):
        with pytest.raises(KeyError):
            spot_to_landmark_distance(topo_furulunden, "Furulunden", "Z99", "toilet")

    def test_cross_grid_landmark_not_used(self, topo_both):
        # Furulunden A1; bibelskolen has a toilet but it's in a different grid.
        # Only the furulunden toilet (at (4,0)) should count.
        d = spot_to_landmark_distance(topo_both, "Furulunden", "A1", "toilet")
        assert d == pytest.approx(4.0)   # not cross-grid distance

    def test_internatet_spot_uses_own_grid_toilet(self, topo_both):
        # Internatet D1 at (0,0); bibelskolen grid toilet at (2,0) → distance = 2
        d = spot_to_landmark_distance(topo_both, "Internatet", "D1", "toilet")
        assert d == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# nearest_landmark_coord
# ---------------------------------------------------------------------------

class TestNearestLandmarkCoord:
    def test_returns_correct_coord(self, topo_furulunden):
        lm = nearest_landmark_coord(topo_furulunden, "Furulunden", "A3", "toilet")
        assert lm is not None
        assert lm.x == 4 and lm.y == 0

    def test_no_landmark_returns_none(self, topo_furulunden):
        lm = nearest_landmark_coord(topo_furulunden, "Furulunden", "A1", "shower")
        assert lm is None

    def test_missing_spot_returns_none(self, topo_furulunden):
        lm = nearest_landmark_coord(topo_furulunden, "Furulunden", "Z99", "toilet")
        assert lm is None


# ---------------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------------

class TestPairwiseDistances:
    def test_two_spots(self, topo_furulunden):
        keys = [("Furulunden", "A1"), ("Furulunden", "A2")]
        dists = pairwise_distances(topo_furulunden, keys)
        assert len(dists) == 1
        assert dists[0] == pytest.approx(1.0)

    def test_three_spots(self, topo_furulunden):
        # A1(0,0), A2(1,0), B1(0,1) → 3 pairs
        keys = [("Furulunden", "A1"), ("Furulunden", "A2"), ("Furulunden", "B1")]
        dists = pairwise_distances(topo_furulunden, keys)
        assert len(dists) == 3

    def test_empty_list_returns_empty(self, topo_furulunden):
        assert pairwise_distances(topo_furulunden, []) == []

    def test_single_spot_returns_empty(self, topo_furulunden):
        assert pairwise_distances(topo_furulunden, [("Furulunden", "A1")]) == []

    def test_missing_spot_skipped(self, topo_furulunden):
        keys = [("Furulunden", "A1"), ("Furulunden", "Z99"), ("Furulunden", "A2")]
        dists = pairwise_distances(topo_furulunden, keys)
        # Z99 missing → only A1 and A2 remain → 1 pair
        assert len(dists) == 1
        assert dists[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mean_group_distance
# ---------------------------------------------------------------------------

class TestMeanGroupDistance:
    def test_two_adjacent_spots(self, topo_furulunden):
        keys = [("Furulunden", "A1"), ("Furulunden", "A2")]
        assert mean_group_distance(topo_furulunden, keys) == pytest.approx(1.0)

    def test_insufficient_spots_returns_cross_distance(self, topo_furulunden):
        assert mean_group_distance(topo_furulunden, []) == CROSS_GRID_DISTANCE
        assert mean_group_distance(topo_furulunden, [("Furulunden", "A1")]) == CROSS_GRID_DISTANCE

    def test_mean_of_three_spots(self, topo_furulunden):
        # A1(0,0), A2(1,0), B1(0,1)
        # d(A1,A2)=1, d(A1,B1)=1, d(A2,B1)=sqrt(2)
        keys = [("Furulunden", "A1"), ("Furulunden", "A2"), ("Furulunden", "B1")]
        expected = (1.0 + 1.0 + math.sqrt(2)) / 3
        assert mean_group_distance(topo_furulunden, keys) == pytest.approx(expected)
