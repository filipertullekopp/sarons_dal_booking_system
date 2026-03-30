"""Tests for saronsdal.spatial.topology_loader."""
import math
import pytest
from pathlib import Path

from tests.conftest import FIXTURES_DIR
from saronsdal.spatial.topology_loader import (
    GridCoord,
    Topology,
    load_topology,
    normalise_spot_id,
    _classify_cell,
)

# Fixture files
FURULUNDEN_MINI = FIXTURES_DIR / "topology_grid_furulunden_mini.csv"
BIBELSKOLEN_MINI = FIXTURES_DIR / "topology_grid_bibelskolen_mini.csv"

# Minimal filenames override for each fixture
_FURULUNDEN_ONLY = {"furulunden": "topology_grid_furulunden_mini.csv"}
_BIBELSKOLEN_ONLY = {"bibelskolen": "topology_grid_bibelskolen_mini.csv"}
_BOTH = {
    "furulunden": "topology_grid_furulunden_mini.csv",
    "bibelskolen": "topology_grid_bibelskolen_mini.csv",
}


# ---------------------------------------------------------------------------
# normalise_spot_id
# ---------------------------------------------------------------------------

class TestNormaliseSpotId:
    def test_already_unpadded(self):
        assert normalise_spot_id("A1") == "A1"

    def test_strips_single_leading_zero(self):
        assert normalise_spot_id("C01") == "C1"

    def test_strips_leading_zero_multi(self):
        assert normalise_spot_id("E07") == "E7"

    def test_two_digit_unchanged(self):
        assert normalise_spot_id("B38") == "B38"

    def test_double_letter_prefix(self):
        assert normalise_spot_id("AB02") == "AB2"

    def test_uppercase_normalised(self):
        assert normalise_spot_id("d22") == "D22"

    def test_no_numeric_suffix_returns_raw(self):
        # No digit suffix — returned unchanged
        assert normalise_spot_id("ABC") == "ABC"

    def test_whitespace_stripped(self):
        assert normalise_spot_id("  A01  ") == "A1"


# ---------------------------------------------------------------------------
# _classify_cell
# ---------------------------------------------------------------------------

class TestClassifyCell:
    def test_empty_string(self):
        kind, a, b = _classify_cell("")
        assert kind == "empty"

    def test_empty_keyword(self):
        kind, a, b = _classify_cell("EMPTY")
        assert kind == "empty"

    def test_emtpy_typo(self):
        # Real topology files use the "EMTPY" typo
        kind, a, b = _classify_cell("EMTPY")
        assert kind == "empty"

    def test_spot_furulunden(self):
        kind, section, spot_id = _classify_cell("Furulunden A1")
        assert kind == "spot"
        assert section == "Furulunden"
        assert spot_id == "A1"

    def test_spot_bibelskolen_normalised(self):
        # Bibelskolen → Internatet
        kind, section, spot_id = _classify_cell("Bibelskolen E01")
        assert kind == "spot"
        assert section == "Internatet"
        assert spot_id == "E1"

    def test_spot_id_zero_padded(self):
        kind, section, spot_id = _classify_cell("Furulunden C01")
        assert kind == "spot"
        assert spot_id == "C1"

    def test_landmark_toilet(self):
        kind, lm_type, _ = _classify_cell("toilet")
        assert kind == "landmark"
        assert lm_type == "toilet"

    def test_landmark_toilets_plural(self):
        kind, lm_type, _ = _classify_cell("toilets")
        assert kind == "landmark"
        assert lm_type == "toilet"

    def test_landmark_river(self):
        kind, lm_type, _ = _classify_cell("river")
        assert kind == "landmark"
        assert lm_type == "river"

    def test_landmark_road(self):
        kind, lm_type, _ = _classify_cell("road")
        assert kind == "landmark"
        assert lm_type == "road"

    def test_landmark_main_road(self):
        kind, lm_type, _ = _classify_cell("main road")
        assert kind == "landmark"
        assert lm_type == "main_road"

    def test_landmark_case_insensitive(self):
        kind, lm_type, _ = _classify_cell("Toilet")
        assert kind == "landmark"
        assert lm_type == "toilet"

    def test_unknown_text(self):
        kind, text, _ = _classify_cell("Something weird 42")
        assert kind == "unknown"
        assert text == "Something weird 42"


# ---------------------------------------------------------------------------
# load_topology — single grid (Furulunden mini)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def topo_furulunden():
    return load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)


class TestLoadTopologyFurulunden:
    """
    Grid layout (col→x, row→y):

       col0            col1            col2      col3   col4
    r0 Furulunden A1   Furulunden A2   Furulunden A3    EMPTY  toilet
    r1 Furulunden B1   Furulunden B2   EMPTY            EMPTY  EMPTY
    r2 EMPTY           EMPTY           EMPTY            river  EMPTY
    r3 EMPTY           EMPTY           Furulunden C01   EMPTY  EMPTY
    r4 road            EMPTY           Furulunden C02   Furulunden C03 EMPTY
    """

    def test_spot_count(self, topo_furulunden):
        keys = topo_furulunden.spot_keys()
        assert len(keys) == 8

    def test_a1_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "A1")
        assert c is not None
        assert c.x == 0 and c.y == 0

    def test_a2_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "A2")
        assert c is not None
        assert c.x == 1 and c.y == 0

    def test_a3_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "A3")
        assert c is not None
        assert c.x == 2 and c.y == 0

    def test_b1_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "B1")
        assert c is not None
        assert c.x == 0 and c.y == 1

    def test_b2_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "B2")
        assert c is not None
        assert c.x == 1 and c.y == 1

    def test_c1_zero_padded_in_file_normalised(self, topo_furulunden):
        # File has "Furulunden C01" → spot_id should be "C1"
        c = topo_furulunden.get_spot_coord("Furulunden", "C1")
        assert c is not None
        assert c.x == 2 and c.y == 3

    def test_c2_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "C2")
        assert c is not None
        assert c.x == 2 and c.y == 4

    def test_c3_coordinates(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "C3")
        assert c is not None
        assert c.x == 3 and c.y == 4

    def test_zero_padded_key_does_not_exist(self, topo_furulunden):
        # "C01" should NOT be stored — only "C1"
        assert topo_furulunden.get_spot_coord("Furulunden", "C01") is None

    def test_grid_name_on_coords(self, topo_furulunden):
        c = topo_furulunden.get_spot_coord("Furulunden", "A1")
        assert c.grid == "furulunden"

    def test_toilet_landmark(self, topo_furulunden):
        coords = topo_furulunden.get_landmark_coords("toilet")
        assert len(coords) == 1
        lm = coords[0]
        assert lm.x == 4 and lm.y == 0  # col4, row0

    def test_river_landmark(self, topo_furulunden):
        coords = topo_furulunden.get_landmark_coords("river")
        assert len(coords) == 1
        lm = coords[0]
        assert lm.x == 3 and lm.y == 2  # col3, row2

    def test_road_landmark(self, topo_furulunden):
        coords = topo_furulunden.get_landmark_coords("road")
        assert len(coords) == 1
        lm = coords[0]
        assert lm.x == 0 and lm.y == 4  # col0, row4

    def test_missing_landmark_returns_empty_list(self, topo_furulunden):
        assert topo_furulunden.get_landmark_coords("shower") == []

    def test_sections_list(self, topo_furulunden):
        assert topo_furulunden.sections() == ["Furulunden"]

    def test_grid_dimensions_recorded(self, topo_furulunden):
        dims = topo_furulunden.grid_dimensions.get("furulunden")
        assert dims is not None
        # 5 columns (0–4), 5 rows (0–4)
        assert dims == (5, 5)


# ---------------------------------------------------------------------------
# load_topology — Bibelskolen → Internatet normalization
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def topo_bibelskolen():
    return load_topology(FIXTURES_DIR, filenames=_BIBELSKOLEN_ONLY)


class TestLoadTopologyBibelskolen:
    """
    Grid layout:

       col0              col1              col2
    r0 Bibelskolen D01   Bibelskolen D02   toilet
    r1 Bibelskolen D03   EMPTY             EMPTY
    """

    def test_section_normalised_to_internatet(self, topo_bibelskolen):
        # Bibelskolen in file → stored as Internatet
        assert ("Internatet", "D1") in topo_bibelskolen.spot_index

    def test_bibelskolen_not_stored_as_is(self, topo_bibelskolen):
        assert ("Bibelskolen", "D1") not in topo_bibelskolen.spot_index

    def test_d1_at_col0_row0(self, topo_bibelskolen):
        c = topo_bibelskolen.get_spot_coord("Internatet", "D1")
        assert c.x == 0 and c.y == 0

    def test_d2_at_col1_row0(self, topo_bibelskolen):
        c = topo_bibelskolen.get_spot_coord("Internatet", "D2")
        assert c.x == 1 and c.y == 0

    def test_d3_at_col0_row1(self, topo_bibelskolen):
        c = topo_bibelskolen.get_spot_coord("Internatet", "D3")
        assert c.x == 0 and c.y == 1

    def test_toilet_at_col2_row0(self, topo_bibelskolen):
        coords = topo_bibelskolen.get_landmark_coords("toilet")
        assert len(coords) == 1
        assert coords[0].x == 2 and coords[0].y == 0

    def test_three_spots_total(self, topo_bibelskolen):
        assert len(topo_bibelskolen.spot_keys()) == 3


# ---------------------------------------------------------------------------
# load_topology — two grids (both loaded together)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def topo_both():
    return load_topology(FIXTURES_DIR, filenames=_BOTH)


class TestLoadTopologyBothGrids:
    def test_total_spot_count(self, topo_both):
        # 8 from furulunden + 3 from bibelskolen
        assert len(topo_both.spot_keys()) == 11

    def test_sections_list_sorted(self, topo_both):
        assert topo_both.sections() == ["Furulunden", "Internatet"]

    def test_furulunden_grid_name(self, topo_both):
        c = topo_both.get_spot_coord("Furulunden", "A1")
        assert c.grid == "furulunden"

    def test_internatet_grid_name(self, topo_both):
        c = topo_both.get_spot_coord("Internatet", "D1")
        assert c.grid == "bibelskolen"

    def test_toilets_in_both_grids(self, topo_both):
        # Both grids have a toilet cell
        coords = topo_both.get_landmark_coords("toilet")
        grids = {c.grid for c in coords}
        assert "furulunden" in grids
        assert "bibelskolen" in grids

    def test_missing_file_skipped_gracefully(self):
        # A filename that doesn't exist should be skipped, not raise
        topo = load_topology(FIXTURES_DIR, filenames={"missing": "no_such_file.csv"})
        assert len(topo.spot_keys()) == 0


# ---------------------------------------------------------------------------
# Topology accessor helpers
# ---------------------------------------------------------------------------

class TestTopologyAccessors:
    def test_spots_in_section(self, topo_furulunden):
        pairs = topo_furulunden.spots_in_section("Furulunden")
        spot_ids = {sid for sid, _ in pairs}
        assert "A1" in spot_ids
        assert "C1" in spot_ids

    def test_spots_in_nonexistent_section(self, topo_furulunden):
        assert topo_furulunden.spots_in_section("Nowhere") == []

    def test_get_spot_coord_missing_returns_none(self, topo_furulunden):
        assert topo_furulunden.get_spot_coord("Furulunden", "Z99") is None
