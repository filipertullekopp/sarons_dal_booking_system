"""Tests for the spot CSV loader."""
import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.ingestion.spot_loader import _parse_bool_flag, _parse_spot_id, load_spots


MINI_SPOTS = FIXTURES_DIR / "mini_spots.csv"


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestParseBoolFlag:
    def test_one_is_true(self):
        assert _parse_bool_flag("1") is True

    def test_empty_is_false(self):
        assert _parse_bool_flag("") is False

    def test_zero_is_false(self):
        assert _parse_bool_flag("0") is False

    def test_true_string(self):
        assert _parse_bool_flag("true") is True

    def test_ja(self):
        assert _parse_bool_flag("Ja") is True

    def test_yes(self):
        assert _parse_bool_flag("yes") is True

    def test_random_string_is_false(self):
        assert _parse_bool_flag("maybe") is False


class TestParseSpotId:
    def test_single_letter_row(self):
        row, pos = _parse_spot_id("A1")
        assert row == "A"
        assert pos == 1

    def test_two_letter_row(self):
        row, pos = _parse_spot_id("AB23")
        assert row == "AB"
        assert pos == 23

    def test_large_position(self):
        row, pos = _parse_spot_id("D100")
        assert row == "D"
        assert pos == 100

    def test_lowercase_normalized_to_upper(self):
        row, pos = _parse_spot_id("b5")
        assert row == "B"
        assert pos == 5

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_spot_id("BADSPOT")

    def test_only_digits_raises(self):
        with pytest.raises(ValueError):
            _parse_spot_id("123")

    def test_only_letters_raises(self):
        with pytest.raises(ValueError):
            _parse_spot_id("ABC")


# ---------------------------------------------------------------------------
# Integration: load_spots
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spots():
    return load_spots(MINI_SPOTS)


def test_load_spots_correct_count(spots):
    # 7 valid spots; rows with empty spot_id or area are skipped
    assert len(spots) == 7


def test_all_widths_are_3m(spots):
    assert all(s.width_m == pytest.approx(3.0) for s in spots)


def test_coordinates_are_none(spots):
    assert all(s.coordinates is None for s in spots)


def test_spot_a1_normal(spots):
    a1 = next(s for s in spots if s.spot_id == "A1")
    assert a1.section == "Furulunden"
    assert a1.row == "A"
    assert a1.position == 1
    assert a1.length_m == pytest.approx(12.5)
    assert a1.hilliness == 0
    assert a1.is_end_of_row is False
    assert a1.is_not_spot is False
    assert a1.is_reserved is False
    assert a1.no_motorhome is False
    assert a1.no_caravan_nor_motorhome is False
    assert a1.is_allocatable is True


def test_end_of_row_flag(spots):
    a2 = next(s for s in spots if s.spot_id == "A2")
    assert a2.is_end_of_row is True
    assert a2.hilliness == 1


def test_not_spot_flag(spots):
    b1 = next(s for s in spots if s.spot_id == "B1")
    assert b1.is_not_spot is True
    assert b1.is_allocatable is False


def test_reserved_flag(spots):
    b2 = next(s for s in spots if s.spot_id == "B2")
    assert b2.is_reserved is True
    assert b2.is_allocatable is False


def test_no_motorhome_flag(spots):
    c1 = next(s for s in spots if s.spot_id == "C1")
    assert c1.no_motorhome is True
    assert c1.no_caravan_nor_motorhome is False


def test_no_caravan_nor_motorhome_flag(spots):
    c2 = next(s for s in spots if s.spot_id == "C2")
    assert c2.no_caravan_nor_motorhome is True
    assert c2.no_motorhome is False


def test_length_norm_parsed(spots):
    d1 = next(s for s in spots if s.spot_id == "D1")
    assert d1.length_norm == pytest.approx(8.5)


def test_length_norm_none_when_missing(spots):
    b1 = next(s for s in spots if s.spot_id == "B1")
    assert b1.length_norm is None


def test_missing_spot_id_rows_skipped():
    # Fixture has rows with empty spot_id and empty area; those should be skipped
    spots = load_spots(MINI_SPOTS)
    assert all(s.spot_id != "" for s in spots)
    assert all(s.section != "" for s in spots)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_spots(FIXTURES_DIR / "nonexistent.csv")
