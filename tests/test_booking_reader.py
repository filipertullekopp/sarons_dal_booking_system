"""Tests for the bookings_basic CSV reader."""
import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.ingestion.booking_reader import load_bookings_basic
from saronsdal.ingestion.schema import SchemaVersion


MINI_BASIC = FIXTURES_DIR / "mini_basic.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bookings():
    return load_bookings_basic(MINI_BASIC)


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_load_bookings_count(bookings):
    # 5 valid rows; row with missing Booking No. is skipped
    assert len(bookings) == 5


def test_all_have_booking_no(bookings):
    assert all(b.booking_no for b in bookings)


def test_booking_numbers_expected(bookings):
    nos = {b.booking_no for b in bookings}
    assert nos == {"B001", "B002", "B003", "B004", "B005"}


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def test_schema_version_is_basic_export(bookings):
    # All rows should be tagged with BASIC_EXPORT
    assert all(b.schema_version == SchemaVersion.BASIC_EXPORT.name for b in bookings)


# ---------------------------------------------------------------------------
# Encoding repair (ftfy)
# ---------------------------------------------------------------------------

def test_encoding_repair_company_field(bookings):
    # B001 has "Betel HommersÃ¥k" in the CSV — ftfy must repair it
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.company == "Betel Hommersåk"


# ---------------------------------------------------------------------------
# Resepsjonen bookings are preserved
# ---------------------------------------------------------------------------

def test_resepsjonen_booking_present(bookings):
    sources = {b.booking_no: b.booking_source for b in bookings}
    assert sources["B004"] == "Resepsjonen"


def test_resepsjonen_booking_has_empty_dimensions(bookings):
    b004 = next(b for b in bookings if b.booking_no == "B004")
    assert b004.raw_length == ""
    assert b004.raw_width == ""


# ---------------------------------------------------------------------------
# Field mapping correctness
# ---------------------------------------------------------------------------

def test_b001_fortelt_ja(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.has_fortelt == "Ja"


def test_b001_markise_nei(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.has_markise == "Nei"


def test_b001_regnr(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.regnr == "AB12345"


def test_b001_raw_length(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.raw_length == "7,5"


def test_b001_raw_width(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.raw_width == "3,5"


def test_b001_location_wish(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.raw_location_wish == "Furutoppen"


def test_b002_guest_message(bookings):
    b002 = next(b for b in bookings if b.booking_no == "B002")
    # "nær" should be present after ftfy repair (no encoding damage in this case)
    assert "nær" in b002.guest_message


def test_b005_cross_dimension_raw_length(bookings):
    b005 = next(b for b in bookings if b.booking_no == "B005")
    assert "med drag" in b005.raw_length


def test_b005_location_wish_spot_ids(bookings):
    b005 = next(b for b in bookings if b.booking_no == "B005")
    assert b005.raw_location_wish == "D25-D27"


# ---------------------------------------------------------------------------
# Specs list is empty before merge
# ---------------------------------------------------------------------------

def test_specs_empty_before_merge(bookings):
    # load_bookings_basic does not attach specs; they start empty
    assert all(b.specs == [] for b in bookings)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load_bookings_basic(FIXTURES_DIR / "nonexistent.csv")


def test_confirmed_field_read(bookings):
    b001 = next(b for b in bookings if b.booking_no == "B001")
    assert b001.confirmed == "Ja"
