"""Tests for the booking merger (basic CSV ← left join → specification CSV)."""
import logging

import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.ingestion.merger import merge_bookings


MINI_BASIC = FIXTURES_DIR / "mini_basic.csv"
MINI_SPEC = FIXTURES_DIR / "mini_spec.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def merged():
    return merge_bookings(MINI_BASIC, MINI_SPEC)


# ---------------------------------------------------------------------------
# Count and identity
# ---------------------------------------------------------------------------

def test_merged_preserves_all_basic_bookings(merged):
    # 5 valid rows in mini_basic.csv
    assert len(merged) == 5


def test_all_booking_nos_present(merged):
    nos = {b.booking_no for b in merged}
    assert nos == {"B001", "B002", "B003", "B004", "B005"}


# ---------------------------------------------------------------------------
# Spec attachment
# ---------------------------------------------------------------------------

def test_b001_has_two_spec_rows(merged):
    b001 = next(b for b in merged if b.booking_no == "B001")
    assert len(b001.specs) == 2


def test_b001_extras_spec_is_campingvogn(merged):
    b001 = next(b for b in merged if b.booking_no == "B001")
    extras = b001.extra_specification_values()
    assert "Campingvogn" in extras


def test_b001_accomm_spec_present(merged):
    b001 = next(b for b in merged if b.booking_no == "B001")
    accomm = b001.accomm_specs()
    assert len(accomm) == 1
    assert "Furulunden" in accomm[0].specification


def test_b002_has_tent_spec(merged):
    b002 = next(b for b in merged if b.booking_no == "B002")
    assert b002.extra_specification_values() == ["Telt - stort (3-8 personer)"]


def test_b003_has_no_specs(merged):
    b003 = next(b for b in merged if b.booking_no == "B003")
    assert b003.specs == []


def test_b004_has_motorhome_spec(merged):
    b004 = next(b for b in merged if b.booking_no == "B004")
    assert "Bobil" in b004.extra_specification_values()


# ---------------------------------------------------------------------------
# Orphan spec rows (B006 in spec but not in basic)
# ---------------------------------------------------------------------------

def test_orphan_spec_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="saronsdal.ingestion.merger"):
        merge_bookings(MINI_BASIC, MINI_SPEC)
    assert any("B006" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# helper methods
# ---------------------------------------------------------------------------

def test_extras_specs_filters_correctly(merged):
    b001 = next(b for b in merged if b.booking_no == "B001")
    extras = b001.extras_specs()
    assert all(s.spec_type.upper() == "EXTRAS" for s in extras)


def test_accomm_specs_filters_correctly(merged):
    b001 = next(b for b in merged if b.booking_no == "B001")
    accomm = b001.accomm_specs()
    assert all(s.spec_type.upper() == "ACCOMM" for s in accomm)


def test_spec_type_preserved(merged):
    b001 = next(b for b in merged if b.booking_no == "B001")
    types = {s.spec_type for s in b001.specs}
    assert "EXTRAS" in types
    assert "ACCOMM" in types


def test_spec_booking_no_matches(merged):
    for booking in merged:
        for spec in booking.specs:
            assert spec.booking_no == booking.booking_no
