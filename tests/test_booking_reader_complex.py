"""Tests for booking_reader robustness with complex quoting and embedded commas."""
import logging
from unittest.mock import patch

import pandas as pd
import pytest

from tests.conftest import FIXTURES_DIR
from saronsdal.ingestion.booking_reader import load_bookings_basic

MINI_COMPLEX = FIXTURES_DIR / "mini_basic_complex.csv"


# ---------------------------------------------------------------------------
# Embedded commas inside quoted fields
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def complex_bookings():
    return load_bookings_basic(MINI_COMPLEX)


def test_complex_load_count(complex_bookings):
    assert len(complex_bookings) == 3


def test_embedded_comma_in_guest_message(complex_bookings):
    # C001 guest message: "Vi ønsker å stå nær familien, helst i Furulunden"
    c001 = next(b for b in complex_bookings if b.booking_no == "C001")
    assert "," in c001.guest_message
    assert "Furulunden" in c001.guest_message


def test_embedded_comma_in_company(complex_bookings):
    # C002 company: "Betel, Hommersåk" — comma inside a quoted company name
    c002 = next(b for b in complex_bookings if b.booking_no == "C002")
    assert c002.company == "Betel, Hommersåk"


def test_embedded_comma_in_guest_message_c002(complex_bookings):
    # C002 guest message: "Ønsker flat plass, nær toalett"
    c002 = next(b for b in complex_bookings if b.booking_no == "C002")
    assert "," in c002.guest_message
    assert "toalett" in c002.guest_message


def test_embedded_comma_in_comment_c003(complex_bookings):
    # C003 guest message contains multiple commas
    c003 = next(b for b in complex_bookings if b.booking_no == "C003")
    assert c003.guest_message.count(",") >= 2


def test_regnr_preserved_c001(complex_bookings):
    c001 = next(b for b in complex_bookings if b.booking_no == "C001")
    assert c001.regnr == "AB99999"


def test_fortelt_ja_c002(complex_bookings):
    c002 = next(b for b in complex_bookings if b.booking_no == "C002")
    assert c002.has_fortelt == "Ja"


# ---------------------------------------------------------------------------
# Python-engine fallback path
# ---------------------------------------------------------------------------

def test_python_engine_fallback_is_used_on_parser_error(tmp_path, caplog):
    """When the C engine raises ParserError the Python engine is used instead."""
    # Copy the complex fixture so we have a real file to open
    import shutil
    dest = tmp_path / "fallback_test.csv"
    shutil.copy(MINI_COMPLEX, dest)

    original_read_csv = pd.read_csv
    call_count = {"n": 0}

    def patched_read_csv(path_arg, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Simulate C engine failure on the first attempt
            raise pd.errors.ParserError("simulated C engine failure")
        return original_read_csv(path_arg, **kwargs)

    with patch("saronsdal.ingestion.booking_reader.pd.read_csv", side_effect=patched_read_csv):
        with caplog.at_level(logging.WARNING, logger="saronsdal.ingestion.booking_reader"):
            bookings = load_bookings_basic(dest)

    # Fallback warning was emitted
    assert any("Python engine" in msg for msg in caplog.messages)
    # Bookings were still loaded successfully
    assert len(bookings) == 3


def test_python_engine_fallback_preserves_embedded_commas(tmp_path):
    """Fallback path must still correctly parse quoted fields with commas."""
    import shutil
    dest = tmp_path / "fallback_commas.csv"
    shutil.copy(MINI_COMPLEX, dest)

    original_read_csv = pd.read_csv
    call_count = {"n": 0}

    def patched_read_csv(path_arg, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise pd.errors.ParserError("simulated C engine failure")
        return original_read_csv(path_arg, **kwargs)

    with patch("saronsdal.ingestion.booking_reader.pd.read_csv", side_effect=patched_read_csv):
        bookings = load_bookings_basic(dest)

    c002 = next(b for b in bookings if b.booking_no == "C002")
    assert c002.company == "Betel, Hommersåk"
