"""Tests for placement preference and group signal extraction."""
import pytest

from saronsdal.models.raw import RawBooking
from saronsdal.normalization.preferences import extract_group_signals, extract_spot_request


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_booking(**kwargs) -> RawBooking:
    defaults = dict(
        booking_no="TEST001",
        booking_source="Nettside",
        booking_date="2024-03-01",
        check_in="2024-07-01",
        check_out="2024-07-08",
        first_name="Test",
        last_name="User",
        company="privat",
        num_guests="2",
        num_rooms="1",
        language="no",
        confirmed="Ja",
        phone="",
        email="",
        guest_message="",
        comment="",
        regnr="0",
        raw_length="",
        raw_width="",
        has_markise="Nei",
        has_fortelt="Nei",
        raw_location_wish="",
        raw_helper="",
        specs=[],
    )
    defaults.update(kwargs)
    return RawBooking(**defaults)


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def test_section_from_location_wish():
    booking = _make_booking(raw_location_wish="Furutoppen")
    req = extract_spot_request(booking)
    assert "Furutoppen" in req.preferred_sections


def test_section_furulunden_from_guest_message():
    booking = _make_booking(guest_message="Vil gjerne stå i Furulunden")
    req = extract_spot_request(booking)
    assert "Furulunden" in req.preferred_sections


def test_section_alias_resolves_to_canonical():
    # "Furuterrassen" is an alias for "Furutoppen"
    booking = _make_booking(raw_location_wish="Furuterrassen")
    req = extract_spot_request(booking)
    assert "Furutoppen" in req.preferred_sections


def test_section_mojibake_alias_vaardalen():
    # "VÃ¥rdalen" is a mojibake alias for "Vårdalen"
    booking = _make_booking(raw_location_wish="VÃ¥rdalen")
    req = extract_spot_request(booking)
    assert "Vårdalen" in req.preferred_sections


def test_multiple_sections_extracted():
    booking = _make_booking(
        guest_message="Enten Furulunden eller Elvebredden"
    )
    req = extract_spot_request(booking)
    assert "Furulunden" in req.preferred_sections
    assert "Elvebredden" in req.preferred_sections


# ---------------------------------------------------------------------------
# Spot ID extraction
# ---------------------------------------------------------------------------

def test_single_spot_id():
    booking = _make_booking(raw_location_wish="B12")
    req = extract_spot_request(booking)
    assert "B12" in req.preferred_spot_ids


def test_spot_id_range_dash():
    booking = _make_booking(raw_location_wish="D25-D27")
    req = extract_spot_request(booking)
    assert req.preferred_spot_ids == ["D25", "D26", "D27"]


def test_spot_id_range_til():
    booking = _make_booking(raw_location_wish="D25 til D27")
    req = extract_spot_request(booking)
    assert "D25" in req.preferred_spot_ids
    assert "D27" in req.preferred_spot_ids
    assert len(req.preferred_spot_ids) == 3


def test_spot_id_numeric_range_no_prefix():
    # "D25 til 27" — second number has no letter prefix, use first
    booking = _make_booking(raw_location_wish="D25 til 27")
    req = extract_spot_request(booking)
    assert "D25" in req.preferred_spot_ids
    assert "D27" in req.preferred_spot_ids


def test_spot_ids_deduped():
    booking = _make_booking(
        raw_location_wish="A1",
        guest_message="Spot A1 please",
    )
    req = extract_spot_request(booking)
    assert req.preferred_spot_ids.count("A1") == 1


# ---------------------------------------------------------------------------
# Near-mention / "together with" extraction
# ---------------------------------------------------------------------------

def test_near_fragment_naer():
    booking = _make_booking(guest_message="Bo nær familien Olsen")
    req = extract_spot_request(booking)
    assert any("Olsen" in frag for frag in req.raw_near_texts)


def test_near_fragment_samen_med():
    booking = _make_booking(guest_message="Ønsker å stå sammen med Familien Hansen")
    req = extract_spot_request(booking)
    assert any("Hansen" in frag for frag in req.raw_near_texts)


def test_near_fragment_deduped():
    # Same name in location_wish and guest_message
    booking = _make_booking(
        raw_location_wish="nær familien Berg",
        guest_message="nær familien Berg",
    )
    req = extract_spot_request(booking)
    count = sum(1 for f in req.raw_near_texts if "Berg" in f)
    assert count == 1


def test_stopword_only_fragment_not_captured():
    # "gjengen" is a generic stopword
    booking = _make_booking(guest_message="Bo nær gjengen")
    req = extract_spot_request(booking)
    # Should not produce a near fragment since "gjengen" is a stopword
    assert not any("gjengen" in f.lower() for f in req.raw_near_texts)


# ---------------------------------------------------------------------------
# Amenity flags
# ---------------------------------------------------------------------------

def test_amenity_near_toilet():
    booking = _make_booking(guest_message="Vil helst stå nær toalett")
    req = extract_spot_request(booking)
    assert "near_toilet" in req.amenity_flags


def test_amenity_flat():
    booking = _make_booking(guest_message="Trenger flat plass for rullestol")
    req = extract_spot_request(booking)
    assert "flat" in req.amenity_flags


def test_amenity_near_river():
    booking = _make_booking(guest_message="Vil stå nær elven")
    req = extract_spot_request(booking)
    assert "near_river" in req.amenity_flags


def test_amenity_near_forest():
    booking = _make_booking(raw_location_wish="nær skogen")
    req = extract_spot_request(booking)
    assert "near_forest" in req.amenity_flags


# ---------------------------------------------------------------------------
# No preference
# ---------------------------------------------------------------------------

def test_no_preference_flag_when_empty():
    booking = _make_booking()
    req = extract_spot_request(booking)
    assert "no_placement_preference" in req.review_flags
    assert req.parse_confidence == pytest.approx(0.5)


def test_preference_confidence_full_when_section_found():
    booking = _make_booking(raw_location_wish="Furutoppen")
    req = extract_spot_request(booking)
    assert "no_placement_preference" not in req.review_flags
    assert req.parse_confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Group signals — organization
# ---------------------------------------------------------------------------

def test_org_from_company_field():
    booking = _make_booking(company="Betel Hommersåk")
    signals = extract_group_signals(booking)
    assert signals.organization == "Betel Hommersåk"
    assert signals.is_org_private is False


def test_private_company_gives_none():
    booking = _make_booking(company="privat")
    signals = extract_group_signals(booking)
    assert signals.organization is None
    assert signals.is_org_private is True


def test_empty_company_gives_none():
    booking = _make_booking(company="")
    signals = extract_group_signals(booking)
    assert signals.organization is None
    assert signals.is_org_private is True


def test_dash_company_gives_none():
    booking = _make_booking(company="-")
    signals = extract_group_signals(booking)
    assert signals.organization is None


def test_mojibake_company_repaired():
    # "Betel HommersÃ¥k" should be repaired by ftfy to "Betel Hommersåk"
    booking = _make_booking(company="Betel HommersÃ¥k")
    signals = extract_group_signals(booking)
    assert signals.organization == "Betel Hommersåk"


# ---------------------------------------------------------------------------
# Group signals — group_field from location wish
# ---------------------------------------------------------------------------

def test_group_field_from_non_section_location_wish():
    # "Bjørnebanden" doesn't match any section name → treated as group signal
    booking = _make_booking(raw_location_wish="Bjørnebanden")
    signals = extract_group_signals(booking)
    assert signals.group_field is not None
    assert "Bjørnebanden" in signals.group_field


def test_section_name_not_treated_as_group():
    # Known section name should NOT become a group_field
    booking = _make_booking(raw_location_wish="Furutoppen")
    signals = extract_group_signals(booking)
    assert signals.group_field is None


def test_section_plus_spotid_not_treated_as_group():
    # "Elvebredden A01" pattern → has section, should not become group
    booking = _make_booking(raw_location_wish="Elvebredden A01")
    signals = extract_group_signals(booking)
    # Has a section, so group_field should be None
    assert signals.group_field is None


def test_near_texts_in_group_signals():
    booking = _make_booking(guest_message="nær familien Hansen")
    signals = extract_group_signals(booking)
    assert any("Hansen" in frag for frag in signals.near_text_fragments)


# ---------------------------------------------------------------------------
# Section row (subsection) extraction
# ---------------------------------------------------------------------------

def test_section_row_from_location_wish():
    """'Vårdalen D' → section Vårdalen, row D."""
    booking = _make_booking(raw_location_wish="Vårdalen D")
    req = extract_spot_request(booking)
    assert "Vårdalen" in req.preferred_sections
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Vårdalen", "D") in rows


def test_section_row_furulunden():
    """'Furulunden B' → section Furulunden, row B."""
    booking = _make_booking(raw_location_wish="Furulunden B")
    req = extract_spot_request(booking)
    assert "Furulunden" in req.preferred_sections
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Furulunden", "B") in rows


def test_section_row_elvebredden():
    """'Elvebredden C' → section Elvebredden, row C."""
    booking = _make_booking(raw_location_wish="Elvebredden C")
    req = extract_spot_request(booking)
    assert "Elvebredden" in req.preferred_sections
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Elvebredden", "C") in rows


def test_section_row_fjellterrassen():
    """'Fjellterrassen A' → section Fjellterrassen, row A."""
    booking = _make_booking(raw_location_wish="Fjellterrassen A")
    req = extract_spot_request(booking)
    assert "Fjellterrassen" in req.preferred_sections
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Fjellterrassen", "A") in rows


def test_section_only_no_row():
    """Section name without a trailing letter → no section row."""
    booking = _make_booking(raw_location_wish="Furutoppen")
    req = extract_spot_request(booking)
    assert "Furutoppen" in req.preferred_sections
    assert req.preferred_section_rows == []


def test_section_row_from_guest_message():
    """Row preference in free-text comment is captured."""
    booking = _make_booking(guest_message="Vi ønsker å stå i Elvebredden B")
    req = extract_spot_request(booking)
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Elvebredden", "B") in rows


def test_multiple_section_rows_same_section():
    """'Furulunden B og Furulunden C' → two separate SectionRow objects."""
    booking = _make_booking(guest_message="Enten Furulunden B eller Furulunden C")
    req = extract_spot_request(booking)
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Furulunden", "B") in rows
    assert ("Furulunden", "C") in rows


def test_multiple_section_rows_different_sections():
    """'Furulunden B' and 'Vårdalen D' each produce their own SectionRow."""
    booking = _make_booking(guest_message="Furulunden B eller Vårdalen D")
    req = extract_spot_request(booking)
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Furulunden", "B") in rows
    assert ("Vårdalen", "D") in rows


def test_spot_id_not_treated_as_row():
    """'Elvebredden D25' — D is followed by digit 25, so NOT a row."""
    booking = _make_booking(raw_location_wish="Elvebredden D25")
    req = extract_spot_request(booking)
    assert "Elvebredden" in req.preferred_sections
    assert "D25" in req.preferred_spot_ids
    assert req.preferred_section_rows == []


def test_standalone_letter_not_treated_as_row():
    """A lone letter with no section name should not produce a row."""
    booking = _make_booking(raw_location_wish="D")
    req = extract_spot_request(booking)
    assert req.preferred_section_rows == []


def test_lowercase_row_letter_normalised_to_uppercase():
    """'Furulunden b' (lowercase) → row stored as uppercase 'B'."""
    booking = _make_booking(raw_location_wish="Furulunden b")
    req = extract_spot_request(booking)
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Furulunden", "B") in rows


def test_mojibake_section_row():
    """Mojibake alias 'VÃ¥rdalen D' should still extract row D."""
    booking = _make_booking(raw_location_wish="VÃ¥rdalen D")
    req = extract_spot_request(booking)
    rows = [(r.section, r.row) for r in req.preferred_section_rows]
    assert ("Vårdalen", "D") in rows


def test_section_row_deduplication():
    """Same section+row in both location_wish and guest_message → only one entry."""
    booking = _make_booking(
        raw_location_wish="Furulunden B",
        guest_message="Ønsker Furulunden B",
    )
    req = extract_spot_request(booking)
    furulunden_b = [r for r in req.preferred_section_rows
                    if r.section == "Furulunden" and r.row == "B"]
    assert len(furulunden_b) == 1
