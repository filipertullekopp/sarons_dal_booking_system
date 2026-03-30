"""Tests for vehicle classification and spot count derivation."""
import pytest

from saronsdal.ingestion.specification_reader import load_specifications
from saronsdal.models.raw import RawBooking, RawSpec
from saronsdal.normalization.equipment import classify_vehicle


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


def _extras_spec(specification: str, booking_no: str = "TEST001") -> RawSpec:
    return RawSpec(
        booking_no=booking_no,
        spec_type="EXTRAS",
        specification=specification,
        room_no="",
        guests=None,
        comment="",
        units=None,
        unit_price=None,
        status="Confirmed",
    )


# ---------------------------------------------------------------------------
# Classification from EXTRAS specification rows
# ---------------------------------------------------------------------------

def test_caravan_from_campingvogn_spec():
    booking = _make_booking(
        raw_length="7.5",
        specs=[_extras_spec("Campingvogn")],
    )
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "caravan"
    assert unit.parse_confidence > 0.5


def test_motorhome_from_bobil_spec():
    booking = _make_booking(
        raw_length="8.0",
        specs=[_extras_spec("Bobil")],
    )
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "motorhome"


def test_tent_large_from_spec():
    booking = _make_booking(
        raw_length="4.10x7",
        specs=[_extras_spec("Telt - stort (3-8 personer)")],
    )
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "tent"
    assert unit.spec_size_hint == "large"


def test_tent_small_from_spec():
    booking = _make_booking(specs=[_extras_spec("Telt - lite (1-2 personer)")])
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "tent"
    assert unit.spec_size_hint == "small"


def test_camplet_from_spec():
    booking = _make_booking(specs=[_extras_spec("Camplet")])
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "camplet"


def test_indoor_spec_returns_unknown():
    booking = _make_booking(specs=[_extras_spec("Internatet")])
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "unknown"
    # Indoor is not flagged as low-confidence-vehicle-type — it's a known booking type
    assert unit.parse_confidence == pytest.approx(0.3 * 0.6)


def test_unknown_spec_value():
    booking = _make_booking(specs=[_extras_spec("Youthplanet")])
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "unknown"


# ---------------------------------------------------------------------------
# Fallback: registration number heuristics
# ---------------------------------------------------------------------------

def test_plate_fallback_gives_caravan():
    booking = _make_booking(regnr="AB12345")
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "caravan"
    assert unit.parse_confidence == pytest.approx(0.60 * 0.6)  # dims.conf=0, type only


def test_telt_in_regnr_gives_tent():
    booking = _make_booking(regnr="Telt")
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "tent"


def test_camplet_in_regnr():
    booking = _make_booking(regnr="camp-let")
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "camplet"


def test_no_spec_no_plate_gives_unknown():
    booking = _make_booking()
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "unknown"
    assert "low_confidence_vehicle_type" in unit.review_flags


# ---------------------------------------------------------------------------
# Width and spot count derivation
# ---------------------------------------------------------------------------

def test_caravan_no_fortelt_2_spots():
    # 2.75 m ≤ 3.0 m narrow threshold → 2 spots
    booking = _make_booking(
        raw_length="7.5",
        has_fortelt="Nei",
        specs=[_extras_spec("Campingvogn")],
    )
    unit = classify_vehicle(booking)
    assert unit.body_width_m == pytest.approx(2.75)
    assert unit.total_width_m == pytest.approx(2.75)
    assert unit.required_spot_count == 2


def test_caravan_with_fortelt_uses_default_width():
    # 2.75 + 2.5 (default fortelt) = 5.25 m → (3, 6] → 3 spots
    booking = _make_booking(
        raw_length="7.5",
        has_fortelt="Ja",
        specs=[_extras_spec("Campingvogn")],
    )
    unit = classify_vehicle(booking)
    assert unit.has_fortelt is True
    assert unit.fortelt_width_m == pytest.approx(2.5)
    assert unit.total_width_m == pytest.approx(5.25)
    assert unit.required_spot_count == 3


def test_caravan_with_parsed_fortelt_width():
    # When the length field contains "pluss telt 3.1 ut fra vogn",
    # fortelt_width_m is parsed from the dimension string
    booking = _make_booking(
        raw_length="7.0 m, pluss telt 3,10 ut fra vogn",
        has_fortelt="Ja",
        specs=[_extras_spec("Campingvogn")],
    )
    unit = classify_vehicle(booking)
    assert unit.fortelt_width_m == pytest.approx(3.10)
    # total_width = 2.75 + 3.10 = 5.85 → (3, 6] → 3 spots
    assert unit.total_width_m == pytest.approx(5.85)
    assert unit.required_spot_count == 3


def test_caravan_very_wide_with_fortelt_4_spots():
    # Need total_width > 6.0. Use a wide parsed fortelt: 2.75 + 3.5 = 6.25 → but 6.25 > 6.0 → 4 spots
    # We can't directly inject parsed dims, so build a booking that triggers the parsed fortelt
    booking = _make_booking(
        raw_length="8.0 m, pluss telt 4,0 ut fra vogn",
        has_fortelt="Ja",
        specs=[_extras_spec("Campingvogn")],
    )
    unit = classify_vehicle(booking)
    # total_width = 2.75 + 4.0 = 6.75 → (6, 9] → 4 spots
    assert unit.total_width_m == pytest.approx(6.75)
    assert unit.required_spot_count == 4


def test_tent_width_from_cross_dimension():
    # 4.10x7 → width=4.10, length=7.0; tent body_width = dims.width_m = 4.10
    # no fortelt → total_width = 4.10 → (3, 6] → 3 spots
    booking = _make_booking(
        raw_length="4.10x7",
        specs=[_extras_spec("Telt - stort (3-8 personer)")],
    )
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "tent"
    assert unit.body_width_m == pytest.approx(4.10)
    assert unit.total_width_m == pytest.approx(4.10)
    assert unit.required_spot_count == 3


def test_tent_single_dimension_flag():
    # Single dimension given for tent — cannot determine width axis
    booking = _make_booking(
        raw_length="640 cm",
        specs=[_extras_spec("Telt - stort (3-8 personer)")],
    )
    unit = classify_vehicle(booking)
    assert unit.vehicle_type == "tent"
    assert unit.body_width_m is None
    assert "tent_single_dimension" in unit.review_flags


# ---------------------------------------------------------------------------
# Registration extraction
# ---------------------------------------------------------------------------

def test_registration_cleaned():
    booking = _make_booking(regnr="AB12345")
    unit = classify_vehicle(booking)
    assert unit.registration == "AB12345"


def test_registration_zero_is_none():
    booking = _make_booking(regnr="0")
    unit = classify_vehicle(booking)
    assert unit.registration is None


def test_registration_dash_is_none():
    booking = _make_booking(regnr="-")
    unit = classify_vehicle(booking)
    assert unit.registration is None


# ---------------------------------------------------------------------------
# has_markise
# ---------------------------------------------------------------------------

def test_has_markise_true():
    booking = _make_booking(has_markise="Ja")
    unit = classify_vehicle(booking)
    assert unit.has_markise is True


def test_has_markise_false():
    booking = _make_booking(has_markise="Nei")
    unit = classify_vehicle(booking)
    assert unit.has_markise is False


# ---------------------------------------------------------------------------
# Conflicting specs
# ---------------------------------------------------------------------------

def test_conflicting_specs_flagged():
    booking = _make_booking(specs=[
        _extras_spec("Campingvogn"),
        _extras_spec("Bobil"),
    ])
    unit = classify_vehicle(booking)
    assert "conflicting_specifications" in unit.review_flags


# ---------------------------------------------------------------------------
# raw_total_width — explicit total width field (new 2026 Sirvoy column)
# ---------------------------------------------------------------------------

def _caravan_booking(**kwargs) -> RawBooking:
    """Caravan booking with a known registration (type=caravan via spec)."""
    return _make_booking(
        specs=[_extras_spec("Campingvogn")],
        raw_length="7.5",
        **kwargs,
    )


def test_explicit_total_width_sets_total_width_m():
    # raw_total_width = "2.8" → total_width_m = 2.8 m ≤ 3.0 → 2 spots
    booking = _caravan_booking(raw_total_width="2.8")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(2.8, abs=0.01)
    assert unit.required_spot_count == 2
    assert "total_width_from_explicit_field" in unit.review_flags


def test_explicit_total_width_takes_precedence_over_inferred():
    # Caravan body=2.75 + default fortelt=2.5 → inferred=5.25 m → 3 spots.
    # But explicit total width = 3.0 m → 2 spots.  Explicit wins.
    booking = _caravan_booking(raw_total_width="3.0", has_fortelt="Ja")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(3.0, abs=0.01)
    assert unit.required_spot_count == 2
    assert "total_width_from_explicit_field" in unit.review_flags


def test_explicit_total_width_comma_decimal():
    # "4,5" → 4.5 m → 3 spots
    booking = _caravan_booking(raw_total_width="4,5")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(4.5, abs=0.01)
    assert unit.required_spot_count == 3


def test_explicit_total_width_cm_unit():
    # "450 cm" → 4.5 m → 3 spots
    booking = _caravan_booking(raw_total_width="450 cm")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(4.5, abs=0.01)
    assert unit.required_spot_count == 3


def test_explicit_total_width_mm_unit():
    # "4500 mm" → 4.5 m → 3 spots
    booking = _caravan_booking(raw_total_width="4500 mm")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(4.5, abs=0.01)
    assert unit.required_spot_count == 3


def test_explicit_total_width_bare_4digit_assumed_mm():
    # "4500" no unit → assumed mm → 4.5 m → 3 spots
    booking = _caravan_booking(raw_total_width="4500")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(4.5, abs=0.01)
    assert unit.required_spot_count == 3


def test_explicit_total_width_out_of_bounds_flagged():
    # "25" → parse_dimensions → 25.0 m (bare, < 100) → outside [1, 9] → flagged
    booking = _caravan_booking(raw_total_width="25")
    unit = classify_vehicle(booking)
    assert "width_out_of_bounds" in unit.review_flags
    # When out-of-bounds, fall back to inferred width (caravan body = 2.75 m, no fortelt → 2.75 m, 2 spots)
    assert unit.total_width_m == pytest.approx(2.75, abs=0.01)
    assert unit.required_spot_count == 2


def test_explicit_total_width_zero_treated_as_missing():
    # "0" → treated as missing → fall back to inferred
    booking = _caravan_booking(raw_total_width="0")
    unit = classify_vehicle(booking)
    assert "total_width_from_explicit_field" not in unit.review_flags
    assert unit.total_width_m == pytest.approx(2.75, abs=0.01)


def test_missing_raw_total_width_falls_back_to_inferred():
    # raw_total_width="" → falls back to inferred caravan width
    booking = _caravan_booking(raw_total_width="")
    unit = classify_vehicle(booking)
    assert "total_width_from_explicit_field" not in unit.review_flags
    assert unit.total_width_m == pytest.approx(2.75, abs=0.01)
    assert unit.required_spot_count == 2


def test_explicit_width_six_metres_four_spots():
    # 6.5 m → 4 spots
    booking = _caravan_booking(raw_total_width="6.5")
    unit = classify_vehicle(booking)
    assert unit.total_width_m == pytest.approx(6.5, abs=0.01)
    assert unit.required_spot_count == 4
