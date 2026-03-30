"""Tests for the dimension parser.

All expected output values are in metres.
"""
import pytest

from saronsdal.normalization.length_parser import parse_dimensions


# ---------------------------------------------------------------------------
# Empty / missing inputs
# ---------------------------------------------------------------------------

def test_empty_string_is_missing():
    d = parse_dimensions("")
    assert d.length_m is None
    assert d.confidence == 0.0
    assert "missing_dimensions" in d.flags


def test_zero_string_is_missing():
    d = parse_dimensions("0")
    assert d.length_m is None
    assert d.confidence == 0.0
    assert "missing_dimensions" in d.flags


def test_dash_is_missing():
    d = parse_dimensions("-")
    assert d.length_m is None
    assert "missing_dimensions" in d.flags


# ---------------------------------------------------------------------------
# Camp-let detection
# ---------------------------------------------------------------------------

def test_camplet_keyword():
    d = parse_dimensions("Camp-let")
    assert d.length_m is None
    assert d.confidence == pytest.approx(0.5)
    assert "camplet_no_dimensions" in d.flags


def test_camplet_no_hyphen():
    d = parse_dimensions("Camplet")
    assert "camplet_no_dimensions" in d.flags


# ---------------------------------------------------------------------------
# Simple numeric formats
# ---------------------------------------------------------------------------

def test_decimal_comma():
    d = parse_dimensions("7,5")
    assert d.length_m == pytest.approx(7.5)
    assert d.confidence >= 0.9


def test_decimal_point():
    d = parse_dimensions("9.4")
    assert d.length_m == pytest.approx(9.4)


def test_with_unit_m():
    d = parse_dimensions("9,4 m")
    assert d.length_m == pytest.approx(9.4)


def test_with_unit_meter():
    d = parse_dimensions("8 meter")
    assert d.length_m == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def test_centimetres():
    d = parse_dimensions("750 cm")
    assert d.length_m == pytest.approx(7.5)


def test_three_digit_no_unit_assumed_cm():
    d = parse_dimensions("753")
    assert d.length_m == pytest.approx(7.53)
    assert "assumed_cm" in d.flags


def test_stated_as_m_but_over_20_assumed_cm_typo():
    # "826m" — stated as metres but value > 20, so treat as cm
    d = parse_dimensions("826m")
    assert d.length_m == pytest.approx(8.26)
    assert "assumed_cm_typo" in d.flags


# ---------------------------------------------------------------------------
# Prefix stripping
# ---------------------------------------------------------------------------

def test_telt_prefix_stripped():
    # "Telt 640 cm" → 6.40 m
    d = parse_dimensions("Telt 640 cm")
    assert d.length_m == pytest.approx(6.40)


def test_med_drag_label_stripped():
    # "(med drag)" parenthetical is stripped
    d = parse_dimensions("9,51 m (med drag)")
    assert d.length_m == pytest.approx(9.51)


def test_parenthetical_inkl_stripped():
    d = parse_dimensions("8.5m (inkl. fortelt)")
    assert d.length_m == pytest.approx(8.5)


# ---------------------------------------------------------------------------
# Fortelt lateral extraction
# ---------------------------------------------------------------------------

def test_fortelt_extracted_from_length_field():
    raw = "770 cm, pluss telt 3,10 ut fra vogn"
    d = parse_dimensions(raw)
    assert d.length_m == pytest.approx(7.70)
    assert d.fortelt_width_m == pytest.approx(3.10)
    assert "fortelt_dimension_parsed" in d.flags


def test_fortelt_with_plus_sign():
    raw = "8m + fortelt 2.5m"
    d = parse_dimensions(raw)
    assert d.length_m == pytest.approx(8.0)
    assert d.fortelt_width_m == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Labeled format: "Lenge Xm bredde Ym"
# ---------------------------------------------------------------------------

def test_labeled_lenge_bredde():
    d = parse_dimensions("Lenge 8m bredde 6m")
    assert d.length_m == pytest.approx(8.0)
    assert d.width_m == pytest.approx(6.0)
    assert "labeled_dimensions" in d.flags


def test_labeled_lengde_bredde():
    d = parse_dimensions("Lengde 7,5 bredde 4")
    assert d.length_m == pytest.approx(7.5)
    assert d.width_m == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Cross-dimension format: "4.10x7"
# ---------------------------------------------------------------------------

def test_cross_dimension_larger_is_length():
    # Tent: smaller = width, larger = length
    d = parse_dimensions("4.10x7")
    assert d.length_m == pytest.approx(7.0)
    assert d.width_m == pytest.approx(4.10)
    assert "two_dimensions_parsed" in d.flags


def test_cross_dimension_with_spaces():
    d = parse_dimensions("4 x 7")
    assert d.length_m == pytest.approx(7.0)
    assert d.width_m == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Separate width field
# ---------------------------------------------------------------------------

def test_separate_width_field():
    d = parse_dimensions("7.5", raw_width="3.5")
    assert d.length_m == pytest.approx(7.5)
    assert d.width_m == pytest.approx(3.5)


def test_width_field_ignored_when_already_known():
    # cross-dimension format already gives width; raw_width should not override
    d = parse_dimensions("4.10x7", raw_width="5.0")
    assert d.width_m == pytest.approx(4.10)


# ---------------------------------------------------------------------------
# Approximate qualifier
# ---------------------------------------------------------------------------

def test_ca_prefix_adds_approximate_flag():
    d = parse_dimensions("ca 7m")
    assert d.length_m == pytest.approx(7.0)
    assert "approximate" in d.flags
    assert d.confidence < 1.0


def test_tror_adds_approximate_flag():
    d = parse_dimensions("tror det er 8m")
    assert "approximate" in d.flags


# ---------------------------------------------------------------------------
# Sanity bounds
# ---------------------------------------------------------------------------

def test_too_short_flag():
    d = parse_dimensions("0.5 m")
    assert "too_short" in d.flags
    assert d.confidence < 0.9


def test_too_long_flag():
    d = parse_dimensions("20 m")
    assert "too_long" in d.flags


def test_plausible_range_no_sanity_flags():
    d = parse_dimensions("8.5 m")
    assert "too_short" not in d.flags
    assert "too_long" not in d.flags


# ---------------------------------------------------------------------------
# Unit inference: bare 4-digit values → mm (real failure modes from 2026 data)
# ---------------------------------------------------------------------------

def test_bare_4digit_treated_as_mm():
    # "8650" with no unit → 8650 mm = 8.65 m  (was incorrectly 86.5 m before fix)
    d = parse_dimensions("8650")
    assert d.length_m == pytest.approx(8.65, abs=0.01)
    assert "assumed_mm" in d.flags
    assert "too_long" not in d.flags


def test_bare_4digit_trailer_annotation_stripped_then_mm():
    # "7530 m/draget" → strip "m/draget", then 7530 mm = 7.53 m
    d = parse_dimensions("7530 m/draget")
    assert d.length_m == pytest.approx(7.53, abs=0.01)
    assert "assumed_mm" in d.flags
    assert "too_long" not in d.flags


def test_bare_3digit_treated_as_cm():
    # "753" → 753 cm = 7.53 m (existing behaviour, unchanged)
    d = parse_dimensions("753")
    assert d.length_m == pytest.approx(7.53, abs=0.01)
    assert "assumed_cm" in d.flags


def test_borderline_too_long_not_rescaled():
    # "17" is above 15 m but below 2×15=30 m → stays as 17 m with "too_long" flag.
    # Must NOT be auto-rescaled (1.7 m would be absurd for a motorhome).
    d = parse_dimensions("17")
    assert d.length_m == pytest.approx(17.0, abs=0.01)
    assert "too_long" in d.flags
    assert "rescaled_to_plausible" not in d.flags


def test_clearly_implausible_rescaled_div10():
    # Sanity rescue: 86.5 m (>30 m) → /10 = 8.65 m (plausible).
    # This triggers step 11b (defence-in-depth; shouldn't occur in normal flow
    # because 8650-bare now takes the assumed_mm path).
    d = parse_dimensions("8650 cm")   # explicit cm → 86.5 m → rescue → 8.65 m
    assert d.length_m == pytest.approx(8.65, abs=0.01)
    assert "rescaled_to_plausible" in d.flags
    assert "rescaled_div10" in d.flags
    assert "too_long" not in d.flags


def test_trailer_annotation_variant_dot():
    # "8.5 m. drag" → strip "m. drag" annotation → 8.5 m (plausible, no clip)
    d = parse_dimensions("8.5 m. drag")
    assert d.length_m == pytest.approx(8.5, abs=0.01)
    assert "too_long" not in d.flags


def test_mm_explicit_unit():
    # Explicit "mm" suffix always divided by 1000.
    d = parse_dimensions("8500 mm")
    assert d.length_m == pytest.approx(8.5, abs=0.01)
    assert "assumed_mm" not in d.flags   # explicit unit, no inference flag


def test_cm_explicit_unit():
    # Explicit "cm" suffix always divided by 100.
    d = parse_dimensions("850 cm")
    assert d.length_m == pytest.approx(8.5, abs=0.01)
    assert "assumed_cm" not in d.flags   # explicit unit, no inference flag
