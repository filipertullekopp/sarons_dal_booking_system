"""Tests for column-name schema detection and resolution."""
import pytest

from saronsdal.ingestion.schema import SchemaVersion, detect_schema_version, resolve_columns


# ---------------------------------------------------------------------------
# detect_schema_version
# ---------------------------------------------------------------------------

def test_detect_basic_export_by_bredde():
    cols = [
        "Booking No.", "Booking source", "Check-in", "Check-out",
        "Bredde på telt/fortelt/markise", "Lengde på bobil, campingvogn eller telt",
    ]
    assert detect_schema_version(cols) == SchemaVersion.BASIC_EXPORT


def test_detect_basic_export_by_fortelt_col():
    cols = [
        "Booking No.", "Booking source",
        "Fortelt til bobil eller campingvogn",
    ]
    assert detect_schema_version(cols) == SchemaVersion.BASIC_EXPORT


def test_detect_basic_export_mojibake_bredde():
    # mojibake column name still triggers BASIC_EXPORT
    cols = [
        "Booking No.",
        "Bredde pÃ¥ telt/fortelt/markise",
    ]
    assert detect_schema_version(cols) == SchemaVersion.BASIC_EXPORT


def test_detect_sirvoy_2024_by_gruppe():
    cols = [
        "Booking No.", "Booking source",
        "For deg som kommer som gruppe, hvilke gruppe tilhører du?",
    ]
    assert detect_schema_version(cols) == SchemaVersion.SIRVOY_2024


def test_detect_sirvoy_2024_mojibake_gruppe():
    cols = [
        "Booking No.",
        "For deg som kommer som gruppe, hvilke gruppe tilhÃ¸rer du?",
    ]
    assert detect_schema_version(cols) == SchemaVersion.SIRVOY_2024


def test_group_field_takes_priority_over_bredde():
    # Real 2026 export: location-wish column contains "Elvebredden" which has
    # "bredde" as a substring, but the group-field column is also present.
    # SIRVOY_2024 must win.
    cols = [
        "Booking No.", "Booking source",
        "Ønsket plassering (Furutoppen, Furulunden, Elvebredden, Fjelltarassen, Vårdalen, Egelandsletta)",
        "For deg som kommer som gruppe, hvilke gruppe tilhører du?",
        "Lengde på bobil, campingvogn (totalt med drag) eller telt",
    ]
    assert detect_schema_version(cols) == SchemaVersion.SIRVOY_2024


def test_elvebredden_in_column_name_does_not_trigger_basic_export():
    # "bredde" inside "Elvebredden" must not falsely match the bredde-width check.
    cols = [
        "Booking No.", "Booking source",
        "Ønsket plassering (Furutoppen, Furulunden, Elvebredden, Fjelltarassen, Vårdalen, Egelandsletta)",
    ]
    # No group field, no fortelt col, no real Bredde-width col → UNKNOWN
    assert detect_schema_version(cols) == SchemaVersion.UNKNOWN


def test_detect_unknown():
    cols = ["id", "name", "date"]
    assert detect_schema_version(cols) == SchemaVersion.UNKNOWN


# ---------------------------------------------------------------------------
# resolve_columns — BASIC_EXPORT
# ---------------------------------------------------------------------------

_BASIC_COLS = [
    "Booking No.", "Booking source", "Booking Date", "Check-in", "Check-out",
    "First Name", "Last Name", "Company", "Number of Guests", "Number of Rooms",
    "Language", "Confirmed", "Phone", "Email", "Guest Message", "Comment",
    "Regnr på bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
    "Lengde på bobil, campingvogn eller telt",
    "Bredde på telt/fortelt/markise",
    "Markise til bobil/campingvogn",
    "Fortelt til bobil eller campingvogn",
    "Ønsket plassering (Bergesli kun for Youthplanet)",
    "Skal være medhjelper i sommer",
]


def test_resolve_basic_export_version():
    col_map, version = resolve_columns(_BASIC_COLS)
    assert version == SchemaVersion.BASIC_EXPORT


def test_resolve_basic_export_common_fields():
    col_map, _ = resolve_columns(_BASIC_COLS)
    assert col_map["booking_no"] == "Booking No."
    assert col_map["check_in"] == "Check-in"
    assert col_map["check_out"] == "Check-out"
    assert col_map["first_name"] == "First Name"
    assert col_map["last_name"] == "Last Name"
    assert col_map["language"] == "Language"
    assert col_map["confirmed"] == "Confirmed"


def test_resolve_basic_export_equipment_fields():
    col_map, _ = resolve_columns(_BASIC_COLS)
    assert col_map["raw_length"] is not None
    assert "Lengde" in col_map["raw_length"]
    assert col_map["raw_width"] is not None
    assert "Bredde" in col_map["raw_width"]
    assert col_map["has_fortelt"] is not None
    assert col_map["has_markise"] is not None


def test_resolve_basic_export_no_group_field():
    col_map, _ = resolve_columns(_BASIC_COLS)
    # group_field column is only in SIRVOY_2024
    assert col_map.get("group_field") is None


# ---------------------------------------------------------------------------
# resolve_columns — mojibake column names
# ---------------------------------------------------------------------------

def test_resolve_mojibake_regnr():
    # mojibake variant of the regnr column
    mojibake_cols = [
        "Booking No.", "Booking source", "Check-in", "Check-out",
        "First Name", "Last Name",
        "Regnr pÃ¥ bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
        "Bredde pÃ¥ telt/fortelt/markise",
    ]
    col_map, _ = resolve_columns(mojibake_cols)
    assert col_map["regnr"] is not None


def test_resolve_mojibake_width():
    mojibake_cols = [
        "Booking No.", "Booking source",
        "Bredde pÃ¥ telt/fortelt/markise",
        "Fortelt til bobil eller campingvogn",
    ]
    col_map, _ = resolve_columns(mojibake_cols)
    assert col_map["raw_width"] is not None


# ---------------------------------------------------------------------------
# resolve_columns — SIRVOY_2024
# ---------------------------------------------------------------------------

_SIRVOY_2024_COLS = [
    "Booking No.", "Booking source", "Booking Date", "Check-in", "Check-out",
    "First Name", "Last Name", "Company", "Number of Guests", "Number of Rooms",
    "Language", "Confirmed", "Phone", "Email", "Guest Message", "Comment",
    "Regnr på bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
    "Lengde på bobil, campingvogn (totalt med drag) eller telt ",
    "For deg som kommer som gruppe, hvilke gruppe tilhører du?",
    "Skriv antall som skal være medhjelpere og hvor de skal hjelpe til:",
]


def test_resolve_sirvoy_2024_has_group_field():
    col_map, version = resolve_columns(_SIRVOY_2024_COLS)
    assert version == SchemaVersion.SIRVOY_2024
    assert col_map.get("group_field") is not None


def test_resolve_sirvoy_2024_no_bredde():
    col_map, _ = resolve_columns(_SIRVOY_2024_COLS)
    # SIRVOY_2024 has no separate width column
    assert col_map.get("raw_width") is None


def test_resolve_sirvoy_2024_length_with_drag():
    col_map, _ = resolve_columns(_SIRVOY_2024_COLS)
    assert col_map["raw_length"] is not None
    assert "med drag" in col_map["raw_length"].lower()


# ---------------------------------------------------------------------------
# resolve_columns — 2026 Sirvoy export column names
# ---------------------------------------------------------------------------

_SIRVOY_2026_COLS = [
    "Booking date", "Booking source", "Booking no.", "Check-in", "Check-out",
    "First name", "Last name", "Company", "Number of guests", "Number of rooms",
    "Language", "Confirmed", "Phone", "Email",
    "Guest comment",   # 2026: was "Guest Message"
    "Internal note",   # 2026: was "Comment"
    "Regnr på bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
    "Lengde på bobil, campingvogn (totalt med drag) eller telt",
    "Ønsket plassering (Furutoppen, Furulunden, Elvebredden, Fjelltarassen, Vårdalen, Egelandsletta)",
    "Skriv antall voksne i boenheten og antall som skal være medhjelpere og hvor de skal hjelpe til:",
    "For deg som kommer som gruppe, hvilke gruppe tilhører du?",
]


def test_resolve_2026_detects_sirvoy_2024():
    # Despite "Elvebredden" containing "bredde", must detect as SIRVOY_2024
    _, version = resolve_columns(_SIRVOY_2026_COLS)
    assert version == SchemaVersion.SIRVOY_2024


def test_resolve_2026_guest_comment_maps_to_guest_message():
    col_map, _ = resolve_columns(_SIRVOY_2026_COLS)
    assert col_map["guest_message"] == "Guest comment"


def test_resolve_2026_internal_note_maps_to_comment():
    col_map, _ = resolve_columns(_SIRVOY_2026_COLS)
    assert col_map["comment"] == "Internal note"


def test_resolve_2026_raw_length_maps():
    col_map, _ = resolve_columns(_SIRVOY_2026_COLS)
    assert col_map["raw_length"] is not None
    assert "med drag" in col_map["raw_length"].lower()


def test_resolve_2026_raw_location_wish_maps():
    col_map, _ = resolve_columns(_SIRVOY_2026_COLS)
    assert col_map["raw_location_wish"] is not None
    assert "Elvebredden" in col_map["raw_location_wish"]


def test_resolve_2026_group_field_maps():
    col_map, _ = resolve_columns(_SIRVOY_2026_COLS)
    assert col_map["group_field"] is not None
    assert "gruppe" in col_map["group_field"].lower()


def test_resolve_2026_booking_no_maps_case_insensitive():
    # 2026 uses "Booking no." (lowercase n) — must resolve via case-insensitive match
    col_map, _ = resolve_columns(_SIRVOY_2026_COLS)
    assert col_map["booking_no"] is not None
