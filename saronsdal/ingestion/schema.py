"""Column-name schema definitions for Sirvoy CSV exports.

The booking_basic CSV has evolved over time and may carry mojibake column names
if the file was exported or re-saved with the wrong encoding.  This module
provides flexible lookup that accepts both the clean UTF-8 name and known
mojibake variants.

Usage:
    from saronsdal.ingestion.schema import resolve_columns, SchemaVersion

    col_map, version = resolve_columns(df.columns.tolist())
    booking_no = df[col_map["booking_no"]]
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class SchemaVersion(Enum):
    BASIC_EXPORT = auto()    # bookings_basic.csv  (has Bredde + Markise + Fortelt cols)
    SIRVOY_2024 = auto()     # 2024b.csv / 2025b.csv style (has group field)
    UNKNOWN = auto()


# ---------------------------------------------------------------------------
# Column-name alias tables
# Each key is a canonical internal field name.
# Each value is a list of possible CSV column names, tried in order.
# ---------------------------------------------------------------------------

# Columns shared by all known export formats.
_COMMON_COLUMNS: Dict[str, List[str]] = {
    "booking_no":       ["Booking No."],
    "booking_source":   ["Booking source"],
    "booking_date":     ["Booking Date"],
    "check_in":         ["Check-in"],
    "check_out":        ["Check-out"],
    "first_name":       ["First Name"],
    "last_name":        ["Last Name"],
    "company":          ["Company"],
    "num_guests":       ["Number of Guests"],
    "num_rooms":        ["Number of Rooms"],
    "language":         ["Language"],
    "confirmed":        ["Confirmed"],
    "phone":            ["Phone"],
    "email":            ["Email"],
    "city":             ["City"],
    "guest_message":    ["Guest Message", "Guest comment"],   # 2026 export uses "Guest comment"
    "comment":          ["Comment", "Internal note"],          # 2026 export uses "Internal note"
    "regnr": [
        "Regnr på bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
        # mojibake variants:
        "Regnr pÃ¥ bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
        "Regnr pa bobil , campingvogn eller bil (skriv 0 hvis du ikke har)",
    ],
    # New column added to Sirvoy form from 2026 onwards.
    # Customers enter their total unit width (including any fortelt) in metres.
    # The field is optional/absent in older exports — returns None when missing.
    "raw_total_width": [
        "Total bredde på din enhet i meter (inkludert fortelt hvis du har det)",
        "Total bredde pÃ¥ din enhet i meter (inkludert fortelt hvis du har det)",
    ],
}

# Columns present only in the basic export (bookings_basic.csv).
_BASIC_ONLY_COLUMNS: Dict[str, List[str]] = {
    "has_markise": [
        "Markise til bobil/campingvogn",
    ],
    "has_fortelt": [
        "Fortelt til bobil eller campingvogn",
    ],
    "raw_length": [
        "Lengde på bobil, campingvogn eller telt",
        "Lengde pÃ¥ bobil, campingvogn eller telt",
        "Lengde pa bobil, campingvogn eller telt",
    ],
    "raw_width": [
        "Bredde på telt/fortelt/markise",
        "Bredde pÃ¥ telt/fortelt/markise",
        "Bredde pa telt/fortelt/markise",
    ],
    "raw_location_wish": [
        "Ønsket plassering (Bergesli kun for Youthplanet)",
        "Ã˜nsket plassering (Bergesli kun for Youthplanet)",
        "nsket plassering (Bergesli kun for Youthplanet)",
    ],
    "raw_helper": [
        "Skal være medhjelper i sommer",
        "Skal vÃ¦re medhjelper i sommer",
        "Skal vare medhjelper i sommer",
    ],
}

# Sirovy 2024 is the spec csv file!?  
_SIRVOY_2024_ONLY_COLUMNS: Dict[str, List[str]] = {
    "raw_length": [
        # 2026+ form adds "i meter" at the end:
        "Lengde på bobil, campingvogn (totalt med drag) eller telt i meter",
        "Lengde pÃ¥ bobil, campingvogn (totalt med drag) eller telt i meter",
        # Earlier 2024/2025 style (with and without trailing space):
        "Lengde på bobil, campingvogn (totalt med drag) eller telt ",
        "Lengde pÃ¥ bobil, campingvogn (totalt med drag) eller telt ",
        "Lengde pa bobil, campingvogn (totalt med drag) eller telt ",
        "Lengde på bobil, campingvogn (totalt med drag) eller telt",
    ],
    "raw_location_wish": [
        "Ønsket plassering (Furutoppen, Furulunden, Elvebredden, Fjelltarassen, Vårdalen, Egelandsletta)",
        "Ã˜nsket plassering (Furutoppen, Furulunden, Elvebredden, Fjelltarassen, VÃ¥rdalen, Egelandsletta)",
    ],
    "group_field": [
        "For deg som kommer som gruppe, hvilke gruppe tilhører du?",
        "For deg som kommer som gruppe, hvilke gruppe tilhÃ¸rer du?",
    ],
    "raw_helper": [
        "Skriv antall som skal være medhjelpere og hvor de skal hjelpe til:",
        "Skriv antall som skal vÃ¦re medhjelpere og hvor de skal hjelpe til:",
    ],
    # No separate width or fortelt/markise columns in this format.
    "has_markise":  [],   # not present
    "has_fortelt":  [],   # not present
    "raw_width":    [],   # not present
}


def _first_match(candidates: List[str], available: List[str]) -> Optional[str]:
    """Return the first candidate column name that exists in `available`."""
    available_lower = {c.lower(): c for c in available}
    for candidate in candidates:
        canonical = available_lower.get(candidate.lower())
        if canonical is not None:
            return canonical
    return None


def detect_schema_version(columns: List[str], logger=None) -> SchemaVersion:
    """
    Identify which CSV export format produced these columns.

    Detection priority (highest first):
      1. group_field column present → SIRVOY_2024
         (takes precedence even if bredde is also present, because some Sirvoy
         exports include section names like "Elvebredden" inside a column header,
         causing a false "bredde" hit)
      2. Bredde-specific column OR fortelt column → BASIC_EXPORT
      3. Neither → UNKNOWN
    """
    import logging as _logging
    _log = logger or _logging.getLogger(__name__)

    cols_lower = {c.lower() for c in columns}

    # bredde must appear as part of the dedicated width column
    # ("Bredde på telt/..."), NOT as a substring of a section name ("elvebredden").
    has_bredde = any(
        "bredde" in c and ("telt" in c or "fortelt" in c or "markise" in c)
        for c in cols_lower
    )
    has_fortelt_col = any(
        "fortelt til bobil" in c for c in cols_lower
    )
    has_group_field = any(
        "gruppe tilh" in c or "gruppe tilhã" in c for c in cols_lower
    )

    _log.debug(
        "Schema signals — has_group_field=%s  has_bredde=%s  has_fortelt_col=%s",
        has_group_field, has_bredde, has_fortelt_col,
    )

    # Group field wins: it is unique to the 2024/2025/2026 Sirvoy export.
    if has_group_field:
        _log.debug("Detected SIRVOY_2024 (group field present)")
        return SchemaVersion.SIRVOY_2024
    if has_bredde or has_fortelt_col:
        _log.debug("Detected BASIC_EXPORT (bredde/fortelt column present)")
        return SchemaVersion.BASIC_EXPORT
    _log.debug("Detected UNKNOWN (no distinguishing columns found)")
    return SchemaVersion.UNKNOWN


def resolve_columns(
    columns: List[str],
) -> Tuple[Dict[str, Optional[str]], SchemaVersion]:
    """
    Map canonical field names to actual column names found in the DataFrame.

    Returns:
        col_map: dict of canonical_name → actual_column_name (or None if absent)
        version: detected SchemaVersion
    """
    version = detect_schema_version(columns)

    format_specific = (
        _BASIC_ONLY_COLUMNS
        if version == SchemaVersion.BASIC_EXPORT
        else _SIRVOY_2024_ONLY_COLUMNS
    )

    all_candidates: Dict[str, List[str]] = {}
    all_candidates.update(_COMMON_COLUMNS)
    all_candidates.update(format_specific)

    # group_field is not in BASIC_ONLY so ensure it gets an empty fallback
    if "group_field" not in all_candidates:
        all_candidates["group_field"] = []

    col_map: Dict[str, Optional[str]] = {}
    for field_name, candidates in all_candidates.items():
        col_map[field_name] = _first_match(candidates, columns)

    return col_map, version
