"""Load bookings_specification.csv into a dict keyed by Booking No.

Each booking may have multiple specification rows (one per line item).
We aggregate them into a list per booking number.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import ftfy
import pandas as pd

from saronsdal.models.raw import RawSpec

logger = logging.getLogger(__name__)

# Expected column names in the specification CSV (case-insensitive match).
_SPEC_COLUMN_ALIASES: Dict[str, List[str]] = {
    "spec_type":       ["Type", "type"],
    "booking_no":      ["Booking No.", "Booking No"],
    "check_in":        ["Check-in", "Check In", "Checkin"],
    "check_out":       ["Check-out", "Check Out", "Checkout"],
    "first_name":      ["First Name"],
    "last_name":       ["Last Name"],
    "specification":   ["Specification", "specification"],
    "room_no":         ["Room No.", "Room No", "Room"],
    "guests":          ["Guests", "guests"],
    "guest_name":      ["Guest Name"],
    "comment":         ["Comment", "comment"],
    "date":            ["Date", "date"],
    "units":           ["Units", "units"],
    "unit_price":      ["Unit Price", "unit price"],
    "total":           ["Total", "total"],
    "reference":       ["Reference", "reference"],
    "status":          ["Status", "status"],
}


def _resolve_spec_columns(
    columns: List[str],
) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in columns}
    result: Dict[str, Optional[str]] = {}
    for field, aliases in _SPEC_COLUMN_ALIASES.items():
        result[field] = next(
            (cols_lower[a.lower()] for a in aliases if a.lower() in cols_lower),
            None,
        )
    return result


def _safe(row: pd.Series, col: Optional[str]) -> str:
    if col is None or col not in row.index:
        return ""
    v = row[col]
    return "" if pd.isna(v) else str(v).strip()


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val.replace(",", ".")) if val else None
    except ValueError:
        return None


def _safe_int(val: str) -> Optional[int]:
    try:
        return int(float(val)) if val else None
    except ValueError:
        return None


def load_specifications(path: Path) -> Dict[str, List[RawSpec]]:
    """
    Load the Sirvoy specification CSV and return a dict mapping Booking No.
    to a list of RawSpec rows.

    Args:
        path: Path to bookings_specification.csv.

    Returns:
        Dict[booking_no, List[RawSpec]].  Missing booking numbers are skipped
        with a warning.  Orphan spec rows (booking_no not in basic CSV) are
        preserved; the merger will log them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Specification file not found: {path}")

    # Specification CSV is typically clean UTF-8, but apply ftfy defensively.
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="latin-1", keep_default_na=False)

    df.columns = [ftfy.fix_text(str(c)) for c in df.columns]
    for col in df.columns:
        df[col] = df[col].apply(lambda v: ftfy.fix_text(str(v)) if v else "")

    col_map = _resolve_spec_columns(df.columns.tolist())

    if col_map.get("booking_no") is None:
        raise ValueError(
            f"Cannot find 'Booking No.' column in {path.name}. "
            f"Columns: {df.columns.tolist()}"
        )
    if col_map.get("specification") is None:
        raise ValueError(
            f"Cannot find 'Specification' column in {path.name}."
        )

    result: Dict[str, List[RawSpec]] = {}
    skipped = 0

    for idx, row in df.iterrows():
        booking_no = _safe(row, col_map["booking_no"])
        if not booking_no:
            logger.warning("Spec row %d: missing Booking No. — skipped", idx)
            skipped += 1
            continue

        spec = RawSpec(
            booking_no=booking_no,
            spec_type=_safe(row, col_map.get("spec_type")),
            specification=_safe(row, col_map.get("specification")),
            room_no=_safe(row, col_map.get("room_no")),
            guests=_safe_int(_safe(row, col_map.get("guests"))),
            comment=_safe(row, col_map.get("comment")),
            units=_safe_float(_safe(row, col_map.get("units"))),
            unit_price=_safe_float(_safe(row, col_map.get("unit_price"))),
            status=_safe(row, col_map.get("status")),
        )
        result.setdefault(booking_no, []).append(spec)

    logger.info(
        "Loaded specifications for %d bookings from %s (%d rows total, %d skipped)",
        len(result),
        path.name,
        len(df),
        skipped,
    )
    return result
