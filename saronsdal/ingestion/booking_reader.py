"""Load bookings_basic.csv into a list of RawBooking objects.

Handles:
- Encoding detection (try UTF-8, fall back to latin-1) + ftfy repair
- Schema detection and flexible column mapping
- Missing cells → empty string (never NaN)
- Rows with missing Booking No. are skipped with a logged warning
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import ftfy
import pandas as pd

from saronsdal.ingestion.schema import SchemaVersion, resolve_columns
from saronsdal.models.raw import RawBooking

logger = logging.getLogger(__name__)


def _strip_row_quoting(content: str) -> str:
    """Handle Sirvoy CSVs where each entire row is wrapped in outer double-quotes.

    Some Sirvoy exports wrap every row in `"..."` and escape internal commas as
    `""...""`.  Standard CSV parsers see the whole row as a single field.
    This function strips the outer quotes and unescapes internal double-quotes
    so that normal CSV parsing works correctly.
    """
    rows = []
    for line in content.splitlines():
        line = line.rstrip("\r")
        if line.startswith('"') and line.endswith('"') and len(line) > 1:
            line = line[1:-1].replace('""', '"')
        rows.append(line)
    return "\n".join(rows)


def _read_csv_with_encoding(path: Path) -> pd.DataFrame:
    """Try UTF-8 (with and without BOM), then fall back to latin-1.

    For each encoding attempt the C engine is tried first.  If it raises a
    ParserError (e.g. complex quoting, embedded newlines in quoted fields),
    the Python engine is used as a fallback with explicit RFC 4180 settings.
    As a final fallback, the file is pre-processed to handle the Sirvoy format
    where each entire row is wrapped in outer double-quotes.
    """
    import io as _io

    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False)
            except pd.errors.ParserError:
                logger.warning(
                    "C parser failed for %s (encoding=%s) — retrying with Python engine",
                    path.name,
                    enc,
                )
                try:
                    df = pd.read_csv(
                        path,
                        dtype=str,
                        encoding=enc,
                        keep_default_na=False,
                        engine="python",
                        sep=",",
                        quotechar='"',
                        on_bad_lines="warn",
                    )
                except Exception:
                    df = pd.DataFrame()

            # If the result has only one column the entire row was treated as a
            # single quoted field — typical of the Sirvoy whole-row-quoted format.
            if len(df.columns) == 1:
                logger.warning(
                    "Single-column parse detected for %s — attempting "
                    "whole-row-unquoting fallback",
                    path.name,
                )
                with open(path, encoding=enc) as fh:
                    raw_content = fh.read()
                cleaned = _strip_row_quoting(raw_content)
                df = pd.read_csv(
                    _io.StringIO(cleaned),
                    dtype=str,
                    keep_default_na=False,
                )

            logger.debug("Loaded %s with encoding=%s", path.name, enc)
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} with any supported encoding")


def _repair_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ftfy encoding repair to all string columns including headers."""
    # Repair column names first so lookups work correctly.
    df.columns = [ftfy.fix_text(str(c)) for c in df.columns]

    # Repair all cell values.
    for col in df.columns:
        df[col] = df[col].apply(lambda v: ftfy.fix_text(str(v)) if v != "" else "")

    return df


def _safe_get(row: pd.Series, col_name: str | None) -> str:
    """Return cell value as string, or "" if column is absent or value is null."""
    if col_name is None or col_name not in row.index:
        return ""
    val = row[col_name]
    return "" if pd.isna(val) or str(val).lower() in ("nan", "none") else str(val).strip()


def load_bookings_basic(path: Path) -> List[RawBooking]:
    """
    Load a Sirvoy basic booking CSV and return a list of RawBooking objects.

    The returned list contains ALL rows that have a non-empty Booking No.,
    including Resepsjonen bookings with missing dimension data.  Those rows
    receive appropriate review flags during the normalization phase.

    Args:
        path: Path to bookings_basic.csv (or bookings_basic.csv.csv).

    Returns:
        List of RawBooking; order matches the CSV row order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Booking file not found: {path}")

    df = _read_csv_with_encoding(path)
    df = _repair_df(df)

    col_map, version = resolve_columns(df.columns.tolist())
    logger.info(
        "Detected schema version=%s for %s (%d rows)",
        version.name,
        path.name,
        len(df),
    )

    if col_map.get("booking_no") is None:
        raise ValueError(
            f"Cannot find 'Booking No.' column in {path.name}. "
            f"Available columns: {df.columns.tolist()}"
        )

    bookings: List[RawBooking] = []
    skipped = 0

    for idx, row in df.iterrows():
        booking_no = _safe_get(row, col_map.get("booking_no"))
        if not booking_no:
            logger.warning("Row %d: missing Booking No. — skipped", idx)
            skipped += 1
            continue

        booking = RawBooking(
            booking_no=booking_no,
            booking_source=_safe_get(row, col_map.get("booking_source")),
            booking_date=_safe_get(row, col_map.get("booking_date")),
            check_in=_safe_get(row, col_map.get("check_in")),
            check_out=_safe_get(row, col_map.get("check_out")),
            first_name=_safe_get(row, col_map.get("first_name")),
            last_name=_safe_get(row, col_map.get("last_name")),
            company=_safe_get(row, col_map.get("company")),
            num_guests=_safe_get(row, col_map.get("num_guests")),
            num_rooms=_safe_get(row, col_map.get("num_rooms")),
            language=_safe_get(row, col_map.get("language")),
            confirmed=_safe_get(row, col_map.get("confirmed")),
            phone=_safe_get(row, col_map.get("phone")),
            email=_safe_get(row, col_map.get("email")),
            city=_safe_get(row, col_map.get("city")),
            guest_message=_safe_get(row, col_map.get("guest_message")),
            comment=_safe_get(row, col_map.get("comment")),
            regnr=_safe_get(row, col_map.get("regnr")),
            raw_length=_safe_get(row, col_map.get("raw_length")),
            raw_width=_safe_get(row, col_map.get("raw_width")),
            has_markise=_safe_get(row, col_map.get("has_markise")),
            has_fortelt=_safe_get(row, col_map.get("has_fortelt")),
            raw_location_wish=_safe_get(row, col_map.get("raw_location_wish")),
            raw_helper=_safe_get(row, col_map.get("raw_helper")),
            raw_total_width=_safe_get(row, col_map.get("raw_total_width")),
            schema_version=version.name,
        )
        bookings.append(booking)
    
    logger.info(
        "Loaded %d bookings from %s (%d skipped)",
        len(bookings),
        path.name,
        skipped,
    )
    return bookings
