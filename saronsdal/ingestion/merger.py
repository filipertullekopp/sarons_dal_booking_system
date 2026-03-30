"""Merge bookings_basic.csv records with bookings_specification.csv records.

Join key: Booking No.
Join type: left — every booking from the basic CSV is preserved, even if it
           has no matching specification rows.

Orphan specification rows (spec booking_no not present in basic CSV) are
logged as warnings but do not cause errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from saronsdal.ingestion.booking_reader import load_bookings_basic
from saronsdal.ingestion.specification_reader import load_specifications
from saronsdal.models.raw import RawBooking, RawSpec

logger = logging.getLogger(__name__)


def merge_bookings(
    basic_path: Path,
    spec_path: Path,
) -> List[RawBooking]:
    """
    Load both CSV files and return a unified list of RawBooking objects.

    Each RawBooking will have its `.specs` list populated with all matching
    RawSpec rows (empty list if none exist).

    Args:
        basic_path: Path to bookings_basic.csv
        spec_path:  Path to bookings_specification.csv

    Returns:
        List of RawBooking in the same order as the basic CSV.
    """
    bookings = load_bookings_basic(basic_path)
    specs_by_no: Dict[str, List[RawSpec]] = load_specifications(spec_path)

    basic_nos = {b.booking_no for b in bookings}

    # Warn on orphan spec rows.
    orphan_nos = set(specs_by_no.keys()) - basic_nos
    if orphan_nos:
        logger.warning(
            "%d booking number(s) found in specification CSV but not in basic CSV: %s",
            len(orphan_nos),
            sorted(orphan_nos),
        )

    # Attach specs to bookings.
    matched = 0
    for booking in bookings:
        spec_list = specs_by_no.get(booking.booking_no, [])
        booking.specs = spec_list
        if spec_list:
            matched += 1

    no_specs = len(bookings) - matched
    logger.info(
        "Merged: %d bookings total, %d have spec rows, %d have no spec rows",
        len(bookings),
        matched,
        no_specs,
    )
    if no_specs > 0:
        logger.debug(
            "Bookings with no spec rows (may need manual data entry): %s",
            [b.booking_no for b in bookings if not b.specs],
        )

    return bookings
