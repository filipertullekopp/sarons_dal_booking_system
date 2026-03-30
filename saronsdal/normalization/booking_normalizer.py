"""Orchestrate normalization of a single merged RawBooking into a Booking.

This module is the single entry point for Phase 1 normalization.
It calls the equipment, preferences, and group-signal sub-modules and
aggregates confidence + review flags into the final Booking object.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import List, Optional

import ftfy

from saronsdal.models.normalized import Booking, BookingSource
from saronsdal.models.raw import RawBooking
from saronsdal.normalization.equipment import classify_vehicle
from saronsdal.normalization.preferences import extract_group_signals, extract_spot_request

logger = logging.getLogger(__name__)

_DATE_FORMATS = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"]


def _parse_date(raw: str) -> Optional[date]:
    """Try several date formats; return None on failure."""
    raw = raw.strip()
    if not raw:
        return None
    # Strip time portion if present.
    raw = raw.split(" ")[0].split("T")[0]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _parse_int(raw: str, default: int = 0) -> int:
    try:
        return int(float(raw.strip()))
    except (ValueError, AttributeError):
        return default


def _normalise_source(raw: str) -> BookingSource:
    raw_l = raw.strip().lower()
    if raw_l in ("nettside", "website", "online"):
        return "website"
    if raw_l in ("resepsjonen", "reception", "front desk"):
        return "reception"
    return "other"


def _normalise_confirmed(raw: str) -> bool:
    return raw.strip().lower() in ("ja", "yes", "1", "true")


def normalise_booking(raw: RawBooking) -> Booking:
    """
    Convert one RawBooking into a fully normalized Booking.

    All sub-normalizations are called here.  Errors in sub-modules are caught
    and converted into review flags so that one bad booking does not abort the
    entire pipeline.
    """
    booking_flags: List[str] = []

    # ------------------------------------------------------------------ identity
    source = _normalise_source(raw.booking_source)

    # ------------------------------------------------------------------ dates
    check_in = _parse_date(raw.check_in)
    check_out = _parse_date(raw.check_out)

    if check_in is None:
        booking_flags.append("missing_check_in")
    if check_out is None:
        booking_flags.append("missing_check_out")
    if check_in and check_out and check_in >= check_out:
        booking_flags.append("invalid_date_range")

    # ------------------------------------------------------------------ person
    first = ftfy.fix_text(raw.first_name.strip())
    last = ftfy.fix_text(raw.last_name.strip())
    full_name = f"{first} {last}".strip()
    if not full_name:
        booking_flags.append("missing_name")

    # Guest count is not used for allocation; Sirvoy sometimes exports 0 here.
    num_guests = max(1, _parse_int(raw.num_guests, default=1))

    # ------------------------------------------------------------------ equipment
    try:
        vehicle = classify_vehicle(raw)
    except Exception as exc:
        logger.error(
            "Equipment classification failed for booking %s: %s",
            raw.booking_no,
            exc,
        )
        from saronsdal.normalization.equipment import classify_vehicle as _cv
        # Produce a minimal unknown vehicle so the booking stays in the pipeline.
        from saronsdal.models.normalized import VehicleUnit
        vehicle = VehicleUnit(
            vehicle_type="unknown",
            spec_size_hint=None,
            body_length_m=None,
            body_width_m=None,
            fortelt_width_m=0.0,
            total_width_m=None,
            required_spot_count=None,
            has_fortelt=False,
            has_markise=False,
            registration=None,
            parse_confidence=0.0,
            review_flags=["classification_error"],
        )
        booking_flags.append("classification_error")

    # ------------------------------------------------------------------ preferences
    try:
        request = extract_spot_request(raw)
    except Exception as exc:
        logger.error(
            "Preference extraction failed for booking %s: %s",
            raw.booking_no,
            exc,
        )
        from saronsdal.models.normalized import SpotRequest
        request = SpotRequest(
            preferred_sections=[],
            preferred_spot_ids=[],
            avoid_sections=[],
            amenity_flags=set(),
            raw_near_texts=[],
            parse_confidence=0.0,
            review_flags=["preference_extraction_error"],
        )
        booking_flags.append("preference_extraction_error")

    # ------------------------------------------------------------------ group signals
    try:
        group_signals = extract_group_signals(raw)
    except Exception as exc:
        logger.error(
            "Group signal extraction failed for booking %s: %s",
            raw.booking_no,
            exc,
        )
        from saronsdal.models.normalized import RawGroupSignals
        group_signals = RawGroupSignals(
            organization=None,
            group_field=None,
            near_text_fragments=[],
            is_org_private=True,
        )
        booking_flags.append("group_signal_extraction_error")

    # ------------------------------------------------------------------ reception placeholder
    if source == "reception" and vehicle.body_length_m is None:
        if "reception_placeholder_data" not in vehicle.review_flags:
            booking_flags.append("reception_placeholder_data")

    # ------------------------------------------------------------------ aggregate confidence
    # Weight: vehicle classification carries more importance than preference parse.
    data_confidence = round(
        vehicle.parse_confidence * 0.6 + request.parse_confidence * 0.4,
        2,
    )

    # ------------------------------------------------------------------ merge all flags
    all_flags: List[str] = (
        booking_flags
        + vehicle.review_flags
        + request.review_flags
    )
    # Deduplicate while preserving order.
    seen: set = set()
    deduped_flags = [f for f in all_flags if not (f in seen or seen.add(f))]

    return Booking(
        booking_no=raw.booking_no,
        booking_source=source,
        check_in=check_in,
        check_out=check_out,
        first_name=first,
        last_name=last,
        full_name=full_name,
        num_guests=num_guests,
        language=raw.language.strip().lower(),
        is_confirmed=_normalise_confirmed(raw.confirmed),
        vehicle=vehicle,
        request=request,
        group_signals=group_signals,
        data_confidence=data_confidence,
        review_flags=deduped_flags,
        raw_guest_message=ftfy.fix_text(raw.guest_message.strip()),
        raw_comment=ftfy.fix_text(raw.comment.strip()),
        raw_location_wish=ftfy.fix_text(raw.raw_location_wish.strip()),
        city=ftfy.fix_text(raw.city.strip()),
    )


def normalise_all(raw_bookings: List[RawBooking]) -> List[Booking]:
    """Normalise a list of RawBooking objects, logging progress."""
    results: List[Booking] = []
    errors = 0
    for raw in raw_bookings:
        try:
            booking = normalise_booking(raw)
            results.append(booking)
        except Exception as exc:
            logger.error(
                "Unhandled error normalising booking %s: %s",
                raw.booking_no,
                exc,
                exc_info=True,
            )
            errors += 1
    logger.info(
        "Normalised %d bookings (%d errors)",
        len(results),
        errors,
    )
    return results
