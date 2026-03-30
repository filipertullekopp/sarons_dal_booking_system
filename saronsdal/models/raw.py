"""Raw data models — typed containers for un-normalized CSV data.

These models hold data exactly as read from the source files, after encoding
repair but before any semantic parsing or normalization.  They are intentionally
permissive (mostly Optional[str]) so that messy or incomplete rows are never
silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RawSpec:
    """One row from bookings_specification.csv, attached to a booking."""

    booking_no: str
    spec_type: str          # "EXTRAS" or "ACCOMM"
    specification: str      # e.g. "Campingvogn", "Telt - stort (3-8 personer)"
    room_no: str            # e.g. "Vårdalen A01 (15,7m)" for ACCOMM rows
    guests: Optional[int]
    comment: str
    units: Optional[float]
    unit_price: Optional[float]
    status: str


@dataclass
class RawBooking:
    """
    One booking record, assembled from bookings_basic.csv merged with any
    matching rows from bookings_specification.csv.

    String fields are left un-parsed; encoding has been repaired with ftfy.
    Missing cells are represented as empty strings ("") — never as NaN.
    """

    # ------------------------------------------------------------------ identity
    booking_no: str
    booking_source: str     # "Nettside" | "Resepsjonen" | other
    booking_date: str       # raw date string, e.g. "2024-03-30 15:56:12"

    # ------------------------------------------------------------------ dates
    check_in: str           # raw date string, e.g. "2024-07-05"
    check_out: str          # raw date string, e.g. "2024-07-13"

    # ------------------------------------------------------------------ person
    first_name: str
    last_name: str
    company: str            # church / org / "privat" / ""
    num_guests: str         # raw string; may be empty
    num_rooms: str          # raw string; may be empty
    language: str           # "no", "en", etc.
    confirmed: str          # "Ja" / "Nei" / ""

    # ------------------------------------------------------------------ contact (kept for staff use)
    phone: str
    email: str

    # ------------------------------------------------------------------ free-text fields
    guest_message: str
    comment: str

    # ------------------------------------------------------------------ equipment fields (basic CSV)
    regnr: str              # registration plate / "0" / "Telt" / other
    raw_length: str         # messy length string
    raw_width: str          # messy width string (separate "Bredde" column)
    has_markise: str        # "Ja" / "Nei" / "" — awning (markise)
    has_fortelt: str        # "Ja" / "Nei" / "" — awning-tent (fortelt)

    # ------------------------------------------------------------------ preferences
    raw_location_wish: str  # free text location preference
    raw_helper: str         # "medhelper" / assistant role text

    # ------------------------------------------------------------------ guest location
    # Booking city as entered by the guest (or "" when absent/unknown).
    city: str = ""

    # ------------------------------------------------------------------ new 2026+ width field
    # "Total bredde på din enhet i meter (inkludert fortelt hvis du har det)"
    # Empty string when the column is absent (pre-2026 exports).
    raw_total_width: str = ""

    # ------------------------------------------------------------------ specification rows
    # All rows from bookings_specification.csv for this booking_no.
    # Populated by the merger; empty list when no spec rows exist.
    specs: List[RawSpec] = field(default_factory=list)

    # ------------------------------------------------------------------ schema metadata
    # Which column mapping was detected when loading this row.
    schema_version: str = ""

    def extras_specs(self) -> List[RawSpec]:
        """Return only EXTRAS-type specification rows (vehicle type indicators)."""
        return [s for s in self.specs if s.spec_type.upper() == "EXTRAS"]

    def accomm_specs(self) -> List[RawSpec]:
        """Return only ACCOMM-type specification rows (room / spot assignments)."""
        return [s for s in self.specs if s.spec_type.upper() == "ACCOMM"]

    def extra_specification_values(self) -> List[str]:
        """Convenience: list of Specification strings from EXTRAS rows."""
        return [s.specification for s in self.extras_specs() if s.specification]

    def accomm_specification_values(self) -> List[str]:
        """Convenience: list of Specification strings from ACCOMM rows."""
        return [s.specification for s in self.accomm_specs() if s.specification]
