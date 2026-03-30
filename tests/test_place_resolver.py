"""Tests for the deterministic place/city-based near-text resolver."""

from __future__ import annotations

import pytest

from saronsdal.llm.candidate_builder import NearTextCandidate
from saronsdal.llm.place_resolver import (
    PlaceRef,
    PlaceResolutionResult,
    _HIGH_CONF_THRESHOLD,
    resolve_place_refs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cand(
    booking_no: str = "99999",
    near_texts: list[str] = None,
    check_in: str = "2026-07-05",
    check_out: str = "2026-07-12",
) -> NearTextCandidate:
    return NearTextCandidate(
        booking_no=booking_no,
        full_name="Test Person",
        raw_near_texts=near_texts or [],
        check_in=check_in,
        check_out=check_out,
    )


def _make_booking(
    booking_no: str,
    city: str,
    check_in: str = "2026-07-05",
    check_out: str = "2026-07-12",
) -> dict:
    return {
        "booking_no": booking_no,
        "full_name": f"Person {booking_no}",
        "check_in": check_in,
        "check_out": check_out,
        "city": city,
        "request": {"preferred_sections": []},
        "group_signals": {"organization": None, "group_field": None},
    }


# ---------------------------------------------------------------------------
# Basic pattern matching
# ---------------------------------------------------------------------------

class TestFragmentExtraction:
    def test_simple_fra_pattern(self):
        bookings = [
            _make_booking("B001", "Varhaug"),
            _make_booking("B002", "Varhaug"),
            _make_booking("B003", "Varhaug"),
        ]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        assert len(result.refs) == 1
        ref = result.refs[0]
        assert ref.match_type == "place_group"
        assert ref.place_tokens == ["varhaug"]
        assert set(ref.matched_booking_nos) == {"B001", "B002", "B003"}
        assert ref.confidence >= 0.70

    def test_de_andre_fra_pattern(self):
        bookings = [_make_booking("B001", "Moi"), _make_booking("B002", "Moi")]
        cand = _make_cand(near_texts=["de andre fra Moi"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.match_type == "place_group"
        assert "moi" in ref.place_tokens
        assert len(ref.matched_booking_nos) == 2

    def test_plain_fra_pattern(self):
        bookings = [_make_booking("C001", "Lund")]
        cand = _make_cand(near_texts=["fra Lund"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.match_type == "place_group"
        assert "lund" in ref.place_tokens


# ---------------------------------------------------------------------------
# Multi-place (slash-separated)
# ---------------------------------------------------------------------------

class TestMultiPlace:
    def test_moi_lund_slash(self):
        bookings = [
            _make_booking("M001", "Moi"),
            _make_booking("L001", "Lund"),
            _make_booking("L002", "Lund"),
        ]
        cand = _make_cand(near_texts=["de andre fra Moi/Lund"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert "moi" in ref.place_tokens
        assert "lund" in ref.place_tokens
        assert "M001" in ref.matched_booking_nos
        assert "L001" in ref.matched_booking_nos
        assert "L002" in ref.matched_booking_nos
        # 3 matches → _CONF_FEW_MATCHES
        assert ref.confidence >= 0.70

    def test_both_places_contribute(self):
        bookings = [
            _make_booking("M001", "Moi"),
            _make_booking("L001", "Lund"),
        ]
        cand = _make_cand(near_texts=["fra Moi/Lund"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert "Moi" in ref.matched_cities or "moi" in [c.lower() for c in ref.matched_cities]
        assert "Lund" in ref.matched_cities or "lund" in [c.lower() for c in ref.matched_cities]


# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

class TestConfidenceTiers:
    def test_no_match_unresolved(self):
        bookings = [_make_booking("B001", "Stavanger")]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.match_type == "unresolved"
        assert ref.confidence == 0.0
        assert ref.matched_booking_nos == []

    def test_one_match_weak(self):
        bookings = [_make_booking("B001", "Varhaug")]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        # One match → weak candidate
        assert ref.match_type == "place_group"
        assert 0.40 <= ref.confidence <= 0.55

    def test_two_to_four_matches_moderate(self):
        bookings = [_make_booking(f"B{i}", "Varhaug") for i in range(3)]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert 0.65 <= ref.confidence <= 0.75

    def test_five_plus_matches_strong(self):
        bookings = [_make_booking(f"B{i}", "Varhaug") for i in range(6)]
        cand = _make_cand(near_texts=["fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.confidence >= 0.75


# ---------------------------------------------------------------------------
# Broad place penalty
# ---------------------------------------------------------------------------

class TestBroadPlace:
    def test_jaeren_reduced_confidence(self):
        bookings = [_make_booking(f"B{i}", "Jæren") for i in range(5)]
        cand = _make_cand(near_texts=["andre fra Jæren"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.is_broad_place is True
        # Even with 5 matches, confidence is reduced
        assert ref.confidence < 0.80

    def test_rogaland_is_broad(self):
        bookings = [_make_booking(f"B{i}", "Rogaland") for i in range(5)]
        cand = _make_cand(near_texts=["folk fra Rogaland"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.is_broad_place is True


# ---------------------------------------------------------------------------
# Church/group references
# ---------------------------------------------------------------------------

class TestChurchGroup:
    def test_betel_flagged_as_church_group(self):
        bookings = [_make_booking("B001", "Hommersåk")]
        cand = _make_cand(near_texts=["gjengen fra Betel"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.match_type == "church_group"
        # No city-roster matching performed
        assert ref.matched_booking_nos == []
        assert ref.confidence == 0.0

    def test_filadelfia_flagged_as_church_group(self):
        bookings = [_make_booking("F001", "Lyngdal")]
        cand = _make_cand(near_texts=["folka fra Filadelfia"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.match_type == "church_group"

    def test_betania_flagged_as_church_group(self):
        cand = _make_cand(near_texts=["bo nær Betania"])
        result = resolve_place_refs(cand, [])
        ref = result.refs[0]
        assert ref.match_type == "church_group"


# ---------------------------------------------------------------------------
# Section names excluded
# ---------------------------------------------------------------------------

class TestSectionExclusion:
    def test_furulunden_not_treated_as_city(self):
        bookings = [_make_booking("B001", "Furulunden")]
        cand = _make_cand(near_texts=["fra Furulunden"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        # Section name → not a near-text group reference
        assert ref.match_type == "unresolved"
        assert "section" in ref.rationale.lower() or "campsite" in ref.rationale.lower()

    def test_vaardalen_not_treated_as_city(self):
        bookings = [_make_booking("B001", "Vårdalen")]
        cand = _make_cand(near_texts=["vi vil bo i Vårdalen"])
        result = resolve_place_refs(cand, bookings)
        # No "fra X" pattern → unresolved (different reason)
        ref = result.refs[0]
        assert ref.match_type == "unresolved"


# ---------------------------------------------------------------------------
# No fra-pattern
# ---------------------------------------------------------------------------

class TestNonFraFragment:
    def test_name_reference_no_fra(self):
        """Pure name references have no fra-pattern — left unresolved by place resolver."""
        bookings = [_make_booking("B001", "Stavanger")]
        cand = _make_cand(near_texts=["Tor A Ramsli"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.match_type == "unresolved"
        assert ref.place_tokens == []


# ---------------------------------------------------------------------------
# Date-range filtering
# ---------------------------------------------------------------------------

class TestDateFiltering:
    def test_non_overlapping_booking_not_included(self):
        """Bookings outside the candidate's date range must not appear in roster."""
        bookings = [
            # Outside range — after checkout
            _make_booking("OUT1", "Varhaug", check_in="2026-07-20", check_out="2026-07-25"),
        ]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert ref.matched_booking_nos == []

    def test_overlapping_booking_included(self):
        bookings = [
            # Overlapping range
            _make_booking("OVR1", "Varhaug", check_in="2026-07-03", check_out="2026-07-09"),
        ]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ref = result.refs[0]
        assert "OVR1" in ref.matched_booking_nos


# ---------------------------------------------------------------------------
# PlaceResolutionResult properties
# ---------------------------------------------------------------------------

class TestResolutionResultProperties:
    def test_has_any_resolution_true(self):
        bookings = [_make_booking("B001", "Varhaug"), _make_booking("B002", "Varhaug")]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        assert result.has_any_resolution is True

    def test_has_any_resolution_false_when_no_match(self):
        cand = _make_cand(near_texts=["Tor A Ramsli"])
        result = resolve_place_refs(cand, [])
        assert result.has_any_resolution is False

    def test_all_high_confidence_with_several_matches(self):
        bookings = [_make_booking(f"B{i}", "Varhaug") for i in range(4)]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        # 4 matches → conf=0.70 >= threshold
        assert result.all_high_confidence is True

    def test_all_high_confidence_false_single_match(self):
        bookings = [_make_booking("B001", "Varhaug")]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        # 1 match → conf=0.45 < threshold
        assert result.all_high_confidence is False

    def test_as_prompt_context_structure(self):
        bookings = [_make_booking("B001", "Varhaug"), _make_booking("B002", "Varhaug")]
        cand = _make_cand(near_texts=["familier fra Varhaug"])
        result = resolve_place_refs(cand, bookings)
        ctx = result.as_prompt_context()
        assert len(ctx) == 1
        item = ctx[0]
        assert "fragment" in item
        assert "place_tokens" in item
        assert "matched_booking_nos" in item
        assert "confidence" in item
        assert "rationale" in item
