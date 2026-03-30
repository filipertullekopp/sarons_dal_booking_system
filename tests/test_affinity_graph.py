"""Tests for affinity_graph.build_graph."""
from datetime import date
from typing import Dict, List, Optional

import pytest

from saronsdal.grouping.affinity_graph import build_graph
from saronsdal.models.grouping import (
    ExtractedReference,
    NormalizedGroupSignals,
)
from saronsdal.models.normalized import (
    Booking,
    RawGroupSignals,
    SpotRequest,
    VehicleUnit,
)


# ---------------------------------------------------------------------------
# Booking factory
# ---------------------------------------------------------------------------

def _vehicle() -> VehicleUnit:
    return VehicleUnit(
        vehicle_type="caravan",
        spec_size_hint=None,
        body_length_m=6.0,
        body_width_m=2.75,
        fortelt_width_m=0.0,
        total_width_m=2.75,
        required_spot_count=2,
        has_fortelt=False,
        has_markise=False,
        registration=None,
        parse_confidence=0.9,
        review_flags=[],
    )


def _request() -> SpotRequest:
    return SpotRequest(
        preferred_sections=[],
        preferred_spot_ids=[],
        avoid_sections=[],
        amenity_flags=set(),
        raw_near_texts=[],
        parse_confidence=1.0,
        review_flags=[],
    )


def _booking(
    booking_no: str,
    first_name: str,
    last_name: str,
    org: Optional[str] = None,
) -> Booking:
    return Booking(
        booking_no=booking_no,
        booking_source="website",
        check_in=date(2026, 7, 1),
        check_out=date(2026, 7, 7),
        first_name=first_name,
        last_name=last_name,
        full_name=f"{first_name} {last_name}",
        num_guests=2,
        language="no",
        is_confirmed=True,
        vehicle=_vehicle(),
        request=_request(),
        group_signals=RawGroupSignals(
            organization=org,
            group_field=None,
            near_text_fragments=[],
            is_org_private=False,
        ),
        data_confidence=0.9,
        review_flags=[],
    )


def _sig(
    booking_no: str,
    canonical_org: Optional[str] = None,
    canonical_group_field: Optional[str] = None,
    refs: Optional[List[ExtractedReference]] = None,
) -> NormalizedGroupSignals:
    return NormalizedGroupSignals(
        booking_no=booking_no,
        canonical_org=canonical_org,
        canonical_group_field=canonical_group_field,
        group_field_is_section=False,
        extracted_references=refs or [],
    )


def _family_ref(surname: str) -> ExtractedReference:
    return ExtractedReference(
        raw_text=f"fam. {surname.capitalize()}",
        normalized_candidate=surname.lower(),
        ref_type="family",
        confidence=0.85,
        source_field="near_text",
    )


def _fullname_ref(first: str, last: str) -> ExtractedReference:
    candidate = f"{first} {last}".lower()
    return ExtractedReference(
        raw_text=f"{first} {last}",
        normalized_candidate=candidate,
        ref_type="full_name",
        confidence=0.82,
        source_field="near_text",
    )


# ---------------------------------------------------------------------------
# Org membership edges
# ---------------------------------------------------------------------------

def test_org_membership_edge():
    """booking with canonical_org → booking↔label edge."""
    bookings = [_booking("B001", "Ola", "Normann")]
    signals = {"B001": _sig("B001", canonical_org="Betel Hommersåk")}
    edges, ambig, alias = build_graph(signals, bookings)

    org_edges = [e for e in edges if e.edge_type == "organization_membership"]
    assert len(org_edges) == 1
    assert org_edges[0].node_a == "B001"
    assert org_edges[0].node_b == "Betel Hommersåk"
    assert org_edges[0].node_a_type == "booking"
    assert org_edges[0].node_b_type == "label"
    assert org_edges[0].weight > 0


def test_group_field_membership_edge():
    """booking with canonical_group_field → booking↔label edge."""
    bookings = [_booking("B001", "Ola", "Normann")]
    signals = {"B001": _sig("B001", canonical_group_field="Betel Hommersåk")}
    edges, _, _ = build_graph(signals, bookings)

    gf_edges = [e for e in edges if e.edge_type == "group_field_membership"]
    assert len(gf_edges) == 1
    assert gf_edges[0].node_b_type == "label"


def test_multiple_bookings_same_org():
    """Three bookings in same org → three booking↔label edges, same label node."""
    bookings = [
        _booking("B001", "Ola", "Normann"),
        _booking("B002", "Kari", "Hansen"),
        _booking("B003", "Per", "Olsen"),
    ]
    signals = {
        "B001": _sig("B001", canonical_org="Betel Hommersåk"),
        "B002": _sig("B002", canonical_org="Betel Hommersåk"),
        "B003": _sig("B003", canonical_org="Betel Hommersåk"),
    }
    edges, _, _ = build_graph(signals, bookings)
    org_edges = [e for e in edges if e.edge_type == "organization_membership"]
    assert len(org_edges) == 3
    label_values = {e.node_b for e in org_edges}
    assert label_values == {"Betel Hommersåk"}


# ---------------------------------------------------------------------------
# Booking↔booking edges via name resolution
# ---------------------------------------------------------------------------

def test_family_ref_resolves_to_single_booking():
    """booking B001 references 'fam. Normann'; B002 has last_name Normann → explicit_family edge."""
    bookings = [
        _booking("B001", "Ola", "Hansen"),
        _booking("B002", "Kari", "Normann"),
    ]
    signals = {
        "B001": _sig("B001", refs=[_family_ref("Normann")]),
        "B002": _sig("B002"),
    }
    edges, ambig, _ = build_graph(signals, bookings)
    bb_edges = [e for e in edges if e.edge_type == "explicit_family"]
    assert len(bb_edges) == 1
    assert frozenset([bb_edges[0].node_a, bb_edges[0].node_b]) == {"B001", "B002"}
    assert ambig == []


def test_fullname_ref_resolves_to_single_booking():
    """Full-name reference resolving to exactly one booking → explicit_named_people edge."""
    bookings = [
        _booking("B001", "Ola", "Hansen"),
        _booking("B002", "Kari", "Normann"),
    ]
    signals = {
        "B001": _sig("B001", refs=[_fullname_ref("Kari", "Normann")]),
        "B002": _sig("B002"),
    }
    edges, ambig, _ = build_graph(signals, bookings)
    bb_edges = [e for e in edges if e.edge_type == "explicit_named_people"]
    assert len(bb_edges) == 1
    assert ambig == []


def test_family_ref_multiple_matches_produces_ambiguous_case():
    """Two bookings share the same surname → AmbiguousGroupCase, no edge."""
    bookings = [
        _booking("B001", "Ola", "Hansen"),
        _booking("B002", "Kari", "Normann"),
        _booking("B003", "Per", "Normann"),  # duplicate surname
    ]
    signals = {
        "B001": _sig("B001", refs=[_family_ref("Normann")]),
        "B002": _sig("B002"),
        "B003": _sig("B003"),
    }
    edges, ambig, _ = build_graph(signals, bookings)
    bb_edges = [e for e in edges if e.edge_type in ("explicit_family", "explicit_named_people")]
    assert bb_edges == []
    assert len(ambig) == 1
    assert ambig[0].booking_no == "B001"
    assert "B002" in ambig[0].matched_booking_nos
    assert "B003" in ambig[0].matched_booking_nos


def test_self_reference_not_resolved():
    """booking with own surname in ref → no edge to itself."""
    bookings = [_booking("B001", "Ola", "Hansen")]
    signals = {"B001": _sig("B001", refs=[_family_ref("Hansen")])}
    edges, ambig, _ = build_graph(signals, bookings)
    bb_edges = [e for e in edges if e.edge_type == "explicit_family"]
    assert bb_edges == []
    assert ambig == []  # only one match (excluded self) → no match at all


def test_duplicate_bb_pair_deduplicated():
    """If B001→B002 and B002→B001 both have refs, only one edge is created."""
    bookings = [
        _booking("B001", "Ola", "Hansen"),
        _booking("B002", "Kari", "Normann"),
    ]
    signals = {
        "B001": _sig("B001", refs=[_family_ref("Normann")]),
        "B002": _sig("B002", refs=[_family_ref("Hansen")]),
    }
    edges, _, _ = build_graph(signals, bookings)
    bb_edges = [e for e in edges if e.node_a_type == "booking" and e.node_b_type == "booking"]
    assert len(bb_edges) == 1  # deduplicated


# ---------------------------------------------------------------------------
# Empty / no-signal cases
# ---------------------------------------------------------------------------

def test_empty_signals_produces_no_edges():
    bookings = [_booking("B001", "Ola", "Hansen")]
    signals = {"B001": _sig("B001")}
    edges, ambig, alias = build_graph(signals, bookings)
    assert edges == []
    assert ambig == []


def test_no_bookings_no_crash():
    edges, ambig, alias = build_graph({}, [])
    assert edges == []
    assert ambig == []
