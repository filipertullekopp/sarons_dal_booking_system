"""Tests for Phase 2.5 candidate selection logic."""
from typing import List, Optional

import pytest

from saronsdal.llm.candidate_builder import (
    CandidateConfig,
    _pretriage_phrase,
    _find_missing_enrichment_signals,
    _find_subsection_patterns,
    _select_group_phrases,
    _select_near_text_no_edges,
    _select_weak_clusters,
    _select_preference_candidates,
    _select_subsection_candidates,
    build_candidates,
)


# ---------------------------------------------------------------------------
# Booking / cluster / edge factories
# ---------------------------------------------------------------------------

def _booking(
    booking_no: str = "B001",
    full_name: str = "Test Person",
    raw_guest_message: str = "",
    raw_comment: str = "",
    raw_location_wish: str = "",
    organization: str = "",
    group_field: str = "",
    near_text_fragments: Optional[List[str]] = None,
    raw_near_texts: Optional[List[str]] = None,
    preferred_sections: Optional[List[str]] = None,
    preferred_section_rows: Optional[List[dict]] = None,
    preferred_spot_ids: Optional[List[str]] = None,
    amenity_flags: Optional[List[str]] = None,
    check_in: str = "2026-07-01",
    check_out: str = "2026-07-07",
) -> dict:
    return {
        "booking_no": booking_no,
        "full_name": full_name,
        "check_in": check_in,
        "check_out": check_out,
        "raw_guest_message": raw_guest_message,
        "raw_comment": raw_comment,
        "raw_location_wish": raw_location_wish,
        "group_signals": {
            "organization": organization or None,
            "group_field": group_field or None,
            "near_text_fragments": near_text_fragments or [],
            "is_org_private": not bool(organization),
        },
        "request": {
            "preferred_sections": preferred_sections or [],
            "preferred_section_rows": preferred_section_rows or [],
            "preferred_spot_ids": preferred_spot_ids or [],
            "amenity_flags": amenity_flags or [],
            "raw_near_texts": raw_near_texts or [],
            "parse_confidence": 0.5,
            "review_flags": [],
        },
        "review_flags": [],
    }


def _bb_edge(node_a: str, node_b: str) -> dict:
    return {
        "node_a": node_a,
        "node_b": node_b,
        "node_a_type": "booking",
        "node_b_type": "booking",
        "edge_type": "explicit_named_people",
        "weight": 0.85,
        "source_field": "near_text",
        "raw_text": "test",
        "normalized_text": "test",
        "confidence": 0.85,
    }


def _cluster(cluster_id: str, label: str, members: List[str], cluster_type: str = "org_group") -> dict:
    return {
        "cluster_id": cluster_id,
        "canonical_label": label,
        "members": members,
        "cluster_type": cluster_type,
        "internal_edges": [],
        "review_flags": [],
    }


# ---------------------------------------------------------------------------
# Pre-triage tests
# ---------------------------------------------------------------------------

def test_pretriage_exact_section_is_obvious_place():
    sections = {"furulunden", "elvebredden", "vårdalen"}
    result = _pretriage_phrase("furulunden", sections, set(), 0.82)
    assert result == "obvious_place"


def test_pretriage_fululunden_is_misspelling():
    """'fululunden' is a misspelling of 'furulunden' — similarity ≈ 0.90."""
    sections = {"furulunden", "elvebredden", "vårdalen"}
    result = _pretriage_phrase("fululunden", sections, set(), 0.82)
    assert result == "obvious_place_misspelling"


def test_pretriage_elvebreden_is_misspelling():
    """'elvebreden' is missing one 'd' from 'elvebredden'."""
    sections = {"furulunden", "elvebredden", "vårdalen"}
    result = _pretriage_phrase("elvebreden", sections, set(), 0.82)
    assert result == "obvious_place_misspelling"


def test_pretriage_known_group_alias():
    groups = {"betel hommersåk", "betania sokndal"}
    result = _pretriage_phrase("betel hommersåk", set(), groups, 0.82)
    assert result == "obvious_known_group"


def test_pretriage_camp_jaeren_is_truly_ambiguous():
    sections = {"furulunden", "elvebredden", "vårdalen"}
    groups = {"betel hommersåk"}
    result = _pretriage_phrase("camp jæren", sections, groups, 0.82)
    assert result == "truly_ambiguous"


def test_pretriage_varhaug_gjengen_is_truly_ambiguous():
    sections = {"furulunden", "elvebredden"}
    result = _pretriage_phrase("varhaug gjengen", sections, set(), 0.82)
    assert result == "truly_ambiguous"


# ---------------------------------------------------------------------------
# Selector 1 — group phrase extraction and selection
# ---------------------------------------------------------------------------

def test_camp_jaeren_selected_when_recurring(tmp_path):
    bookings = [
        _booking("B001", raw_guest_message="Vi er Camp Jæren"),
        _booking("B002", raw_guest_message="Camp Jæren gjengen"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, summary = _select_group_phrases(bookings, set(), set(), config)
    phrases = [c.phrase for c in candidates]
    assert any("camp jæren" in p or "camp jæren gjengen" in p for p in phrases)


def test_camp_jaeren_selected_as_singleton_with_boost():
    """freq=1 but 'camp' is a boost token → selected."""
    bookings = [_booking("B001", raw_guest_message="Vi er Camp Jæren")]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, _ = _select_group_phrases(bookings, set(), set(), config)
    phrases = [c.phrase for c in candidates]
    assert any("camp jæren" in p for p in phrases)
    # Verify it's flagged as singleton_high_signal
    camp_candidate = next((c for c in candidates if "camp jæren" in c.phrase), None)
    assert camp_candidate is not None
    assert camp_candidate.is_singleton_high_signal is True


def test_singleton_without_boost_not_selected():
    """freq=1 with no boost token → NOT selected when min_frequency=2."""
    bookings = [_booking("B001", raw_guest_message="Ønsker å stå ved Haugen Neset")]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, _ = _select_group_phrases(bookings, set(), set(), config)
    assert candidates == []


def test_furulunden_filtered_as_obvious_place():
    """Section name in free text → obvious_place → not in candidates."""
    sections = {"furulunden"}
    bookings = [
        _booking("B001", raw_guest_message="Furulunden B"),
        _booking("B002", raw_guest_message="Furulunden B"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, summary = _select_group_phrases(bookings, sections, set(), config)
    phrases = [c.phrase for c in candidates]
    assert "furulunden" not in phrases
    assert summary.obvious_place > 0


def test_fululunden_filtered_as_misspelling():
    """Misspelling of section name → obvious_place_misspelling → not in candidates."""
    sections = {"furulunden"}
    bookings = [
        _booking("B001", group_field="Fululunden"),
        _booking("B002", group_field="Fululunden"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, summary = _select_group_phrases(bookings, sections, set(), config)
    phrases = [c.phrase for c in candidates]
    assert "fululunden" not in phrases
    assert summary.obvious_place_misspelling > 0


def test_known_group_filtered():
    """Already-known group alias → obvious_known_group → not in candidates."""
    groups = {"betel hommersåk"}
    bookings = [
        _booking("B001", organization="Betel Hommersåk"),
        _booking("B002", organization="Betel Hommersåk"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, summary = _select_group_phrases(bookings, set(), groups, config)
    phrases = [c.phrase for c in candidates]
    assert "betel hommersåk" not in phrases
    assert summary.obvious_known_group > 0


def test_group_phrase_extracted_from_organization_field():
    """Group phrases should be scanned from the organization field."""
    bookings = [
        _booking("B001", organization="Betania Sokndal"),
        _booking("B002", organization="Betania Sokndal"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, _ = _select_group_phrases(bookings, set(), set(), config)
    phrases = [c.phrase for c in candidates]
    assert any("betania sokndal" in p for p in phrases)


def test_varhaug_gjengen_selected_when_recurring():
    """'Varhaug gjengen' pattern: proper noun + gjengen suffix."""
    bookings = [
        _booking("B001", raw_guest_message="Vi er fra Varhaug gjengen"),
        _booking("B002", raw_guest_message="Varhaug gjengen"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, _ = _select_group_phrases(bookings, set(), set(), config)
    phrases = [c.phrase for c in candidates]
    assert any("varhaug" in p for p in phrases)


def test_dimension_flags_never_trigger_group_selection():
    """Bookings with only dimension-related content should never be selected."""
    bookings = [
        _booking("B001", raw_guest_message="7,5 meter med fortelt"),
        _booking("B002", raw_guest_message="Campingvogn 8m"),
    ]
    config = CandidateConfig(group_phrase_min_frequency=2)
    candidates, _ = _select_group_phrases(bookings, set(), set(), config)
    # "7,5 meter med fortelt" should not produce group phrase candidates
    assert all(c.frequency >= 1 for c in candidates)
    phrases = [c.phrase for c in candidates]
    # No dimension-related phrases should appear
    assert not any(p in phrases for p in ["meter", "fortelt"])


# ---------------------------------------------------------------------------
# Selector 2 — unresolved near text with no edges
# ---------------------------------------------------------------------------

def test_near_text_with_no_edge_selected():
    bookings = [_booking("B001", raw_near_texts=["Tor A Ramsli"])]
    selected = _select_near_text_no_edges(bookings, set())
    assert len(selected) == 1
    assert selected[0].booking_no == "B001"
    assert "Tor A Ramsli" in selected[0].raw_near_texts


def test_near_text_with_bb_edge_not_selected():
    """Booking with a booking↔booking edge should not be selected."""
    bookings = [_booking("B001", raw_near_texts=["Kari Normann"])]
    bb_bnos = {"B001"}
    selected = _select_near_text_no_edges(bookings, bb_bnos)
    assert selected == []


def test_booking_with_no_near_text_not_selected():
    bookings = [_booking("B001")]
    selected = _select_near_text_no_edges(bookings, set())
    assert selected == []


def test_near_text_from_group_signals_fragment():
    """near_text_fragments from group_signals should also trigger selection."""
    bookings = [_booking("B001", near_text_fragments=["familien Husvik"])]
    selected = _select_near_text_no_edges(bookings, set())
    assert len(selected) == 1


# ---------------------------------------------------------------------------
# Selector 3 — weak label clusters
# ---------------------------------------------------------------------------

def test_org_group_cluster_selected():
    clusters = [_cluster("cl_001", "Betania Sokndal", ["B001", "B002"], "org_group")]
    results = _select_weak_clusters(clusters, set(), set(), CandidateConfig())
    assert len(results) == 1
    assert results[0].canonical_label == "Betania Sokndal"


def test_family_unit_cluster_not_selected():
    """family_unit clusters have direct booking↔booking edges — skip."""
    clusters = [_cluster("cl_001", "Husvik", ["B001", "B002"], "family_unit")]
    results = _select_weak_clusters(clusters, set(), set(), CandidateConfig())
    assert results == []


def test_section_label_cluster_classified_as_obvious_place():
    """A label that matches a section alias → obvious_place, still included for review."""
    sections = {"furulunden"}
    clusters = [_cluster("cl_001", "Furulunden", ["B001", "B002"], "org_group")]
    results = _select_weak_clusters(clusters, sections, set(), CandidateConfig())
    assert len(results) == 1
    assert results[0].pretriage_bucket == "obvious_place"


def test_misspelling_cluster_classified_as_misspelling():
    sections = {"furulunden"}
    clusters = [_cluster("cl_001", "Fululunden", ["B001", "B002"], "org_group")]
    results = _select_weak_clusters(clusters, sections, set(), CandidateConfig())
    assert len(results) == 1
    assert results[0].pretriage_bucket == "obvious_place_misspelling"


def test_known_group_cluster_not_selected():
    """Already-known group → discard (no Gemini needed)."""
    groups = {"betel hommersåk"}
    clusters = [_cluster("cl_001", "Betel Hommersåk", ["B001", "B002"], "org_group")]
    results = _select_weak_clusters(clusters, set(), groups, CandidateConfig())
    assert results == []


# ---------------------------------------------------------------------------
# Selector 4 — unstructured preference text
# ---------------------------------------------------------------------------

def test_away_from_river_selected():
    """'lengst vekke fra elva' → avoid_river signal → selected."""
    bookings = [_booking(
        "B001",
        raw_guest_message="Elvebredden, men lengst vekke fra elva",
        preferred_sections=["Elvebredden"],
    )]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert len(results) == 1
    assert "avoid_river" in results[0].missing_signals


def test_same_as_last_year_selected():
    bookings = [_booking(
        "B001",
        raw_guest_message="Gjerne samme plass som i fjor, foran bibelskolen",
    )]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert len(results) == 1
    assert "same_as_last_year" in results[0].missing_signals


def test_extra_space_lastebil_selected():
    bookings = [_booking(
        "B001",
        raw_guest_message="Skulle gjerne hatt plass til 2 biler da jeg må ha med lastebilen",
    )]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert len(results) == 1
    assert "extra_space" in results[0].missing_signals


def test_short_text_not_selected():
    """Text shorter than min_length (20 chars) → not selected."""
    bookings = [_booking("B001", raw_guest_message="vekke fra elva")]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert results == []


def test_empty_text_not_selected():
    bookings = [_booking("B001")]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert results == []


def test_preference_selected_even_when_section_extracted():
    """Section may be extracted, but avoidance nuance still missing → selected."""
    bookings = [_booking(
        "B001",
        raw_guest_message="Elvebredden, men lengst vekke fra elva",
        preferred_sections=["Elvebredden"],  # section IS already extracted
    )]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert len(results) == 1
    assert "Elvebredden" in results[0].extracted_sections


def test_near_bibelskolen_positional_selected():
    bookings = [_booking(
        "B001",
        raw_guest_message="Gjerne foran bibelskolen, samme sted som alltid",
    )]
    results = _select_preference_candidates(bookings, CandidateConfig())
    assert len(results) == 1
    assert "near_bibelskolen" in results[0].missing_signals


# ---------------------------------------------------------------------------
# Selector 5 — subsection detection
# ---------------------------------------------------------------------------

def test_row_alternatives_detected():
    """'Vårdalen D eller E' → section extracted but only D captured → subsection candidate."""
    bookings = [_booking(
        "B001",
        raw_location_wish="Vårdalen D eller E. Evt F",
        preferred_sections=["Vårdalen"],
        preferred_section_rows=[{"section": "Vårdalen", "row": "D"}],
    )]
    results = _select_subsection_candidates(bookings)
    assert len(results) == 1
    assert any("E" in p for p in results[0].unresolved_patterns)


def test_felt_pattern_detected():
    """'felt A' → subsection candidate."""
    bookings = [_booking(
        "B001",
        raw_location_wish="Elvebredden felt A",
        preferred_sections=["Elvebredden"],
        preferred_section_rows=[],
    )]
    results = _select_subsection_candidates(bookings)
    assert len(results) == 1
    assert any("felt_row:A" in p for p in results[0].unresolved_patterns)


def test_no_section_not_subsection_candidate():
    """No section extracted → nothing to add rows to."""
    bookings = [_booking("B001", raw_location_wish="D eller E")]
    results = _select_subsection_candidates(bookings)
    assert results == []


def test_already_captured_row_not_flagged():
    """If both D and E are already captured, no unresolved patterns."""
    bookings = [_booking(
        "B001",
        raw_location_wish="Furulunden D eller E",
        preferred_sections=["Furulunden"],
        preferred_section_rows=[
            {"section": "Furulunden", "row": "D"},
            {"section": "Furulunden", "row": "E"},
        ],
    )]
    results = _select_subsection_candidates(bookings)
    assert results == []


# ---------------------------------------------------------------------------
# Integration: build_candidates
# ---------------------------------------------------------------------------

def test_build_candidates_returns_candidateset(tmp_path):
    """Smoke test: build_candidates doesn't crash and returns a CandidateSet."""
    bookings = [
        _booking("B001", raw_guest_message="Camp Jæren gjengen"),
        _booking("B002", raw_guest_message="Vi reiser med Camp Jæren"),
        _booking("B003", raw_near_texts=["Tor A Ramsli"]),
        _booking("B004",
                 raw_guest_message="Vil bo på Elvebredden, lengst vekke fra elva",
                 preferred_sections=["Elvebredden"]),
    ]
    clusters = [_cluster("cl_001", "Varhaug gjengen", ["B001", "B002"], "org_group")]
    cs = build_candidates(bookings, clusters, [], [], CandidateConfig(group_phrase_min_frequency=2))
    # Camp Jæren should be in group phrases (freq=2 or singleton boost)
    phrases = [c.phrase for c in cs.group_phrases]
    assert any("camp jæren" in p for p in phrases)
    # B003 should be in near_text (no edges, has near_text)
    near_bnos = [c.booking_no for c in cs.near_text]
    assert "B003" in near_bnos
    # B004 should be in preferences (avoid_river signal)
    pref_bnos = [c.booking_no for c in cs.preferences]
    assert "B004" in pref_bnos
