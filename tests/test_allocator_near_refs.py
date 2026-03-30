"""Tests for near-text reference integration in the allocator.

Guards the contract between Phase 2.5 reference_resolutions.jsonl and the
group-proximity machinery in allocator._build_near_ref_map.
"""
import pytest
from types import SimpleNamespace

from saronsdal.allocation.allocator import (
    NEAR_REF_CONFIDENCE_THRESHOLD,
    SYMMETRIC_REF_TYPES,
    _build_near_ref_map,
    _constraint_strength,
)
from saronsdal.llm.schemas import NearTextSuggestion, ResolvedRef


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _ref(
    matched_booking_no,
    confidence: float = 0.9,
    match_type: str = "full_name",
    raw_fragment: str = "test fragment",
) -> ResolvedRef:
    return ResolvedRef(
        raw_fragment=raw_fragment,
        matched_booking_no=matched_booking_no,
        match_type=match_type,
        confidence=confidence,
    )


def _suggestion(booking_no: str, refs, place_refs=None) -> NearTextSuggestion:
    return NearTextSuggestion(
        booking_no=booking_no,
        full_name="Test Guest",
        resolved_refs=refs,
        unresolved_fragments=[],
        notes="",
        place_refs=place_refs or [],
    )


def _place_ref(
    matched_booking_nos,
    confidence: float = 0.80,
    match_type: str = "place_group",
    fragment: str = "familier fra Varhaug",
) -> dict:
    """Build a place_ref dict as serialised by PlaceResolutionResult.as_prompt_context()."""
    return {
        "fragment": fragment,
        "place_tokens": ["varhaug"],
        "match_type": match_type,
        "matched_booking_nos": matched_booking_nos,
        "matched_cities": ["Varhaug"],
        "confidence": confidence,
        "is_broad_place": False,
        "rationale": "test",
    }


# ---------------------------------------------------------------------------
# Basic map building
# ---------------------------------------------------------------------------

class TestBuildNearRefMap:
    def test_empty_input(self):
        assert _build_near_ref_map([]) == {}

    def test_none_input(self):
        assert _build_near_ref_map(None) == {}

    def test_single_high_confidence_ref(self):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9)])]
        m = _build_near_ref_map(suggestions)
        assert m == {"B001": ["B002"]}

    def test_multiple_targets_for_one_booking(self):
        suggestions = [
            _suggestion("B001", [
                _ref("B002", confidence=0.9),
                _ref("B003", confidence=0.85),
            ])
        ]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m
        assert sorted(m["B001"]) == ["B002", "B003"]

    def test_multiple_bookings(self):
        suggestions = [
            _suggestion("B001", [_ref("B002")]),
            _suggestion("B003", [_ref("B004")]),
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"] == ["B002"]
        assert m["B003"] == ["B004"]


# ---------------------------------------------------------------------------
# Confidence filtering
# ---------------------------------------------------------------------------

class TestConfidenceFiltering:
    def test_at_threshold_is_included(self):
        # Exactly at threshold → included
        suggestions = [_suggestion("B001", [_ref("B002", confidence=NEAR_REF_CONFIDENCE_THRESHOLD)])]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m

    def test_below_threshold_is_excluded(self):
        # Just below threshold → excluded
        below = NEAR_REF_CONFIDENCE_THRESHOLD - 0.01
        suggestions = [_suggestion("B001", [_ref("B002", confidence=below)])]
        m = _build_near_ref_map(suggestions)
        assert "B001" not in m

    def test_well_below_threshold_excluded(self):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.3)])]
        assert _build_near_ref_map(suggestions) == {}

    def test_mixed_confidence_only_high_included(self):
        below = NEAR_REF_CONFIDENCE_THRESHOLD - 0.01
        suggestions = [
            _suggestion("B001", [
                _ref("B002", confidence=0.9),   # included
                _ref("B003", confidence=below),  # excluded
            ])
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"] == ["B002"]
        assert "B003" not in m["B001"]


# ---------------------------------------------------------------------------
# Unresolved reference filtering
# ---------------------------------------------------------------------------

class TestUnresolvedFiltering:
    def test_none_matched_booking_no_excluded(self):
        suggestions = [_suggestion("B001", [_ref(None, confidence=0.95)])]
        assert _build_near_ref_map(suggestions) == {}

    def test_match_type_unresolved_excluded(self):
        suggestions = [
            _suggestion("B001", [_ref("B002", confidence=0.9, match_type="unresolved")])
        ]
        assert _build_near_ref_map(suggestions) == {}

    def test_mix_resolved_and_unresolved(self):
        suggestions = [
            _suggestion("B001", [
                _ref("B002", confidence=0.9, match_type="full_name"),   # included
                _ref(None,   confidence=0.9, match_type="unresolved"),   # excluded (no match)
                _ref("B003", confidence=0.9, match_type="unresolved"),   # excluded (match_type)
            ])
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"] == ["B002"]


# ---------------------------------------------------------------------------
# Match type acceptance
# ---------------------------------------------------------------------------

class TestMatchTypes:
    @pytest.mark.parametrize("match_type", [
        "full_name", "surname", "family", "first_name", "group_reference"
    ])
    def test_non_unresolved_match_types_accepted_when_above_threshold(self, match_type):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9, match_type=match_type)])]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m

    def test_unresolved_match_type_always_excluded(self):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=1.0, match_type="unresolved")])]
        assert _build_near_ref_map(suggestions) == {}


# ---------------------------------------------------------------------------
# Directionality: near-text references are NOT symmetric
# ---------------------------------------------------------------------------

class TestDirectionality:
    def test_a_wants_near_b_does_not_create_reverse_link(self):
        # A expresses wish to be near B
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9)])]
        m = _build_near_ref_map(suggestions)
        # A gets the link, B does NOT automatically get pulled toward A
        assert "B001" in m
        assert "B002" not in m

    def test_explicit_bidirectional_when_both_express_wish(self):
        suggestions = [
            _suggestion("B001", [_ref("B002")]),
            _suggestion("B002", [_ref("B001")]),
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"] == ["B002"]
        assert m["B002"] == ["B001"]


# ---------------------------------------------------------------------------
# De-duplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_target_from_two_refs_deduplicated(self):
        suggestions = [
            _suggestion("B001", [
                _ref("B002", confidence=0.9, raw_fragment="fragment 1"),
                _ref("B002", confidence=0.8, raw_fragment="fragment 2"),
            ])
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"].count("B002") == 1


# ---------------------------------------------------------------------------
# Booking with no cluster but with near-text ref
# ---------------------------------------------------------------------------

class TestNoClusterWithNearRef:
    def test_booking_with_only_near_ref_gets_entry(self):
        # This booking is not in any group_map cluster
        suggestions = [_suggestion("B099", [_ref("B042", confidence=0.85)])]
        m = _build_near_ref_map(suggestions)
        assert "B099" in m
        assert m["B099"] == ["B042"]


# ---------------------------------------------------------------------------
# Booking with both cluster membership and near-text refs
# ---------------------------------------------------------------------------

class TestClusterPlusNearRef:
    def test_union_of_cluster_and_near_ref_targets(self):
        """Simulates the allocation loop: merging group_map + near_ref_map."""
        group_map = {"B001": ["B002"]}           # from resolved_groups
        near_ref_map = {"B001": ["B003"]}         # from reference_resolutions

        co_members   = group_map.get("B001", [])
        near_targets = near_ref_map.get("B001", [])
        relevant     = set(co_members) | set(near_targets)

        assert relevant == {"B002", "B003"}

    def test_cluster_only_booking_unaffected(self):
        group_map = {"B004": ["B005"]}
        near_ref_map = {}                         # B004 has no near-text refs

        relevant = set(group_map.get("B004", [])) | set(near_ref_map.get("B004", []))
        assert relevant == {"B005"}

    def test_near_ref_only_booking_unaffected_by_missing_cluster(self):
        group_map = {}                            # B006 not in any cluster
        near_ref_map = {"B006": ["B007"]}

        relevant = set(group_map.get("B006", [])) | set(near_ref_map.get("B006", []))
        assert relevant == {"B007"}


# ---------------------------------------------------------------------------
# Custom threshold override
# ---------------------------------------------------------------------------

class TestCustomThreshold:
    def test_stricter_threshold_excludes_moderate_confidence(self):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.70)])]
        # Default threshold (0.65) includes it
        assert _build_near_ref_map(suggestions, threshold=0.65) != {}
        # Stricter threshold (0.80) excludes it
        assert _build_near_ref_map(suggestions, threshold=0.80) == {}

    def test_looser_threshold_includes_low_confidence(self):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.50)])]
        # Default threshold excludes
        assert _build_near_ref_map(suggestions) == {}
        # Looser threshold includes
        assert _build_near_ref_map(suggestions, threshold=0.40) != {}


# ---------------------------------------------------------------------------
# Symmetric types: group_reference links are bidirectional
# ---------------------------------------------------------------------------

class TestSymmetricTypes:
    def test_group_reference_injects_reverse_link(self):
        """A → B via group_reference must also add B → A."""
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9, match_type="group_reference")])]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m and "B002" in m["B001"]
        assert "B002" in m and "B001" in m["B002"]

    def test_personal_name_ref_does_not_inject_reverse(self):
        """full_name refs stay directional — B should NOT get a link back to A."""
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9, match_type="full_name")])]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m
        assert "B002" not in m

    def test_surname_ref_does_not_inject_reverse(self):
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9, match_type="surname")])]
        m = _build_near_ref_map(suggestions)
        assert "B002" not in m

    def test_group_reference_reverse_is_deduplicated(self):
        """When B also references A as group_reference, we should not get duplicates."""
        suggestions = [
            _suggestion("B001", [_ref("B002", confidence=0.9, match_type="group_reference")]),
            _suggestion("B002", [_ref("B001", confidence=0.9, match_type="group_reference")]),
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"].count("B002") == 1
        assert m["B002"].count("B001") == 1

    def test_group_reference_reverse_below_threshold_not_injected(self):
        """A below-threshold group_reference should not inject a reverse link either."""
        below = NEAR_REF_CONFIDENCE_THRESHOLD - 0.01
        suggestions = [_suggestion("B001", [_ref("B002", confidence=below, match_type="group_reference")])]
        m = _build_near_ref_map(suggestions)
        assert m == {}

    def test_symmetric_ref_types_constant_contains_group_reference(self):
        assert "group_reference" in SYMMETRIC_REF_TYPES

    def test_custom_symmetric_types_override(self):
        """Caller can make surname symmetric by passing a custom symmetric_types set."""
        suggestions = [_suggestion("B001", [_ref("B002", confidence=0.9, match_type="surname")])]
        m = _build_near_ref_map(suggestions, symmetric_types=frozenset({"surname"}))
        assert "B002" in m and "B001" in m["B002"]

    def test_multiple_group_ref_targets_all_get_reverse(self):
        """A single booking referencing multiple via group_reference gets reverse links for each."""
        suggestions = [_suggestion("B001", [
            _ref("B002", confidence=0.9, match_type="group_reference"),
            _ref("B003", confidence=0.9, match_type="group_reference"),
        ])]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m["B002"]
        assert "B001" in m["B003"]


# ---------------------------------------------------------------------------
# Constraint strength: near-ref source and target get +1 boost
# ---------------------------------------------------------------------------

from saronsdal.models.normalized import (
    Booking, RawGroupSignals, SpotRequest, VehicleUnit,
)


def _minimal_booking(booking_no: str, organization=None, group_field=None) -> Booking:
    """Minimal booking with no preferences and no group signals by default."""
    return Booking(
        booking_no=booking_no,
        booking_source="website",
        check_in=None,
        check_out=None,
        first_name="Test",
        last_name="Guest",
        full_name="Test Guest",
        num_guests=2,
        language="no",
        is_confirmed=True,
        vehicle=VehicleUnit(
            vehicle_type="caravan",
            spec_size_hint=None,
            body_length_m=7.0,
            body_width_m=2.75,
            fortelt_width_m=0.0,
            total_width_m=2.75,
            required_spot_count=1,
            has_fortelt=False,
            has_markise=False,
            registration=None,
            parse_confidence=1.0,
        ),
        request=SpotRequest(
            preferred_sections=[],
            preferred_spot_ids=[],
            avoid_sections=[],
            amenity_flags=set(),
            raw_near_texts=[],
            parse_confidence=1.0,
        ),
        group_signals=RawGroupSignals(
            organization=organization,
            group_field=group_field,
            near_text_fragments=[],
            is_org_private=True,
        ),
        data_confidence=1.0,
    )


class TestConstraintStrengthNearRefBoost:
    def test_no_near_ref_signal_no_boost(self):
        b = _minimal_booking("B001")
        s = _constraint_strength(b)
        assert s == 0

    def test_source_of_near_ref_gets_plus_one(self):
        b = _minimal_booking("B001")
        near_ref_map = {"B001": ["B002"]}
        s = _constraint_strength(b, near_ref_map=near_ref_map)
        assert s == 1

    def test_target_of_near_ref_gets_plus_one(self):
        b = _minimal_booking("B002")
        near_ref_target_set = {"B002"}
        s = _constraint_strength(b, near_ref_target_set=near_ref_target_set)
        assert s == 1

    def test_both_source_and_target_still_plus_one(self):
        """Being both source and target does not double-count."""
        b = _minimal_booking("B001")
        near_ref_map = {"B001": ["B002"]}
        near_ref_target_set = {"B001"}  # also a target
        s = _constraint_strength(b, near_ref_map=near_ref_map, near_ref_target_set=near_ref_target_set)
        assert s == 1

    def test_unrelated_booking_gets_no_boost(self):
        b = _minimal_booking("B099")
        near_ref_map = {"B001": ["B002"]}
        near_ref_target_set = {"B002"}
        s = _constraint_strength(b, near_ref_map=near_ref_map, near_ref_target_set=near_ref_target_set)
        assert s == 0

    def test_near_ref_boost_stacks_with_group_signal(self):
        b = _minimal_booking("B001", organization="TestOrg")
        near_ref_map = {"B001": ["B002"]}
        s = _constraint_strength(b, near_ref_map=near_ref_map)
        # +1 for group/org signal, +1 for near-ref source
        assert s == 2

    def test_near_ref_boost_stacks_with_section_pref(self):
        b = _minimal_booking("B001")
        b.request.preferred_sections.append("Furulunden")
        near_ref_target_set = {"B001"}
        s = _constraint_strength(b, near_ref_target_set=near_ref_target_set)
        # +2 for section pref, +1 for near-ref target
        assert s == 3


# ---------------------------------------------------------------------------
# place_refs integration: deterministic place/city group references
# ---------------------------------------------------------------------------

class TestPlaceRefs:
    """place_refs carry matched_booking_nos (plural) and must all be added as
    directed near-ref targets for the source booking."""

    def test_single_place_ref_with_multiple_targets(self):
        """One place_ref with three matched_booking_nos → three targets for source."""
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref(["B002", "B003", "B004"])])
        ]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m
        assert sorted(m["B001"]) == ["B002", "B003", "B004"]

    def test_multiple_place_refs_targets_merged(self):
        """Two place_refs on same booking — all targets merged into one list."""
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[
                            _place_ref(["B002", "B003"], fragment="frag1"),
                            _place_ref(["B004", "B005"], fragment="frag2"),
                        ])
        ]
        m = _build_near_ref_map(suggestions)
        assert sorted(m["B001"]) == ["B002", "B003", "B004", "B005"]

    def test_place_ref_targets_deduplicated(self):
        """Same target appearing in two place_refs is deduplicated."""
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[
                            _place_ref(["B002", "B003"], fragment="frag1"),
                            _place_ref(["B002", "B004"], fragment="frag2"),
                        ])
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"].count("B002") == 1

    def test_place_ref_below_threshold_excluded(self):
        below = NEAR_REF_CONFIDENCE_THRESHOLD - 0.01
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref(["B002"], confidence=below)])
        ]
        assert _build_near_ref_map(suggestions) == {}

    def test_place_ref_at_threshold_included(self):
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref(["B002"],
                                               confidence=NEAR_REF_CONFIDENCE_THRESHOLD)])
        ]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m

    def test_place_ref_unresolved_match_type_excluded(self):
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref(["B002"], match_type="unresolved")])
        ]
        assert _build_near_ref_map(suggestions) == {}

    def test_place_ref_directional_no_reverse_by_default(self):
        """place_group is not in SYMMETRIC_REF_TYPES — no reverse link injected."""
        assert "place_group" not in SYMMETRIC_REF_TYPES
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref(["B002", "B003"])])
        ]
        m = _build_near_ref_map(suggestions)
        assert "B001" in m
        assert "B002" not in m  # no reverse link
        assert "B003" not in m  # no reverse link

    def test_place_ref_and_resolved_ref_merged_for_same_booking(self):
        """place_refs and resolved_refs targets are merged into one list."""
        suggestions = [
            _suggestion("B001",
                        [_ref("B010", confidence=0.9)],  # resolved_ref target
                        place_refs=[_place_ref(["B002", "B003"])])  # place_ref targets
        ]
        m = _build_near_ref_map(suggestions)
        assert sorted(m["B001"]) == ["B002", "B003", "B010"]

    def test_place_ref_empty_matched_booking_nos_ignored(self):
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref([])])  # empty list
        ]
        assert _build_near_ref_map(suggestions) == {}

    def test_place_ref_none_in_matched_booking_nos_skipped(self):
        suggestions = [
            _suggestion("B001", [],
                        place_refs=[_place_ref([None, "B002", None])])
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"] == ["B002"]

    def test_place_ref_non_dict_entry_skipped(self):
        """Malformed (non-dict) entries in place_refs do not crash the builder."""
        suggestions = [
            _suggestion("B001", [],
                        place_refs=["bad_entry", None, _place_ref(["B002"])])
        ]
        m = _build_near_ref_map(suggestions)
        assert m["B001"] == ["B002"]

    def test_place_refs_none_treated_as_empty(self):
        """A NearTextSuggestion with place_refs=None doesn't crash."""
        s = NearTextSuggestion(
            booking_no="B001",
            full_name="Test",
            resolved_refs=[],
            unresolved_fragments=[],
            notes="",
            place_refs=None,
        )
        # Should not raise; place_refs=None means no place targets
        m = _build_near_ref_map([s])
        assert m == {}

    def test_30030_style_scenario(self):
        """Reproduce the booking-30030 pattern: one booking with a place_ref
        pointing to 6 co-attendees from the same city."""
        targets = ["30031", "30034", "30037", "30038", "30040", "30077"]
        suggestions = [
            _suggestion("30030", [],
                        place_refs=[_place_ref(targets, confidence=0.80)])
        ]
        m = _build_near_ref_map(suggestions)
        assert "30030" in m
        assert sorted(m["30030"]) == sorted(targets)
        # Targets do NOT get reverse links (directional by default)
        for t in targets:
            assert t not in m
