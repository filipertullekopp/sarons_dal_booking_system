"""Tests for group-context seeding (first-in-cluster anchor placement).

Covers:
  - GroupContext and compute_seed_score() in group_scorer
  - _build_group_context_map() in allocator
  - rank_candidates() seeding integration
  - Explanation dict fields in AllocationResult
"""
import re
import pytest
from typing import List, Optional, Set

from tests.conftest import FIXTURES_DIR
from saronsdal.allocation.allocator import (
    _build_group_context_map,
    _build_group_map,
    _build_near_ref_map,
    _build_subsection_map,
)
from saronsdal.allocation.candidate_ranker import rank_candidates, W_PREF, W_GROUP
from saronsdal.allocation.group_scorer import GroupContext, GroupScore, compute_seed_score
from saronsdal.models.normalized import (
    Booking, RawGroupSignals, SectionRow, Spot, SpotRequest, VehicleUnit,
)
from saronsdal.models.triplet import Triplet
from saronsdal.spatial.topology_loader import load_topology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FURULUNDEN_ONLY = {"furulunden": "topology_grid_furulunden_mini.csv"}


@pytest.fixture(scope="module")
def topo():
    return load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)


# ---------------------------------------------------------------------------
# Minimal factories
# ---------------------------------------------------------------------------

def _spot(
    section: str = "Furulunden",
    spot_id: str = "A1",
    row: str = "A",
    length_m: float = 10.0,
    no_motorhome: bool = False,
    no_caravan: bool = False,
    is_not_spot: bool = False,
    is_reserved: bool = False,
) -> Spot:
    return Spot(
        spot_id=spot_id,
        section=section,
        row=row,
        position=1,
        length_m=length_m,
        hilliness=0,
        is_end_of_row=False,
        is_not_spot=is_not_spot,
        is_reserved=is_reserved,
        no_motorhome=no_motorhome,
        no_caravan_nor_motorhome=no_caravan,
    )


def _booking(
    booking_no: str = "B001",
    preferred_sections: Optional[List[str]] = None,
    preferred_spot_ids: Optional[List[str]] = None,
    preferred_rows: Optional[List[SectionRow]] = None,
    organization: Optional[str] = None,
    group_field: Optional[str] = None,
    vehicle_type: str = "caravan",
) -> Booking:
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
            vehicle_type=vehicle_type,
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
            preferred_sections=preferred_sections or [],
            preferred_spot_ids=preferred_spot_ids or [],
            preferred_section_rows=preferred_rows or [],
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


def _group_context(
    section_weights=None,
    row_weights=None,
    spot_id_weights=None,
    contributing=None,
    sources=None,
) -> GroupContext:
    return GroupContext(
        section_weights=section_weights or {},
        row_weights=row_weights or {},
        spot_id_weights=spot_id_weights or {},
        contributing_booking_nos=contributing or [],
        contributor_sources=sources or {},
    )


def _triplet(
    section: str = "Furulunden",
    spot_id: str = "A1",
    row: str = "A",
    length_m: float = 10.0,
) -> Triplet:
    m = re.match(r"([A-Z]+)(\d+)", spot_id)
    row_letter = m.group(1)
    first_num = int(m.group(2))
    spot_ids = [f"{row_letter}{first_num + i}" for i in range(3)]
    anchor = _spot(section, spot_ids[0], row or row_letter, length_m)
    return Triplet(
        room_id=f"{section} {spot_ids[0]}-{spot_ids[2]}",
        section=section,
        row=row or row_letter,
        spot_ids=spot_ids,
        first_spot=anchor,
    )


# ---------------------------------------------------------------------------
# compute_seed_score: basic mechanics
# ---------------------------------------------------------------------------

class TestComputeSeedScore:
    def test_empty_context_returns_zero(self, topo):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context()
        gs = compute_seed_score(spot, ctx)
        assert gs.score == 0.0

    def test_section_match_returns_nonzero(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 2})
        gs = compute_seed_score(spot, ctx)
        assert gs.score > 0.0

    def test_section_no_match_returns_zero(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Bibelskolen": 3})
        gs = compute_seed_score(spot, ctx)
        assert gs.score == 0.0

    def test_full_section_match_score_is_one(self):
        """Single section, single supporter, spot is in that section → score = 1.0."""
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 1})
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(1.0)

    def test_partial_section_weight(self):
        """2 of 3 supporters prefer this section → section_component = 2/3."""
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 2, "Bibelskolen": 1})
        gs = compute_seed_score(spot, ctx)
        # Only section dimension is non-empty → score = section_component = 2/3
        # abs tolerance accounts for 4-decimal rounding in compute_seed_score
        assert gs.score == pytest.approx(2 / 3, abs=1e-3)

    def test_row_match(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(row_weights={"A": 1})
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(1.0)

    def test_row_no_match(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(row_weights={"B": 1})
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(0.0)

    def test_spot_id_match(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(spot_id_weights={"A1": 1})
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(1.0)

    def test_spot_id_no_match(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(spot_id_weights={"B2": 1})
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(0.0)

    def test_all_three_dimensions_match(self):
        """All three dimensions match → mean of three 1.0 components = 1.0."""
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(
            section_weights={"Furulunden": 1},
            row_weights={"A": 1},
            spot_id_weights={"A1": 1},
        )
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(1.0)

    def test_two_dimensions_partial_match(self):
        """Section matches (1/1), row doesn't (0/1) → mean = (1.0 + 0.0) / 2 = 0.5."""
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(
            section_weights={"Furulunden": 1},
            row_weights={"B": 1},  # wrong row
        )
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(0.5)

    def test_missing_dimension_excluded_from_mean(self):
        """A context with only section data should not penalise the absence of rows."""
        spot = _spot("Furulunden", "A1", "A")
        ctx_section_only = _group_context(section_weights={"Furulunden": 1})
        ctx_both = _group_context(
            section_weights={"Furulunden": 1},
            row_weights={"A": 1},
        )
        # Section-only: score = 1.0 (only one dimension)
        assert compute_seed_score(spot, ctx_section_only).score == pytest.approx(1.0)
        # Both matching: score = 1.0 (both match)
        assert compute_seed_score(spot, ctx_both).score == pytest.approx(1.0)

    def test_score_range_is_zero_to_one(self):
        """Score is always in [0, 1] regardless of weight values."""
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(
            section_weights={"Furulunden": 5, "Bibelskolen": 1},
            row_weights={"A": 3, "B": 2},
        )
        gs = compute_seed_score(spot, ctx)
        assert 0.0 <= gs.score <= 1.0

    def test_sentinel_fields(self):
        """mean_distance=0.0 and member_count=0 are sentinels for seeded results."""
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 1})
        gs = compute_seed_score(spot, ctx)
        assert gs.mean_distance == 0.0
        assert gs.member_count == 0

    def test_notes_populated(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(
            section_weights={"Furulunden": 2},
            contributing=["B002"],
            sources={"B002": "cluster"},
        )
        gs = compute_seed_score(spot, ctx)
        assert len(gs.notes) > 0

    def test_match_note_contains_vote_count(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(
            section_weights={"Furulunden": 2, "Bibelskolen": 1},
            contributing=["B002", "B003"],
            sources={"B002": "cluster", "B003": "near_ref"},
        )
        gs = compute_seed_score(spot, ctx)
        # At least one note should mention the vote ratio
        assert any("2" in n and "3" in n for n in gs.notes)

    def test_no_match_note_shows_preferred_sections(self):
        spot = _spot("Furulunden", "A1", "A")  # Furulunden
        ctx = _group_context(
            section_weights={"Bibelskolen": 3},  # group prefers elsewhere
        )
        gs = compute_seed_score(spot, ctx)
        assert any("Bibelskolen" in n for n in gs.notes)


# ---------------------------------------------------------------------------
# compute_seed_score: weighted ordering (repeated preference = stronger pull)
# ---------------------------------------------------------------------------

class TestSeedScoreWeightedOrdering:
    def test_higher_weight_section_scores_higher(self):
        spot_high = _spot("Furulunden", "A1", "A")  # preferred by 3
        spot_low = _spot("Bibelskolen", "D1", "D")  # preferred by 1
        ctx = _group_context(section_weights={"Furulunden": 3, "Bibelskolen": 1})
        score_high = compute_seed_score(spot_high, ctx).score
        score_low = compute_seed_score(spot_low, ctx).score
        assert score_high > score_low

    def test_unanimous_preference_scores_one(self):
        spot = _spot("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 5})  # all 5 agree
        gs = compute_seed_score(spot, ctx)
        assert gs.score == pytest.approx(1.0)

    def test_spot_with_most_votes_ranks_first(self):
        spots = [
            _spot("Furulunden", "A1", "A"),  # 0 spot votes
            _spot("Furulunden", "B2", "B"),  # 2 spot votes
        ]
        ctx = _group_context(spot_id_weights={"B2": 2, "A1": 0})
        s_A1 = compute_seed_score(spots[0], ctx).score
        s_B2 = compute_seed_score(spots[1], ctx).score
        assert s_B2 > s_A1


# ---------------------------------------------------------------------------
# _build_group_context_map
# ---------------------------------------------------------------------------

class TestBuildGroupContextMap:
    def test_no_links_produces_empty_map(self):
        bookings = [_booking("B001"), _booking("B002")]
        result = _build_group_context_map(bookings, {}, {}, {})
        assert result == {}

    def test_cluster_member_preferences_collected(self):
        b1 = _booking("B001")  # no prefs
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        group_map = {"B001": ["B002"], "B002": ["B001"]}
        result = _build_group_context_map([b1, b2], group_map, {}, {})
        # B001's context should contain B002's section preference
        assert "B001" in result
        assert result["B001"].section_weights.get("Furulunden", 0) == 1

    def test_near_ref_member_preferences_collected(self):
        b1 = _booking("B001")
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        near_ref_map = {"B001": ["B002"]}
        result = _build_group_context_map([b1, b2], {}, near_ref_map, {})
        assert "B001" in result
        assert result["B001"].section_weights.get("Furulunden", 0) == 1

    def test_multiple_members_vote_weights_accumulate(self):
        b1 = _booking("B001")
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        b3 = _booking("B003", preferred_sections=["Furulunden"])
        b4 = _booking("B004", preferred_sections=["Bibelskolen"])
        group_map = {"B001": ["B002", "B003", "B004"]}
        result = _build_group_context_map([b1, b2, b3, b4], group_map, {}, {})
        ctx = result["B001"]
        assert ctx.section_weights["Furulunden"] == 2
        assert ctx.section_weights["Bibelskolen"] == 1

    def test_own_preferences_not_included(self):
        """The booking itself should not vote in its own group context."""
        b1 = _booking("B001", preferred_sections=["Bibelskolen"])
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        group_map = {"B001": ["B002"], "B002": ["B001"]}
        result = _build_group_context_map([b1, b2], group_map, {}, {})
        # B001's context is built from B002 only — should see Furulunden, not Bibelskolen
        ctx = result["B001"]
        assert ctx.section_weights.get("Furulunden", 0) == 1
        assert ctx.section_weights.get("Bibelskolen", 0) == 0

    def test_member_with_no_preferences_excluded_from_contributing(self):
        b1 = _booking("B001")
        b2 = _booking("B002")  # no preferences at all
        group_map = {"B001": ["B002"]}
        result = _build_group_context_map([b1, b2], group_map, {}, {})
        # B002 has no preferences → no context for B001 (no weights)
        assert "B001" not in result

    def test_contributor_sources_cluster(self):
        b1 = _booking("B001")
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        group_map = {"B001": ["B002"]}
        result = _build_group_context_map([b1, b2], group_map, {}, {})
        assert result["B001"].contributor_sources.get("B002") == "cluster"

    def test_contributor_sources_near_ref(self):
        b1 = _booking("B001")
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        near_ref_map = {"B001": ["B002"]}
        result = _build_group_context_map([b1, b2], {}, near_ref_map, {})
        assert result["B001"].contributor_sources.get("B002") == "near_ref"

    def test_contributor_sources_both(self):
        b1 = _booking("B001")
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        group_map = {"B001": ["B002"]}
        near_ref_map = {"B001": ["B002"]}
        result = _build_group_context_map([b1, b2], group_map, near_ref_map, {})
        assert result["B001"].contributor_sources.get("B002") == "both"

    def test_per_member_deduplication(self):
        """A member listing the same section twice still counts as one vote."""
        b1 = _booking("B001")
        # B002 has the same section appearing twice (once explicitly, once via Phase 2.5)
        # We simulate this by having two identical sections in preferred_sections
        b2 = _booking("B002", preferred_sections=["Furulunden", "Furulunden"])
        group_map = {"B001": ["B002"]}
        result = _build_group_context_map([b1, b2], group_map, {}, {})
        # Deduplication: one member = one vote, not two
        assert result["B001"].section_weights["Furulunden"] == 1

    def test_unknown_booking_ref_skipped(self):
        """A near_ref_map entry pointing to a booking not in our dataset is ignored."""
        b1 = _booking("B001", preferred_sections=["Furulunden"])
        near_ref_map = {"B001": ["B999"]}  # B999 doesn't exist
        result = _build_group_context_map([b1], {}, near_ref_map, {})
        assert "B001" not in result  # no context because B999 has no data


# ---------------------------------------------------------------------------
# rank_candidates seeding integration
# ---------------------------------------------------------------------------

class TestRankCandidatesSeeding:
    def test_seeding_active_when_no_group_assigned(self, topo):
        """When group_spots is empty, group_context is used for scoring."""
        booking = _booking("B001")
        t_match = _triplet("Furulunden", "A1", "A")  # matches seeded section
        t_other = _triplet("Furulunden", "B1", "B")  # different row
        ctx = _group_context(section_weights={"Furulunden": 2}, row_weights={"A": 2})
        ranked = rank_candidates(
            booking=booking,
            candidates=[t_match, t_other],
            topo=topo,
            group_spots=None,
            group_context=ctx,
        )
        assert len(ranked) == 2
        # A1 matches section AND row → higher seed score → should rank first
        assert ranked[0].spot.spot_id == "A1"

    def test_seeding_not_active_when_group_assigned(self, topo):
        """Once co-members are assigned, group_context is ignored."""
        booking = _booking("B001")
        t_match = _triplet("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Bibelskolen": 5})  # strong seed elsewhere
        # But a co-member is already assigned in Furulunden
        ranked_with_group = rank_candidates(
            booking=booking,
            candidates=[t_match],
            topo=topo,
            group_spots=[("Furulunden", "A2")],
            group_context=ctx,  # should be ignored
        )
        ranked_without_group = rank_candidates(
            booking=booking,
            candidates=[t_match],
            topo=topo,
            group_spots=[("Furulunden", "A2")],
            group_context=None,
        )
        # Scores must be identical — context was ignored
        assert ranked_with_group[0].total_score == ranked_without_group[0].total_score

    def test_seeded_group_score_in_result(self, topo):
        """RankedCandidate.group_score reflects the seed score when seeding is active."""
        booking = _booking("B001")
        t = _triplet("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 1})
        ranked = rank_candidates(
            booking=booking,
            candidates=[t],
            topo=topo,
            group_spots=None,
            group_context=ctx,
        )
        assert len(ranked) == 1
        assert ranked[0].group_score > 0.0

    def test_total_score_formula_with_seeding(self, topo):
        """total = W_PREF * pref + W_GROUP * seed_score."""
        booking = _booking("B001")
        t = _triplet("Furulunden", "A1", "A")
        ctx = _group_context(section_weights={"Furulunden": 1})
        ranked = rank_candidates(
            booking=booking,
            candidates=[t],
            topo=topo,
            group_spots=None,
            group_context=ctx,
        )
        r = ranked[0]
        expected = W_PREF * r.preference_score + W_GROUP * r.group_score
        assert r.total_score == pytest.approx(expected, abs=1e-4)

    def test_no_context_no_seeding_baseline_behaviour(self, topo):
        """With no group and no context, group_score = 0 (unchanged behaviour)."""
        booking = _booking("B001")
        t = _triplet("Furulunden", "A1", "A")
        ranked = rank_candidates(
            booking=booking,
            candidates=[t],
            topo=topo,
            group_spots=None,
            group_context=None,
        )
        assert ranked[0].group_score == 0.0

    def test_strong_own_preference_dominates_conflicting_seed(self, topo):
        """W_PREF=0.7 ensures explicit spot preference outweighs a conflicting seed."""
        # B001 explicitly requests A1
        booking = _booking("B001", preferred_spot_ids=["A1"])
        t_a1 = _triplet("Furulunden", "A1", "A")
        t_b2 = _triplet("Furulunden", "B2", "B")
        # Group context strongly prefers B row
        ctx = _group_context(row_weights={"B": 10})
        ranked = rank_candidates(
            booking=booking,
            candidates=[t_a1, t_b2],
            topo=topo,
            group_spots=None,
            preferred_spot_ids=["A1"],
            group_context=ctx,
        )
        # A1 should still rank first because booking's explicit spot preference
        # is captured in the 0.7-weighted pref_score
        assert ranked[0].spot.spot_id == "A1"


# ---------------------------------------------------------------------------
# Explanation dict fields
# ---------------------------------------------------------------------------

class TestExplanationFields:
    def test_seeded_explanation_contains_context_fields(self):
        """When seeding was used, explanation must include all group_context_* keys.

        Scenario: B001 has preferred_spot_ids → constraint_strength=10, sorts first.
        B002 has preferred_sections → constraint_strength=2, sorts second.
        B001 is first-in-cluster, so it gets seeded from B002's section preference.
        """
        from saronsdal.allocation.allocator import allocate
        from saronsdal.models.grouping import ResolvedCluster
        from saronsdal.spatial.topology_loader import load_topology

        # B001 has a spot ID request → constraint strength +10 → sorts first
        b1 = _booking("B001", preferred_spot_ids=["A1"])
        # B002 has a section preference → constraint strength +2 → sorts second
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        triplets = [_triplet("Furulunden", "A1", "A"), _triplet("Furulunden", "B2", "B")]

        cluster = ResolvedCluster(
            cluster_id="C1",
            members=["B001", "B002"],
            canonical_label=None,
            cluster_type="name_ref",
            min_link_strength=1.0,
            max_link_strength=1.0,
            internal_edges=[],
        )

        topo_local = load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)
        results = allocate(
            bookings=[b1, b2],
            triplets=triplets,
            topo=topo_local,
            clusters=[cluster],
        )

        # B001 sorts first (strength=10) and is first-in-cluster → seeded from B002
        b1_result = next(r for r in results if r.booking_no == "B001")
        assert b1_result.explanation.get("group_context_seeded") is True
        exp = b1_result.explanation
        assert "group_context_contributors" in exp
        assert "group_context_section_weights" in exp
        assert "group_context_row_weights" in exp
        assert "group_context_spot_id_weights" in exp
        # B002's section preference should appear in the seeded weights
        assert exp["group_context_section_weights"].get("Furulunden", 0) == 1

    def test_non_seeded_explanation_has_seeded_false(self):
        """The second cluster member uses proximity scoring, not seeding."""
        from saronsdal.allocation.allocator import allocate
        from saronsdal.models.grouping import ResolvedCluster
        from saronsdal.spatial.topology_loader import load_topology

        # B001 sorts first (spot IDs → strength=10), B002 sorts second (sections → 2)
        b1 = _booking("B001", preferred_spot_ids=["A1"])
        b2 = _booking("B002", preferred_sections=["Furulunden"])
        triplets = [_triplet("Furulunden", "A1", "A"), _triplet("Furulunden", "B2", "B")]

        cluster = ResolvedCluster(
            cluster_id="C1",
            members=["B001", "B002"],
            canonical_label=None,
            cluster_type="name_ref",
            min_link_strength=1.0,
            max_link_strength=1.0,
            internal_edges=[],
        )

        topo_local = load_topology(FIXTURES_DIR, filenames=_FURULUNDEN_ONLY)
        results = allocate(
            bookings=[b1, b2],
            triplets=triplets,
            topo=topo_local,
            clusters=[cluster],
        )

        # B002 is placed second — B001 already assigned → proximity, not seeding
        b2_result = next(r for r in results if r.booking_no == "B002")
        assert b2_result.explanation.get("group_context_seeded") is False
