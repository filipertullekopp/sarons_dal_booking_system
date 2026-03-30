"""Regression tests for allocator._build_group_map.

Specifically guards against schema drift between the Phase 2 ResolvedCluster
model and the allocator's group-map construction.
"""
import pytest
from types import SimpleNamespace

from saronsdal.allocation.allocator import _build_group_map
from saronsdal.models.grouping import ResolvedCluster, AffinityEdge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cluster(members, cluster_id="C1"):
    """Build a minimal ResolvedCluster using the current schema (members)."""
    return ResolvedCluster(
        cluster_id=cluster_id,
        members=members,
        canonical_label=None,
        cluster_type="name_ref",
        min_link_strength=1.0,
        max_link_strength=1.0,
        internal_edges=[],
    )


def _old_cluster(member_booking_nos, cluster_id="C1"):
    """Simulate a cluster object using the old schema field name."""
    return SimpleNamespace(
        cluster_id=cluster_id,
        member_booking_nos=member_booking_nos,
    )


def _neither_cluster(cluster_id="C1"):
    """Simulate a cluster object with neither field (corrupt data)."""
    return SimpleNamespace(cluster_id=cluster_id)


# ---------------------------------------------------------------------------
# Current schema (members)
# ---------------------------------------------------------------------------

class TestCurrentSchema:
    def test_two_member_cluster(self):
        clusters = [_cluster(["B001", "B002"])]
        gmap = _build_group_map(clusters)
        assert gmap["B001"] == ["B002"]
        assert gmap["B002"] == ["B001"]

    def test_three_member_cluster(self):
        clusters = [_cluster(["B001", "B002", "B003"])]
        gmap = _build_group_map(clusters)
        assert sorted(gmap["B001"]) == ["B002", "B003"]
        assert sorted(gmap["B002"]) == ["B001", "B003"]
        assert sorted(gmap["B003"]) == ["B001", "B002"]

    def test_single_member_cluster(self):
        # A cluster of one has no co-members
        clusters = [_cluster(["B001"])]
        gmap = _build_group_map(clusters)
        assert gmap["B001"] == []

    def test_two_separate_clusters(self):
        clusters = [
            _cluster(["B001", "B002"], cluster_id="C1"),
            _cluster(["B003", "B004"], cluster_id="C2"),
        ]
        gmap = _build_group_map(clusters)
        assert gmap["B001"] == ["B002"]
        assert gmap["B003"] == ["B004"]
        # Members of different clusters are not linked
        assert "B003" not in gmap["B001"]

    def test_booking_only_appears_in_own_cluster(self):
        clusters = [
            _cluster(["B001", "B002"]),
            _cluster(["B003", "B004"]),
        ]
        gmap = _build_group_map(clusters)
        assert set(gmap["B001"]) == {"B002"}

    def test_uses_members_field_on_real_dataclass(self):
        """ResolvedCluster.members is the canonical field — must not raise AttributeError."""
        cluster = _cluster(["X1", "X2"])
        assert hasattr(cluster, "members")
        assert not hasattr(cluster, "member_booking_nos")
        gmap = _build_group_map([cluster])
        assert "X1" in gmap


# ---------------------------------------------------------------------------
# Backward compatibility (member_booking_nos — old schema)
# ---------------------------------------------------------------------------

class TestOldSchema:
    def test_old_field_name_works(self):
        clusters = [_old_cluster(["B001", "B002"])]
        gmap = _build_group_map(clusters)
        assert gmap["B001"] == ["B002"]
        assert gmap["B002"] == ["B001"]

    def test_old_field_three_members(self):
        clusters = [_old_cluster(["B001", "B002", "B003"])]
        gmap = _build_group_map(clusters)
        assert sorted(gmap["B001"]) == ["B002", "B003"]

    def test_mix_of_old_and_new_schema(self):
        clusters = [
            _cluster(["B001", "B002"]),
            _old_cluster(["B003", "B004"]),
        ]
        gmap = _build_group_map(clusters)
        assert gmap["B001"] == ["B002"]
        assert gmap["B003"] == ["B004"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_cluster_list(self):
        assert _build_group_map([]) == {}

    def test_none_clusters(self):
        assert _build_group_map(None) == {}

    def test_cluster_with_neither_field_is_skipped(self, caplog):
        import logging
        clusters = [_neither_cluster("BAD"), _cluster(["B001", "B002"])]
        with caplog.at_level(logging.WARNING):
            gmap = _build_group_map(clusters)
        # The bad cluster is skipped; the good one still maps
        assert gmap["B001"] == ["B002"]
        assert any("neither" in m.lower() for m in caplog.messages)

    def test_empty_members_list(self):
        clusters = [_cluster([])]
        gmap = _build_group_map(clusters)
        assert gmap == {}
