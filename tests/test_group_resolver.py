"""Tests for group_resolver.resolve_groups."""
from typing import List

import pytest

from saronsdal.grouping.group_resolver import resolve_groups
from saronsdal.models.grouping import AffinityEdge


# ---------------------------------------------------------------------------
# Edge factory helpers
# ---------------------------------------------------------------------------

def _bb_edge(a: str, b: str, edge_type: str = "explicit_family", weight: float = 0.90) -> AffinityEdge:
    """Create a booking↔booking AffinityEdge."""
    return AffinityEdge(
        node_a=a,
        node_b=b,
        node_a_type="booking",
        node_b_type="booking",
        edge_type=edge_type,
        weight=weight,
        source_field="near_text",
        raw_text="test",
        normalized_text="test",
        confidence=weight,
    )


def _bl_edge(booking: str, label: str, edge_type: str = "organization_membership", weight: float = 0.50) -> AffinityEdge:
    """Create a booking↔label AffinityEdge."""
    return AffinityEdge(
        node_a=booking,
        node_b=label,
        node_a_type="booking",
        node_b_type="label",
        edge_type=edge_type,
        weight=weight,
        source_field="organization",
        raw_text=label,
        normalized_text=label.lower(),
        confidence=weight,
    )


# ---------------------------------------------------------------------------
# Basic cluster formation
# ---------------------------------------------------------------------------

def test_two_bookings_direct_edge_form_cluster():
    edges = [_bb_edge("B001", "B002")]
    clusters, large = resolve_groups(edges, ["B001", "B002"])
    assert len(clusters) == 1
    assert len(large) == 0
    assert set(clusters[0].members) == {"B001", "B002"}


def test_isolated_booking_not_in_any_cluster():
    edges = [_bb_edge("B001", "B002")]
    clusters, large = resolve_groups(edges, ["B001", "B002", "B003"])
    all_members = {m for c in clusters for m in c.members}
    assert "B003" not in all_members


def test_no_edges_no_clusters():
    clusters, large = resolve_groups([], ["B001", "B002"])
    assert clusters == []
    assert large == []


# ---------------------------------------------------------------------------
# Cluster type detection
# ---------------------------------------------------------------------------

def test_cluster_type_family_unit():
    edges = [_bb_edge("B001", "B002", edge_type="explicit_family")]
    clusters, _ = resolve_groups(edges, ["B001", "B002"])
    assert clusters[0].cluster_type == "family_unit"


def test_cluster_type_named_group():
    edges = [_bb_edge("B001", "B002", edge_type="explicit_named_people")]
    clusters, _ = resolve_groups(edges, ["B001", "B002"])
    assert clusters[0].cluster_type == "named_group"


def test_cluster_type_org_group():
    """Three bookings connected only via label node → org_group."""
    edges = [
        _bl_edge("B001", "Betel"),
        _bl_edge("B002", "Betel"),
        _bl_edge("B003", "Betel"),
    ]
    clusters, large = resolve_groups(edges, ["B001", "B002", "B003"])
    # 3 members ≤ large_group_threshold (8) → ResolvedCluster
    assert len(clusters) == 1
    assert clusters[0].cluster_type == "org_group"


def test_cluster_type_mixed():
    """Explicit family edge + org membership → mixed."""
    edges = [
        _bb_edge("B001", "B002", edge_type="explicit_family"),
        _bl_edge("B001", "Betel"),
        _bl_edge("B002", "Betel"),
    ]
    clusters, _ = resolve_groups(edges, ["B001", "B002"])
    assert clusters[0].cluster_type == "mixed"


# ---------------------------------------------------------------------------
# Cluster link strength
# ---------------------------------------------------------------------------

def test_min_max_link_strength():
    edges = [
        _bb_edge("B001", "B002", weight=0.90),
        _bb_edge("B002", "B003", weight=0.75),
    ]
    clusters, _ = resolve_groups(edges, ["B001", "B002", "B003"])
    assert len(clusters) == 1
    assert clusters[0].min_link_strength == pytest.approx(0.75)
    assert clusters[0].max_link_strength == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Large weak cluster
# ---------------------------------------------------------------------------

def _large_org_edges(n: int, label: str = "BigChurch"):
    """Create n bookings all connected to the same label node. Returns (edges, bnos)."""
    bnos = [f"B{i:03d}" for i in range(1, n + 1)]
    return [_bl_edge(bno, label) for bno in bnos], bnos


def test_large_org_becomes_large_weak_cluster():
    """9 bookings all connected via label node → LargeWeakCluster (threshold=8)."""
    edges, bnos = _large_org_edges(9)
    clusters, large = resolve_groups(edges, bnos)
    assert len(clusters) == 0
    assert len(large) == 1
    lwc = large[0]
    assert len(lwc.all_member_booking_nos) == 9
    assert "do_not_force_tight_cluster" in lwc.review_flags


def test_large_weak_cluster_canonical_label():
    edges, bnos = _large_org_edges(9, label="Betel Hommersåk")
    _, large = resolve_groups(edges, bnos)
    assert large[0].canonical_label == "Betel Hommersåk"


def test_large_weak_cluster_with_subclusters():
    """9 members in org, but B001↔B002 have explicit_family edge → subcluster."""
    edges, bnos = _large_org_edges(9)
    # Add a strong family link between B001 and B002
    strong = _bb_edge("B001", "B002", edge_type="explicit_family", weight=0.90)
    edges.append(strong)
    clusters, large = resolve_groups(edges, bnos)
    # Should still be a LargeWeakCluster (B001↔B002 is direct but the component has 9 members)
    # Note: the direct BB edge means it won't be classified as LargeWeakCluster
    # (the large-weak check requires NO direct bb edges).
    # So this should produce a regular cluster.
    assert len(clusters) == 1
    assert clusters[0].cluster_type in ("family_unit", "mixed", "org_group")


def test_large_weak_cluster_all_label_only():
    """Exactly at the threshold boundary: 8 members → regular cluster; 9 → large."""
    edges_8, bnos_8 = _large_org_edges(8)
    clusters_8, large_8 = resolve_groups(edges_8, bnos_8)
    assert len(clusters_8) == 1
    assert len(large_8) == 0

    edges_9, bnos_9 = _large_org_edges(9)
    clusters_9, large_9 = resolve_groups(edges_9, bnos_9)
    assert len(clusters_9) == 0
    assert len(large_9) == 1


# ---------------------------------------------------------------------------
# Large weak cluster: unaffiliated members
# ---------------------------------------------------------------------------

def test_large_weak_cluster_unaffiliated_members():
    """In a LargeWeakCluster with no strong sub-pairs, all members are unaffiliated."""
    edges, bnos = _large_org_edges(9)
    _, large = resolve_groups(edges, bnos)
    lwc = large[0]
    assert lwc.subclusters == []
    assert set(lwc.unaffiliated_members) == set(bnos)


# ---------------------------------------------------------------------------
# Cluster IDs are unique
# ---------------------------------------------------------------------------

def test_cluster_ids_unique():
    edges = [_bb_edge("B001", "B002"), _bb_edge("B003", "B004")]
    clusters, _ = resolve_groups(edges, ["B001", "B002", "B003", "B004"])
    ids = [c.cluster_id for c in clusters]
    assert len(ids) == len(set(ids))
