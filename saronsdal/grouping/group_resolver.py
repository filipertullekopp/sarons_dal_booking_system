"""Resolve the affinity graph into reviewable clusters.

Uses NetworkX internally for connected-component analysis.
All public return types are plain Python dataclasses (no NetworkX objects escape).

Two-level resolution:
  1. Find all connected components (booking + label nodes).
  2. For each component:
       a. If booking count > large_group_threshold AND all inter-booking
          connections go through a label node → LargeWeakCluster.
             i. Inside: find strong sub-pairs (explicit_family /
                explicit_named_people weight ≥ subcluster_min_strength)
                and resolve them as ResolvedCluster subclusters.
            ii. Remaining members with no strong sub-link → unaffiliated_members.
       b. Otherwise → ResolvedCluster.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import yaml

from saronsdal.models.grouping import (
    AffinityEdge,
    LargeWeakCluster,
    ResolvedCluster,
)

logger = logging.getLogger(__name__)

_CONFIG_ROOT = Path(__file__).parent.parent / "config"

_STRONG_EDGE_TYPES = {"explicit_family", "explicit_named_people"}
_LABEL_PREFIX = "label:"


def _load_rules(rules_path: Optional[Path] = None) -> dict:
    p = rules_path or _CONFIG_ROOT / "group_rules.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _node_id(node_type: str, value: str) -> str:
    """Internal graph node identifier."""
    if node_type == "label":
        return f"{_LABEL_PREFIX}{value}"
    return value  # booking_no as-is


def _is_label_node(node_id: str) -> bool:
    return node_id.startswith(_LABEL_PREFIX)


def _label_value(node_id: str) -> str:
    return node_id[len(_LABEL_PREFIX):]


def _cluster_type(edges: List[AffinityEdge]) -> str:
    types = {e.edge_type for e in edges}
    if types <= {"explicit_family"}:
        return "family_unit"
    if "explicit_named_people" in types:
        return "named_group"
    if types <= {"organization_membership", "group_field_membership"}:
        return "org_group"
    return "mixed"


def _member_edges(
    members: Set[str],
    edges_by_node: Dict[str, List[AffinityEdge]],
) -> List[AffinityEdge]:
    """Return all edges where BOTH endpoints are booking members (or label nodes
    that connect only members)."""
    result = []
    seen: Set[frozenset] = set()
    for m in members:
        for e in edges_by_node.get(m, []):
            pair = frozenset([e.node_a, e.node_b])
            if pair not in seen:
                seen.add(pair)
                result.append(e)
    return result


def resolve_groups(
    edges: List[AffinityEdge],
    all_booking_nos: List[str],
    rules_path: Optional[Path] = None,
) -> Tuple[List[ResolvedCluster], List[LargeWeakCluster]]:
    """
    Resolve affinity edges into clusters.

    Parameters
    ----------
    edges:
        All AffinityEdge objects from build_graph().
    all_booking_nos:
        Complete list of booking_nos (even those with no edges —
        they are simply not added to any cluster).
    rules_path:
        Optional override for group_rules.yaml.

    Returns
    -------
    clusters:         ResolvedCluster list (small / strong groups)
    large_clusters:   LargeWeakCluster list (big org groups, do_not_force)
    """
    rules = _load_rules(rules_path)
    thresholds = rules.get("thresholds", {})
    large_thresh = thresholds.get("large_group_threshold", 8)
    subcluster_min = thresholds.get("subcluster_min_strength", 0.80)

    # ----- Build NetworkX graph -----
    G = nx.Graph()

    # Add all booking nodes (even isolated ones won't form components > 1).
    for bno in all_booking_nos:
        G.add_node(bno, node_type="booking")

    # Index edges by node for fast membership lookup.
    edges_by_node: Dict[str, List[AffinityEdge]] = {}
    for e in edges:
        na = _node_id(e.node_a_type, e.node_a)
        nb = _node_id(e.node_b_type, e.node_b)
        G.add_node(na, node_type=e.node_a_type)
        G.add_node(nb, node_type=e.node_b_type)
        G.add_edge(na, nb, weight=e.weight, edge_type=e.edge_type)
        edges_by_node.setdefault(na, []).append(e)
        edges_by_node.setdefault(nb, []).append(e)

    clusters: List[ResolvedCluster] = []
    large_clusters: List[LargeWeakCluster] = []

    # ----- Process each connected component -----
    for component in nx.connected_components(G):
        # Separate booking and label nodes.
        booking_members: Set[str] = {
            n for n in component if not _is_label_node(n)
        }
        label_nodes: Set[str] = {
            n for n in component if _is_label_node(n)
        }

        if len(booking_members) < 2:
            # Isolated booking or only one member — no cluster.
            continue

        # Canonical label: prefer the highest-weight label node.
        canonical_label: Optional[str] = None
        if label_nodes:
            # Use the label node with the most booking-connections.
            canonical_label = _label_value(
                max(
                    label_nodes,
                    key=lambda ln: G.degree(ln),
                )
            )

        # Collect all edges relevant to this component's booking members.
        comp_edges = _member_edges(booking_members, edges_by_node)

        # Are there ANY direct booking↔booking edges?
        direct_bb_edges = [
            e for e in comp_edges
            if e.node_a_type == "booking" and e.node_b_type == "booking"
        ]

        # ----- Large weak cluster check -----
        if (
            len(booking_members) > large_thresh
            and not direct_bb_edges
        ):
            # All members connected only through a label node.
            # Find any strong sub-pairs hidden in the org's near-text signals.
            strong_sub_edges = [
                e for e in comp_edges
                if e.edge_type in _STRONG_EDGE_TYPES and e.weight >= subcluster_min
            ]
            subclusters = _build_subclusters(
                strong_sub_edges, booking_members
            )
            sub_members: Set[str] = set()
            for sc in subclusters:
                sub_members.update(sc.members)

            unaffiliated = sorted(booking_members - sub_members)

            lwc = LargeWeakCluster(
                cluster_id=f"lwc_{uuid.uuid4().hex[:8]}",
                canonical_label=canonical_label or "<unknown>",
                all_member_booking_nos=sorted(booking_members),
                subclusters=subclusters,
                unaffiliated_members=unaffiliated,
                review_flags=[],  # __post_init__ will add do_not_force_tight_cluster
            )
            large_clusters.append(lwc)
            logger.info(
                "LargeWeakCluster %s: %d members, %d subclusters, %d unaffiliated",
                lwc.cluster_id, len(booking_members),
                len(subclusters), len(unaffiliated),
            )
            continue

        # ----- Regular resolved cluster -----
        weights = [e.weight for e in comp_edges]
        ctype = _cluster_type(comp_edges)
        flags = []
        if len(booking_members) > large_thresh:
            flags.append("large_cluster_review")

        cluster = ResolvedCluster(
            cluster_id=f"cl_{uuid.uuid4().hex[:8]}",
            members=sorted(booking_members),
            canonical_label=canonical_label,
            cluster_type=ctype,
            min_link_strength=min(weights) if weights else 0.0,
            max_link_strength=max(weights) if weights else 0.0,
            internal_edges=comp_edges,
            review_flags=flags,
        )
        clusters.append(cluster)

    logger.info(
        "Group resolution: %d clusters, %d large-weak clusters",
        len(clusters), len(large_clusters),
    )
    return clusters, large_clusters


def _build_subclusters(
    strong_edges: List[AffinityEdge],
    all_members: Set[str],
) -> List[ResolvedCluster]:
    """
    Build ResolvedCluster objects from strong booking↔booking edges within
    a set of org members.  Uses a fresh NetworkX subgraph.
    """
    if not strong_edges:
        return []

    H = nx.Graph()
    for e in strong_edges:
        H.add_edge(e.node_a, e.node_b, weight=e.weight)

    # Only retain nodes that are actual booking members.
    to_remove = [n for n in H.nodes if n not in all_members]
    H.remove_nodes_from(to_remove)

    subclusters = []
    for comp in nx.connected_components(H):
        if len(comp) < 2:
            continue
        comp_edges = [
            e for e in strong_edges
            if e.node_a in comp and e.node_b in comp
        ]
        weights = [e.weight for e in comp_edges]
        sc = ResolvedCluster(
            cluster_id=f"sc_{uuid.uuid4().hex[:8]}",
            members=sorted(comp),
            canonical_label=None,
            cluster_type=_cluster_type(comp_edges),
            min_link_strength=min(weights),
            max_link_strength=max(weights),
            internal_edges=comp_edges,
            review_flags=["subcluster_within_large_group"],
        )
        subclusters.append(sc)
    return subclusters
