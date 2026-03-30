"""Typed domain models for Phase 2: group detection and resolution.

Design principles:
- ExtractedReference = raw regex output; no booking-to-booking matching yet.
- ResolvedReference = matched to specific booking(s); is_ambiguous=True when
  multiple bookings matched (no edge created in that case).
- AffinityEdge connects either two booking nodes or a booking node to a label
  node.  Label nodes represent canonical org/group names.  The edge type
  encodes the strength category; weight encodes the numeric value.
- AliasMatchSuggestion and AmbiguousGroupCase are review artifacts only —
  they never drive cluster formation.
- LargeWeakCluster wraps a canonical org group whose members must NOT be
  forced into one tight allocation.  Strong sub-pairs live in subclusters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

EdgeType = Literal[
    "explicit_family",
    "explicit_named_people",
    "group_field_membership",
    "organization_membership",
]

RefType = Literal[
    "family",           # "fam. Husvik" — surname only
    "full_name",        # "Otto Husvik" — first + last
    "first_name_only",  # single capitalised word after proximity trigger
    "organization",     # capitalised multi-word that looks like an org, not a person
    "ambiguous",        # could be family, org, or person — not enough context
]

NodeType = Literal["booking", "label"]

ClusterType = Literal[
    "family_unit",   # all edges are explicit_family
    "named_group",   # at least one explicit_named_people edge
    "org_group",     # connected only via org/group_field membership
    "mixed",         # combination of the above
]


# ---------------------------------------------------------------------------
# Extraction (per-booking, no cross-booking lookup yet)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedReference:
    """
    Raw output of name_reference_extractor for one text fragment.

    This is NOT yet matched to any booking.  It is purely a regex/pattern result
    with a type guess and a confidence estimate.
    """
    raw_text: str                   # original substring that matched
    normalized_candidate: str       # lowercased, ftfy-repaired form
    ref_type: RefType
    confidence: float               # 0.0 – 1.0; reflects pattern match quality
    source_field: str               # "group_field" | "near_text" | "organization" | …


@dataclass
class NormalizedGroupSignals:
    """
    Per-booking output of signal_aggregator.py.

    Canonical labels are the result of alias lookup + noise-word stripping.
    None means the field was absent, blocklisted, or matched a section name.
    """
    booking_no: str
    canonical_org: Optional[str]         # None if absent, private, or blocklisted
    canonical_group_field: Optional[str] # None if absent or matched a section name
    group_field_is_section: bool         # True when group_field was a section name
    extracted_references: List[ExtractedReference]
    review_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Resolution (cross-booking name matching)
# ---------------------------------------------------------------------------

@dataclass
class ResolvedReference:
    """
    An ExtractedReference that has been looked up against the booking roster.

    When is_ambiguous is True (multiple bookings matched the name), no graph
    edge is created — the reference is emitted as an AmbiguousGroupCase instead.
    """
    extracted: ExtractedReference
    matched_booking_nos: List[str]  # len == 1 → hard link; len > 1 → ambiguous
    match_confidence: float
    is_ambiguous: bool              # True when len(matched_booking_nos) > 1


@dataclass
class AmbiguousGroupCase:
    """
    A name reference that could not be resolved to a single booking.

    Emitted as a review artifact; never drives cluster formation.
    """
    booking_no: str                 # the booking that made the reference
    reference_raw_text: str
    normalized_candidate: str
    matched_booking_nos: List[str]  # all candidates (empty = unresolved)
    reason: str                     # "multiple_surname_matches" | "unresolved" | …
    confidence: float
    source_field: str


@dataclass
class AliasMatchSuggestion:
    """
    Weak inferred label match — review suggestion only, never a graph edge.

    Produced when a free-text fragment resembles a known canonical label but
    does not hit an exact alias in group_aliases.yaml.
    """
    booking_no: str
    raw_text: str
    normalized_text: str
    candidate_canonical: Optional[str]   # best alias guess, or None
    confidence: float
    reason: str                          # "partial_alias_match" | "noise_stripped_match" | …
    source_field: str


# ---------------------------------------------------------------------------
# Graph edges
# ---------------------------------------------------------------------------

@dataclass
class AffinityEdge:
    """
    One weighted relationship in the affinity graph.

    node_a / node_b are either booking_no strings or canonical label strings.
    node_a_type / node_b_type tell you which.

    Only four edge types are allowed.  inferred_alias_match is NOT an edge type —
    it is represented by AliasMatchSuggestion above.
    """
    node_a: str
    node_b: str
    node_a_type: NodeType
    node_b_type: NodeType
    edge_type: EdgeType
    weight: float           # from group_rules.yaml link_strength
    source_field: str       # which field produced this evidence
    raw_text: str           # original text that triggered this edge
    normalized_text: str    # canonical form used for matching
    confidence: float       # edge-specific confidence (may differ from weight)


# ---------------------------------------------------------------------------
# Resolved clusters
# ---------------------------------------------------------------------------

@dataclass
class ResolvedCluster:
    """
    A group of bookings connected by affinity edges, resolved to a cluster.

    internal_edges contains all edges between members of this cluster (including
    edges via intermediate label nodes).  The edge list is the evidence chain.
    """
    cluster_id: str
    members: List[str]                      # booking_nos
    canonical_label: Optional[str]          # org/group label if one exists
    cluster_type: ClusterType
    min_link_strength: float
    max_link_strength: float
    internal_edges: List[AffinityEdge]
    review_flags: List[str] = field(default_factory=list)


@dataclass
class LargeWeakCluster:
    """
    A large org/church group whose members must NOT be forced into one tight
    allocation unit by the optimizer.

    Always carries "do_not_force_tight_cluster" in review_flags.
    Strong sub-pairs (explicit_family / explicit_named_people links) are
    pulled out into subclusters so the optimizer can still honour them.
    """
    cluster_id: str
    canonical_label: str
    all_member_booking_nos: List[str]       # flat list of all booking members
    subclusters: List[ResolvedCluster]      # strong sub-groups within the org
    unaffiliated_members: List[str]         # members with no strong sub-link
    review_flags: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if "do_not_force_tight_cluster" not in self.review_flags:
            self.review_flags.insert(0, "do_not_force_tight_cluster")
