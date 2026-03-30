"""Build the affinity graph between bookings.

Graph architecture
------------------
Two node types:
  booking nodes  — keyed by booking_no
  label nodes    — keyed by canonical org/group label string

Edges:
  booking ↔ label  : organization_membership or group_field_membership
                     (avoids N² edges for large church groups)
  booking ↔ booking: explicit_family or explicit_named_people
                     ONLY when a name reference resolves to exactly ONE other
                     booking with confidence ≥ min_name_resolution_confidence

"inferred_alias_match" is NOT an edge type.  It appears only in
AliasMatchSuggestion review objects.

NetworkX is used internally for connected-component analysis in group_resolver.
The public return values of this module are plain Python dataclasses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

from saronsdal.models.normalized import Booking
from saronsdal.models.grouping import (
    AffinityEdge,
    AliasMatchSuggestion,
    AmbiguousGroupCase,
    ExtractedReference,
    NormalizedGroupSignals,
    ResolvedReference,
)

logger = logging.getLogger(__name__)

_CONFIG_ROOT = Path(__file__).parent.parent / "config"


def _load_rules(rules_path: Optional[Path] = None) -> dict:
    p = rules_path or _CONFIG_ROOT / "group_rules.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Name resolution helpers
# ---------------------------------------------------------------------------

def _build_roster(bookings: List[Booking]) -> Dict[str, Booking]:
    """Return booking_no → Booking lookup."""
    return {b.booking_no: b for b in bookings}


def _last_name_index(bookings: List[Booking]) -> Dict[str, List[str]]:
    """Return lowercase last_name → list of booking_nos."""
    idx: Dict[str, List[str]] = {}
    for b in bookings:
        key = b.last_name.strip().lower()
        if key:
            idx.setdefault(key, []).append(b.booking_no)
    return idx


def _full_name_index(bookings: List[Booking]) -> Dict[str, List[str]]:
    """Return lowercase full_name → list of booking_nos."""
    idx: Dict[str, List[str]] = {}
    for b in bookings:
        key = b.full_name.strip().lower()
        if key:
            idx.setdefault(key, []).append(b.booking_no)
    return idx


def _resolve_reference(
    ref: ExtractedReference,
    source_booking_no: str,
    last_name_idx: Dict[str, List[str]],
    full_name_idx: Dict[str, List[str]],
    min_confidence: float,
) -> Tuple[Optional[ResolvedReference], Optional[AmbiguousGroupCase]]:
    """
    Try to match an ExtractedReference against the booking roster.

    Returns (ResolvedReference, None) on a clear match,
            (None, AmbiguousGroupCase) when multiple bookings match or resolution
            confidence is below threshold,
            (None, None) when no match is found.
    """
    candidate = ref.normalized_candidate

    if ref.ref_type == "family":
        # Surname-only lookup.
        matches = [
            bno for bno in last_name_idx.get(candidate, [])
            if bno != source_booking_no
        ]
        if not matches:
            return None, None  # Unresolved — no edge, no case
        if len(matches) > 1:
            return None, AmbiguousGroupCase(
                booking_no=source_booking_no,
                reference_raw_text=ref.raw_text,
                normalized_candidate=candidate,
                matched_booking_nos=matches,
                reason="multiple_surname_matches",
                confidence=ref.confidence,
                source_field=ref.source_field,
            )
        # Exactly one match.
        resolved_conf = ref.confidence * 0.90  # surname-only slightly less certain
        if resolved_conf < min_confidence:
            return None, AmbiguousGroupCase(
                booking_no=source_booking_no,
                reference_raw_text=ref.raw_text,
                normalized_candidate=candidate,
                matched_booking_nos=matches,
                reason="low_resolution_confidence",
                confidence=resolved_conf,
                source_field=ref.source_field,
            )
        return ResolvedReference(
            extracted=ref,
            matched_booking_nos=matches,
            match_confidence=resolved_conf,
            is_ambiguous=False,
        ), None

    if ref.ref_type == "full_name":
        # Full name lookup — try exact match first.
        matches = [
            bno for bno in full_name_idx.get(candidate, [])
            if bno != source_booking_no
        ]
        if matches:
            resolved_conf = ref.confidence * 0.95
            if resolved_conf < min_confidence:
                return None, None
            if len(matches) > 1:
                return None, AmbiguousGroupCase(
                    booking_no=source_booking_no,
                    reference_raw_text=ref.raw_text,
                    normalized_candidate=candidate,
                    matched_booking_nos=matches,
                    reason="multiple_full_name_matches",
                    confidence=resolved_conf,
                    source_field=ref.source_field,
                )
            return ResolvedReference(
                extracted=ref,
                matched_booking_nos=matches,
                match_confidence=resolved_conf,
                is_ambiguous=False,
            ), None

        # Full-name miss → fall back to last-name lookup at lower confidence.
        parts = candidate.rsplit(" ", 1)
        if len(parts) == 2:
            last = parts[1]
            last_matches = [
                bno for bno in last_name_idx.get(last, [])
                if bno != source_booking_no
            ]
            if len(last_matches) == 1:
                resolved_conf = ref.confidence * 0.65
                if resolved_conf >= min_confidence:
                    return ResolvedReference(
                        extracted=ref,
                        matched_booking_nos=last_matches,
                        match_confidence=resolved_conf,
                        is_ambiguous=False,
                    ), None
            elif len(last_matches) > 1:
                return None, AmbiguousGroupCase(
                    booking_no=source_booking_no,
                    reference_raw_text=ref.raw_text,
                    normalized_candidate=candidate,
                    matched_booking_nos=last_matches,
                    reason="full_name_fallback_surname_ambiguous",
                    confidence=ref.confidence * 0.65,
                    source_field=ref.source_field,
                )

    # first_name_only, organization, ambiguous — never produce booking edges.
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    signals: Dict[str, NormalizedGroupSignals],
    bookings: List[Booking],
    rules_path: Optional[Path] = None,
) -> Tuple[List[AffinityEdge], List[AmbiguousGroupCase], List[AliasMatchSuggestion]]:
    """
    Build affinity edges from aggregated group signals.

    Returns:
        edges           — all affinity edges (booking↔label and booking↔booking)
        ambiguous_cases — name references that could not be unambiguously resolved
        alias_suggestions — free-form labels that might match a known alias
    """
    rules = _load_rules(rules_path)
    strengths = rules.get("link_strength", {})
    min_res_conf = rules["thresholds"]["min_name_resolution_confidence"]
    min_edge_conf = rules["thresholds"]["min_edge_confidence"]

    last_name_idx = _last_name_index(bookings)
    full_name_idx = _full_name_index(bookings)

    edges: List[AffinityEdge] = []
    ambiguous_cases: List[AmbiguousGroupCase] = []
    alias_suggestions: List[AliasMatchSuggestion] = []

    # Track booking↔booking pairs to avoid duplicate directed edges.
    seen_bb_pairs: Set[frozenset] = set()

    for bno, sig in signals.items():

        # ---- org membership edge -------------------------------------------
        if sig.canonical_org:
            label_id = sig.canonical_org
            edge = AffinityEdge(
                node_a=bno,
                node_b=label_id,
                node_a_type="booking",
                node_b_type="label",
                edge_type="organization_membership",
                weight=strengths.get("organization_membership", 0.50),
                source_field="organization",
                raw_text=sig.canonical_org,
                normalized_text=sig.canonical_org.lower(),
                confidence=strengths.get("organization_membership", 0.50),
            )
            edges.append(edge)

        # ---- group_field membership edge -----------------------------------
        if sig.canonical_group_field:
            label_id = sig.canonical_group_field
            edge = AffinityEdge(
                node_a=bno,
                node_b=label_id,
                node_a_type="booking",
                node_b_type="label",
                edge_type="group_field_membership",
                weight=strengths.get("group_field_membership", 0.65),
                source_field="group_field",
                raw_text=sig.canonical_group_field,
                normalized_text=sig.canonical_group_field.lower(),
                confidence=strengths.get("group_field_membership", 0.65),
            )
            edges.append(edge)
            # If canonical_group_field differs from canonical_org but looks like
            # it could be an alias of an unrecognised org → AliasMatchSuggestion.
            if (sig.canonical_org is None and
                    sig.canonical_group_field not in [e.node_b for e in edges
                                                       if e.node_a == bno and
                                                          e.edge_type == "organization_membership"]):
                # Already emitted as a group_field edge; no additional suggestion needed.
                pass

        # ---- resolved name references (booking↔booking edges) -------------
        for ref in sig.extracted_references:
            if ref.ref_type not in ("family", "full_name"):
                continue  # first_name_only / org / ambiguous → no edge

            resolved, ambig = _resolve_reference(
                ref, bno, last_name_idx, full_name_idx, min_res_conf
            )

            if ambig is not None:
                ambiguous_cases.append(ambig)
                continue

            if resolved is None:
                continue

            if resolved.match_confidence < min_edge_conf:
                continue

            target_bno = resolved.matched_booking_nos[0]
            pair = frozenset([bno, target_bno])
            if pair in seen_bb_pairs:
                continue  # already have an edge for this pair
            seen_bb_pairs.add(pair)

            edge_type: str = (
                "explicit_family" if ref.ref_type == "family"
                else "explicit_named_people"
            )
            weight = strengths.get(edge_type, 0.85)

            edge = AffinityEdge(
                node_a=bno,
                node_b=target_bno,
                node_a_type="booking",
                node_b_type="booking",
                edge_type=edge_type,  # type: ignore[arg-type]
                weight=weight,
                source_field=ref.source_field,
                raw_text=ref.raw_text,
                normalized_text=ref.normalized_candidate,
                confidence=resolved.match_confidence,
            )
            edges.append(edge)
            logger.debug(
                "Edge %s ↔ %s  type=%s  conf=%.2f  raw=%r",
                bno, target_bno, edge_type, resolved.match_confidence, ref.raw_text,
            )

    logger.info(
        "Affinity graph: %d edges, %d ambiguous cases, %d alias suggestions",
        len(edges), len(ambiguous_cases), len(alias_suggestions),
    )
    return edges, ambiguous_cases, alias_suggestions
