"""Aggregate Phase 1 Booking fields into Phase 2 NormalizedGroupSignals.

Reads directly from canonical Phase 1 Booking fields:
  - booking.group_signals.organization
  - booking.group_signals.group_field
  - booking.group_signals.near_text_fragments
  - booking.request.raw_near_texts

Does NOT re-read raw CSV or re-run Phase 1 extraction logic.
Does NOT cross-reference bookings — that happens in affinity_graph.py.

Output: one NormalizedGroupSignals per Booking.
"""

from __future__ import annotations

from typing import Dict, List

from saronsdal.models.normalized import Booking
from saronsdal.models.grouping import ExtractedReference, NormalizedGroupSignals
from saronsdal.grouping.group_normalizer import GroupNormalizer
from saronsdal.grouping.name_reference_extractor import NameReferenceExtractor


def aggregate_signals(
    booking: Booking,
    normalizer: GroupNormalizer,
    extractor: NameReferenceExtractor,
) -> NormalizedGroupSignals:
    """
    Produce NormalizedGroupSignals for one booking.

    All text consumed here comes from the Phase 1 Booking object directly.
    """
    flags: List[str] = []
    refs: List[ExtractedReference] = []

    gs = booking.group_signals

    # --- Canonical org -------------------------------------------------------
    canonical_org = None
    if gs.organization and not gs.is_org_private:
        label = normalizer.normalize(gs.organization, source_field="organization")
        if label is not None:
            canonical_org = label.canonical
        # If label is None but org was present, it was blocklisted/section/private.

    # --- Canonical group_field -----------------------------------------------
    canonical_group_field = None
    group_field_is_section = False

    if gs.group_field:
        # Section guard: if Phase 1 somehow passed a section name through,
        # flag it and do not create a group node.
        if normalizer.is_section_name(gs.group_field):
            group_field_is_section = True
            flags.append("group_field_is_section")
        else:
            label = normalizer.normalize(gs.group_field, source_field="group_field")
            if label is not None:
                canonical_group_field = label.canonical
                # Also run name extractor on group_field — it may contain person names
                # in addition to or instead of an org label.
                person_refs = extractor.extract(gs.group_field, source_field="group_field")
                refs.extend(person_refs)
            else:
                # Normalizer rejected it (blocklisted/too short) — still try to extract
                # person/family references from the raw text.
                person_refs = extractor.extract(gs.group_field, source_field="group_field")
                refs.extend(person_refs)

    # --- Near-text fragments (from Phase 1 group_signals) --------------------
    for fragment in gs.near_text_fragments:
        fragment_refs = extractor.extract(fragment, source_field="near_text")
        refs.extend(fragment_refs)

    # --- Near-text fragments (from Phase 1 SpotRequest) ----------------------
    for fragment in booking.request.raw_near_texts:
        fragment_refs = extractor.extract(fragment, source_field="request_near_text")
        refs.extend(fragment_refs)

    # Deduplicate refs by (ref_type, normalized_candidate).
    seen_keys = set()
    deduped: List[ExtractedReference] = []
    for r in refs:
        key = (r.ref_type, r.normalized_candidate)
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(r)

    return NormalizedGroupSignals(
        booking_no=booking.booking_no,
        canonical_org=canonical_org,
        canonical_group_field=canonical_group_field,
        group_field_is_section=group_field_is_section,
        extracted_references=deduped,
        review_flags=flags,
    )


def aggregate_all(
    bookings: List[Booking],
    normalizer: GroupNormalizer,
    extractor: NameReferenceExtractor,
) -> Dict[str, NormalizedGroupSignals]:
    """
    Run aggregate_signals for every booking.

    Returns a dict keyed by booking_no for O(1) lookup during graph building.
    """
    return {
        b.booking_no: aggregate_signals(b, normalizer, extractor)
        for b in bookings
    }
