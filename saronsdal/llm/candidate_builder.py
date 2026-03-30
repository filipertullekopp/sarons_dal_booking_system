"""Phase 2.5 — Deterministic candidate selection for LLM enrichment.

Reads Phase 1 + Phase 2 outputs and selects high-value cases that the
deterministic pipeline could not fully resolve.

Five selectors (all signal-driven — never triggered by review_flags):

  1. recurring_unknown_group_phrases
       Scans all free-text fields for group-like phrases not already in
       group_aliases.yaml or sections.yaml.  Primary mechanism for
       discovering "Camp Jæren", "Varhaug gjengen", etc.

  2. unresolved_near_text_with_no_edges
       Bookings with raw_near_texts that produced zero booking↔booking
       affinity edges.  Covers "Tor A Ramsli", "Gudmund / Ingerid Haugsvær".

  3. weak_label_only_clusters
       ResolvedClusters whose type is org_group (only label edges, no
       direct booking↔booking links).  Gemini classifies each as
       place / misspelling / known group / new group.

  4. unstructured_preference_text
       Bookings whose free text contains enrichment signals not captured
       by Phase 1 — avoidance, same-as-last-year, extra space, positional
       nuance — even when a section was already extracted.

  5. subsection_detection_candidates
       Section extracted but text hints at additional row alternatives
       ("D eller E"), "felt A" patterns, or multi-row spot ranges not
       resolved to preferred_section_rows.

Pre-triage (deterministic, runs before Gemini):
  obvious_place             — exact section alias match
  obvious_place_misspelling — fuzzy match to section alias (ratio ≥ 0.82)
  obvious_known_group       — exact match in group_aliases.yaml aliases
  truly_ambiguous           — passes all guards → sent to Gemini
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import ftfy
import yaml

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_ROOT = Path(__file__).parent.parent / "config"


@dataclass
class CandidateConfig:
    """Tunable thresholds for candidate selection."""
    # Minimum frequency for a recurring group phrase to be selected.
    group_phrase_min_frequency: int = 2
    # Additional group signal tokens that qualify a singleton phrase for selection.
    singleton_boost_tokens: FrozenSet[str] = frozenset({
        "camp", "pinsemenigheten", "menigheten", "kirken",
        "betania", "betel", "pinsekirken", "filadelfia",
    })
    # SequenceMatcher ratio threshold for "obvious_place_misspelling" detection.
    misspelling_threshold: float = 0.82
    # Minimum raw-text length (chars) for preference candidates.
    pref_text_min_length: int = 20


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GroupPhraseCandidate:
    """A recurring or high-signal group phrase that was not already known."""
    phrase: str                     # normalized (lowercase) form
    raw_variants: List[str]         # distinct observed spellings
    frequency: int                  # booking count
    booking_nos: List[str]
    source_fields: List[str]        # which fields it appeared in
    is_singleton_high_signal: bool  # True when freq=1 but has a boost token
    pretriage_bucket: str           # "truly_ambiguous" | "obvious_*"


@dataclass
class NearTextCandidate:
    """Booking with raw near-text fragments that produced zero affinity edges."""
    booking_no: str
    full_name: str
    raw_near_texts: List[str]
    check_in: Optional[str]
    check_out: Optional[str]


@dataclass
class WeakClusterCandidate:
    """An org_group cluster with only label edges — possibly a place or misspelling."""
    cluster_id: str
    canonical_label: str
    member_booking_nos: List[str]
    pretriage_bucket: str   # "truly_ambiguous" | "obvious_place" | "obvious_place_misspelling"


@dataclass
class PreferenceCandidate:
    """Booking whose free text contains enrichment signals Phase 1 didn't capture."""
    booking_no: str
    full_name: str
    raw_text: str           # combined text that triggered selection
    source_fields: List[str]
    extracted_sections: List[str]   # what Phase 1 already found
    missing_signals: List[str]      # signal names detected but not in structured output


@dataclass
class SubsectionCandidate:
    """Section was extracted but text has additional row/subsection nuance."""
    booking_no: str
    full_name: str
    raw_text: str
    extracted_section: str
    already_captured_rows: List[str]    # rows already in preferred_section_rows
    unresolved_patterns: List[str]      # patterns found that weren't captured


@dataclass
class PreTriageSummary:
    """Counts from the group phrase pre-triage step."""
    obvious_place: int = 0
    obvious_place_misspelling: int = 0
    obvious_known_group: int = 0
    truly_ambiguous: int = 0
    total_phrases_scanned: int = 0


@dataclass
class CandidateSet:
    """All candidates produced by build_candidates()."""
    group_phrases: List[GroupPhraseCandidate] = field(default_factory=list)
    near_text: List[NearTextCandidate] = field(default_factory=list)
    weak_clusters: List[WeakClusterCandidate] = field(default_factory=list)
    preferences: List[PreferenceCandidate] = field(default_factory=list)
    subsections: List[SubsectionCandidate] = field(default_factory=list)
    pretriage_summary: PreTriageSummary = field(default_factory=PreTriageSummary)

    @property
    def total(self) -> int:
        return (len(self.group_phrases) + len(self.near_text)
                + len(self.weak_clusters) + len(self.preferences)
                + len(self.subsections))


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_section_aliases(sections_path: Optional[Path] = None) -> Set[str]:
    """Return all section aliases (lowercased, ftfy-repaired)."""
    p = sections_path or _CONFIG_ROOT / "sections.yaml"
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    aliases: Set[str] = set()
    for entry in cfg.get("sections", {}).values():
        aliases.add(ftfy.fix_text(entry["canonical"]).lower())
        for alias in entry.get("aliases", []):
            aliases.add(ftfy.fix_text(alias).lower())
    for s in cfg.get("section_name_set", []):
        aliases.add(ftfy.fix_text(s).lower())
    return aliases


def _load_group_alias_normalized(aliases_path: Optional[Path] = None) -> Set[str]:
    """Return all group alias strings (lowercased, ftfy-repaired)."""
    p = aliases_path or _CONFIG_ROOT / "group_aliases.yaml"
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    known: Set[str] = set()
    for entry in cfg.get("organizations", {}).values():
        known.add(ftfy.fix_text(entry["canonical"]).lower())
        for alias in entry.get("aliases", []):
            known.add(ftfy.fix_text(alias).lower())
    return known


# ---------------------------------------------------------------------------
# Pre-triage
# ---------------------------------------------------------------------------

def _pretriage_phrase(
    phrase: str,
    section_aliases: Set[str],
    group_alias_normalized: Set[str],
    misspelling_threshold: float,
) -> str:
    """
    Classify a normalized phrase deterministically before sending to Gemini.

    Returns one of:
        "obvious_place"             — exact section alias
        "obvious_place_misspelling" — fuzzy match to section alias
        "obvious_known_group"       — in group_aliases.yaml
        "truly_ambiguous"           — needs Gemini
    """
    p = phrase.lower().strip()

    if p in section_aliases:
        return "obvious_place"

    if p in group_alias_normalized:
        return "obvious_known_group"

    # Fuzzy match against each section alias — catches "fululunden", "elvebreden".
    for alias in section_aliases:
        ratio = SequenceMatcher(None, p, alias).ratio()
        if ratio >= misspelling_threshold:
            return "obvious_place_misspelling"

    return "truly_ambiguous"


# ---------------------------------------------------------------------------
# Group phrase extraction patterns
# ---------------------------------------------------------------------------

# Capitalized multi-word phrase (each token ≥ 3 chars, at least 2 tokens).
# Negative lookahead (?!\s+[A-Z]\d) prevents absorbing person names that are
# followed by a spot ID ("Alfred Antonsen D28-D30" → stop before "D28").
_MULTI_CAP_RE = re.compile(
    r'\b([A-ZÆØÅ][a-zæøå]{2,}(?:\s+[A-ZÆØÅ][a-zæøå]{2,})+)(?!\s+[A-Z]\d)\b',
    re.UNICODE,
)

# "X gjengen/gruppa/laget/kretsen" — e.g. "Varhaug gjengen"
# No IGNORECASE: [A-ZÆØÅ] must match uppercase only, so "fra Varhaug gjengen"
# won't absorb "fra" into the proper-noun group.
_GROUP_SUFFIX_RE = re.compile(
    r'\b([A-ZÆØÅ][a-zæøå]{2,}(?:\s+[A-ZÆØÅ][a-zæøå]{2,})?)'
    r'\s+(?:gjengen|gruppa|laget|kretsen|fellesskapet)\b',
    re.UNICODE,
)

# "camp/fra X" — e.g. "Camp Jæren", "folk fra Jæren"
# Uses (?i:...) inline flags so only the keyword is case-insensitive while the
# proper-noun capture group stays uppercase-anchored.
_GROUP_FROM_OR_CAMP_RE = re.compile(
    r'\b(?i:camp)\s+([A-ZÆØÅ][a-zæøå]{2,}(?:\s+[A-ZÆØÅ][a-zæøå]{2,})?)\b'
    r'|\b(?i:gjengen|andre|folk)\s+fra\s+([A-ZÆØÅ][a-zæøå]{2,}(?:\s+[A-ZÆØÅ][a-zæøå]{2,})?)\b',
    re.UNICODE,
)

# Single proper noun (≥ 5 chars) followed by a standalone row letter (not a spot ID).
# Catches "Furulunden B", "Elvebredden C" but NOT "Furulunden B25", "Camp Jæren",
# or short common words like "Felt B".
_SINGLE_NOUN_BEFORE_LETTER_RE = re.compile(
    r'\b([A-ZÆØÅ][a-zæøå]{4,})\s+[A-Z]\b',
    re.UNICODE,
)

# Common Norwegian surname suffixes used to detect person-name phrases.
_SURNAME_SUFFIX_RE = re.compile(
    r'(?:sen|ssen|son|bø|haug|vik|svik|rud|land|lund|berg|stad|heim|strand|nes|ås|gård|gard)$',
    re.IGNORECASE | re.UNICODE,
)

# Formal church phrases starting with Pinsemenigheten / Menigheten / etc.
_CHURCH_RE = re.compile(
    r'\b(?:Pinsemenigheten|Menigheten|Pinsekirken|Filadelfiakirken|Betania|Betel)'
    r'\s+[A-ZÆØÅ][a-zæøå]{2,}(?:\s+[A-ZÆØÅ][a-zæøå]{2,})?\b',
    re.UNICODE,
)

_GROUP_SIGNAL_TOKENS: FrozenSet[str] = frozenset({
    "camp", "pinsemenigheten", "menigheten", "kirken",
    "betania", "betel", "pinsekirken", "filadelfia", "gjengen",
})

# Phrases that START with a placement-preference verb are preference language,
# not social-group labels ("foretrekker Vårdalen" → preference, not group).
_PREF_VERB_RE = re.compile(
    r'^(?:foretrekker|ønsker|ønske|vil\s+bo|helst|gjerne|vil\s+ha|'
    r'har\s+lyst|ønsker\s+å|foretrekkes|foretrekker\s+å)\b',
    re.IGNORECASE | re.UNICODE,
)


def _extract_group_phrases(text: str) -> List[Tuple[str, str]]:
    """
    Extract candidate group phrases from one text string.

    Returns list of (raw_phrase, normalized_phrase) tuples.
    """
    text = ftfy.fix_text(text)
    results: List[Tuple[str, str]] = []

    def _add(raw: str) -> None:
        raw = raw.strip()
        if len(raw) < 4:
            return
        norm = raw.lower()
        results.append((raw, norm))

    for m in _MULTI_CAP_RE.finditer(text):
        _add(m.group(1))

    for m in _GROUP_SUFFIX_RE.finditer(text):
        # Include the suffix token in the phrase.
        _add(m.group(0))

    for m in _GROUP_FROM_OR_CAMP_RE.finditer(text):
        captured = m.group(1) or m.group(2)
        if captured:
            # Include the prefix word ("camp X", "folk fra X" → keep full match).
            _add(m.group(0))

    for m in _CHURCH_RE.finditer(text):
        _add(m.group(0))

    # Single proper noun before a standalone row letter — catches "Furulunden B".
    for m in _SINGLE_NOUN_BEFORE_LETTER_RE.finditer(text):
        _add(m.group(1))

    return results


def _is_likely_person_name(phrase: str) -> bool:
    """Return True when a phrase looks like a person name rather than a group label.

    Uses Norwegian surname suffixes as a signal.  Only applies to 2–4 word
    phrases so short single-word org names and long church names are not
    accidentally flagged.
    """
    words = phrase.strip().split()
    if not (2 <= len(words) <= 4):
        return False
    return bool(_SURNAME_SUFFIX_RE.search(words[-1]))


def _has_boost_token(phrase: str, boost_tokens: FrozenSet[str]) -> bool:
    """Return True if phrase contains any singleton-boost token."""
    words = set(phrase.lower().split())
    return bool(words & boost_tokens)


# ---------------------------------------------------------------------------
# Enrichment signal patterns (preference selector)
# ---------------------------------------------------------------------------

# Each entry: (signal_name, compiled_pattern).
# All signals are "always missing" when they fire (Phase 1 doesn't capture
# any of them), except spot_range_hint which is skipped when spot_ids are
# already populated.
_ENRICH_SIGNALS: List[Tuple[str, re.Pattern]] = [
    # River avoidance — Phase 1 captures sections but not avoidance intent
    ("avoid_river",       re.compile(r"\bvekke\s+fra\s+(?:elv\w*|vann\w*)\b", re.I)),
    ("avoid_river",       re.compile(r"\blengst\s+vekke\b", re.I)),
    ("avoid_river",       re.compile(r"\bikke\s+(?:ved|nær)\s+elv\w*\b", re.I)),
    ("avoid_river",       re.compile(r"\bpå\s+andre\s+siden\s+av\s+elv\w*\b", re.I)),
    ("avoid_noise",       re.compile(r"\bvekke\s+fra\s+(?:støy|bråk|folk)\b", re.I)),
    # Historical placement — many variants in real data
    ("same_as_last_year", re.compile(r"\bsamme\s+(?:plass|plassering|sted)?\s*som\s+(?:vi\s+(?:har\s+hatt|hadde)|i\s+fjor)\b", re.I)),
    ("same_as_last_year", re.compile(r"\bi\s+fjor\b", re.I)),
    ("same_as_last_year", re.compile(r"\bde\s+siste\s+\d+\s+år\w*\b", re.I)),
    ("same_as_last_year", re.compile(r"\bhatt\s+(?:det|plass\w*)\s+i\s+fjor\b", re.I)),
    ("same_as_last_year", re.compile(r"\bsiste\s+\d*\s*år\b", re.I)),
    ("same_as_last_year", re.compile(r"\btidligere\s+år\b", re.I)),
    ("same_as_last_year", re.compile(r"\bsame\s+(?:spot|place)\s+as\s+last\s+year\b", re.I)),
    # Extra space / large vehicles
    ("extra_space",       re.compile(r"\bekstra\s+plass\b", re.I)),
    ("extra_space",       re.compile(r"\bbehov\s+for\s+ekstra\b", re.I)),
    ("extra_space",       re.compile(r"\bplass\s+til\s+\d+\s*biler?\b", re.I)),
    ("extra_space",       re.compile(r"\blastebil\b", re.I)),
    ("extra_space",       re.compile(r"\banneks\b", re.I)),
    ("extra_space",       re.compile(r"\bekstra\s+anneks\b|\bbredt\s+fortelt\b", re.I)),
    # Positional nuance around bibelskolen / main hall
    ("near_bibelskolen",  re.compile(r"\bbibelskolen\b", re.I)),
    ("near_hall",         re.compile(r"\b(?:nær|ved|nærmere?|nærme|nærmest)\s+hallen?\b", re.I)),
    # Terrain / ground conditions
    ("flat_ground",       re.compile(r"\bflat\s+plass\b|\bflatt?\s+underlag\b", re.I)),
    ("terrain_pref",      re.compile(r"\bnedoverbakke\b|\bi\s+høyden\b|\bpå\s+skrå\b", re.I)),
    # Amenities (Phase 1 doesn't extract these)
    ("near_toilet",       re.compile(r"\btoalett\b|\bwc\b", re.I)),
    # Drainage / water management
    ("drainage_issue",    re.compile(r"\bdrenering\b|\bbekk\s+igjennom\b", re.I)),
    # Forest / natural edge preference
    ("near_forest",       re.compile(r"\b(?:langs|nær|inn\s+mot)\s+skogen?\b", re.I)),
    ("near_forest",       re.compile(r"\bskogen?\s+som\s+naboer?\b", re.I)),
    # Quiet / calm spot
    ("quiet_spot",        re.compile(r"\brolig\s+plass\b|\bstille\s+(?:plass|område)\b", re.I)),
    # Accessibility needs
    ("accessibility",     re.compile(r"\bbegrenset\s+førlighet\b|\brullator\b|\brullestol\b", re.I)),
    # Spot range hint — fire only when Phase 1 didn't extract spot_ids
    ("spot_range_hint",   re.compile(r"\b[A-Za-z]?\d{1,3}\s*[-–]\s*[A-Za-z]?\d{1,3}\b")),
    ("spot_range_hint",   re.compile(r"\b[A-Za-z]\d{1,3}\s+til\s+\d{1,3}\b", re.I)),
]


def _find_missing_enrichment_signals(booking: dict) -> Tuple[List[str], str]:
    """
    Return (list_of_missing_signal_names, combined_text_that_triggered).

    All signals are treated as "not captured" by Phase 1 (it only extracts
    sections / spot IDs, not preference nuance).  spot_range_hint is the
    only exception: skipped when preferred_spot_ids is already populated.
    """
    raw_text = " ".join(filter(None, [
        booking.get("raw_guest_message", ""),
        booking.get("raw_comment", ""),
        booking.get("raw_location_wish", ""),
    ]))
    if not raw_text.strip():
        return [], ""

    spot_ids: List[str] = booking.get("request", {}).get("preferred_spot_ids", [])

    missing: List[str] = []
    for signal_name, pattern in _ENRICH_SIGNALS:
        if not pattern.search(raw_text):
            continue
        if signal_name in missing:
            continue  # deduplicate same signal matched by multiple patterns
        if signal_name == "spot_range_hint" and spot_ids:
            continue  # spot IDs already captured in structured output
        missing.append(signal_name)

    return missing, raw_text


# ---------------------------------------------------------------------------
# Subsection detection patterns
# ---------------------------------------------------------------------------

# "D eller E", "D eller E og F" — multiple row alternatives
_ROW_ALTERNATIVES_RE = re.compile(
    r'\b([A-Z])\s+(?:eller|evt\.?|alternativt|or)\s+([A-Z])\b',
    re.IGNORECASE,
)

# "felt A", "felt B-C"
_FELT_RE = re.compile(r'\bfelt\s+([A-Z])\b', re.IGNORECASE)

# Section alias followed by multiple candidate rows in one fragment.
# Catches "Vårdalen D eller E" as two rows when only one was captured.
_MULTI_ROW_HINT_RE = re.compile(
    r'\b[A-Z]\s+(?:eller|evt\.?|alternativt)\s+[A-Z]\b',
    re.IGNORECASE,
)

# Spot range like "A15-A18", "E23-25", "e07-09" — letter+number hyphen number.
_SPOT_RANGE_RE = re.compile(
    r'\b([A-Za-z]\d{1,3})\s*[-–]\s*([A-Za-z]?\d{1,3})\b',
)

# "E01 til 12" — letter+number "til" number
_SPOT_RANGE_TIL_RE = re.compile(
    r'\b([A-Za-z]\d{1,3})\s+til\s+(\d{1,3})\b',
    re.IGNORECASE,
)


def _find_subsection_patterns(booking: dict) -> Tuple[List[str], str]:
    """
    Return (list_of_unresolved_pattern_descriptions, combined_raw_text).

    Called only when preferred_sections is non-empty.
    """
    raw_text = " ".join(filter(None, [
        booking.get("raw_location_wish", ""),
        booking.get("raw_guest_message", ""),
        booking.get("raw_comment", ""),
    ]))
    if not raw_text.strip():
        return [], ""

    request = booking.get("request", {})
    captured_rows = {r["row"] for r in request.get("preferred_section_rows", [])}

    patterns_found: List[str] = []

    # Multiple row alternatives not in captured_rows.
    for m in _ROW_ALTERNATIVES_RE.finditer(raw_text):
        r1, r2 = m.group(1).upper(), m.group(2).upper()
        if r2 not in captured_rows:
            desc = f"row_alternatives:{r1}/{r2}"
            if desc not in patterns_found:
                patterns_found.append(desc)

    # "felt A" — row via "felt" keyword.
    for m in _FELT_RE.finditer(raw_text):
        row = m.group(1).upper()
        if row not in captured_rows:
            desc = f"felt_row:{row}"
            if desc not in patterns_found:
                patterns_found.append(desc)

    # Spot ranges like "A15-A18", "E23-25", "e07-09".
    for m in _SPOT_RANGE_RE.finditer(raw_text):
        desc = f"spot_range:{m.group(1).upper()}-{m.group(2).upper()}"
        if desc not in patterns_found:
            patterns_found.append(desc)

    # "E01 til 12" style ranges.
    for m in _SPOT_RANGE_TIL_RE.finditer(raw_text):
        desc = f"spot_range:{m.group(1).upper()}_til_{m.group(2)}"
        if desc not in patterns_found:
            patterns_found.append(desc)

    return patterns_found, raw_text


# ---------------------------------------------------------------------------
# Text source extraction
# ---------------------------------------------------------------------------

def _get_text_sources(booking: dict) -> List[Tuple[str, str]]:
    """Return list of (field_name, text) for all group-signal-relevant fields."""
    sources = []
    gs = booking.get("group_signals", {})
    req = booking.get("request", {})

    for field_name, text in [
        ("organization",    gs.get("organization") or ""),
        ("group_field",     gs.get("group_field") or ""),
        ("guest_message",   booking.get("raw_guest_message", "")),
        ("comment",         booking.get("raw_comment", "")),
        ("location_wish",   booking.get("raw_location_wish", "")),
    ]:
        if text.strip():
            sources.append((field_name, text))

    # near_text_fragments from group_signals
    for frag in gs.get("near_text_fragments", []):
        if frag.strip():
            sources.append(("near_text_fragment", frag))

    # raw_near_texts from request
    for frag in req.get("raw_near_texts", []):
        if frag.strip():
            sources.append(("request_near_text", frag))

    return sources


# ---------------------------------------------------------------------------
# Group-phrase source helpers (person-name exclusion)
# ---------------------------------------------------------------------------

def _get_group_phrase_sources(booking: dict) -> List[Tuple[str, str]]:
    """Like _get_text_sources but excludes near-text fields (person-reference fields).

    near_text_fragments and request_near_texts contain entries like
    "Alfred Antonsen D28-D30" and "Tor A Ramsli" — these feed Selector 2,
    not Selector 1.
    """
    return [
        (name, text)
        for name, text in _get_text_sources(booking)
        if name not in ("near_text_fragment", "request_near_text")
    ]


def _build_person_refs(bookings: List[dict]) -> Set[str]:
    """Return a set of normalized phrases found in near-text fragments.

    Near-text fragments are person-reference fields ("next to Alfred Antonsen",
    "ved Kari Olsen").  Multi-word capitalized phrases extracted from them are
    likely person names and should be excluded from group-label candidates.
    """
    refs: Set[str] = set()
    for booking in bookings:
        gs = booking.get("group_signals", {})
        req = booking.get("request", {})
        near_texts = (
            gs.get("near_text_fragments", [])
            + req.get("raw_near_texts", [])
        )
        for frag in near_texts:
            if not frag.strip():
                continue
            text = ftfy.fix_text(frag)
            for m in _MULTI_CAP_RE.finditer(text):
                refs.add(m.group(1).lower())
    return refs


def _build_org_phrases(bookings: List[dict]) -> Set[str]:
    """Return normalized organization/group_field values.

    These are explicitly set structured fields — even if they match
    person-name heuristics they must never be excluded (e.g. "Klippen Sandnes",
    "Betania Grimstad" both end in place-name-style words but are known groups).
    """
    org: Set[str] = set()
    for booking in bookings:
        gs = booking.get("group_signals", {})
        for val in (gs.get("organization") or "", gs.get("group_field") or ""):
            val = val.strip()
            if val:
                org.add(ftfy.fix_text(val).lower())
    return org


# ---------------------------------------------------------------------------
# Selector 1 — recurring_unknown_group_phrases
# ---------------------------------------------------------------------------

def _select_group_phrases(
    bookings: List[dict],
    section_aliases: Set[str],
    group_alias_normalized: Set[str],
    config: CandidateConfig,
) -> Tuple[List[GroupPhraseCandidate], PreTriageSummary]:
    """
    Extract, count, pre-triage, and select group-like phrases.
    Returns (candidates, pretriage_summary).
    """
    person_refs = _build_person_refs(bookings)
    org_phrases = _build_org_phrases(bookings)

    # phrase_norm → {raw_variants, booking_nos, source_fields}
    phrase_data: Dict[str, dict] = defaultdict(
        lambda: {"raw_variants": set(), "booking_nos": [], "source_fields": set()}
    )

    for booking in bookings:
        bno = booking.get("booking_no", "?")
        for field_name, text in _get_group_phrase_sources(booking):
            # For structured org/group fields, always include the whole value as
            # a candidate — even a single word like "Fululunden" (misspelling).
            if field_name in ("organization", "group_field"):
                raw_whole = ftfy.fix_text(text.strip())
                if len(raw_whole) >= 4:
                    norm_whole = raw_whole.lower()
                    phrase_data[norm_whole]["raw_variants"].add(raw_whole)
                    if bno not in phrase_data[norm_whole]["booking_nos"]:
                        phrase_data[norm_whole]["booking_nos"].append(bno)
                    phrase_data[norm_whole]["source_fields"].add(field_name)
            for raw_phrase, norm_phrase in _extract_group_phrases(text):
                phrase_data[norm_phrase]["raw_variants"].add(raw_phrase)
                if bno not in phrase_data[norm_phrase]["booking_nos"]:
                    phrase_data[norm_phrase]["booking_nos"].append(bno)
                phrase_data[norm_phrase]["source_fields"].add(field_name)

    summary = PreTriageSummary(total_phrases_scanned=len(phrase_data))
    candidates: List[GroupPhraseCandidate] = []

    for norm_phrase, data in sorted(phrase_data.items()):
        # Exclude likely person names unless they come from a structured
        # org/group_field or contain an explicit group signal token.
        is_person = norm_phrase in person_refs or _is_likely_person_name(norm_phrase)
        is_protected = norm_phrase in org_phrases or bool(
            set(norm_phrase.split()) & _GROUP_SIGNAL_TOKENS
        )
        if is_person and not is_protected:
            continue  # route to near-text selector, not group-phrase selector

        if _PREF_VERB_RE.search(norm_phrase):
            continue  # preference language — not a group label

        freq = len(data["booking_nos"])
        bucket = _pretriage_phrase(
            norm_phrase, section_aliases, group_alias_normalized,
            config.misspelling_threshold,
        )

        if bucket == "obvious_place":
            summary.obvious_place += 1
            continue
        if bucket == "obvious_place_misspelling":
            summary.obvious_place_misspelling += 1
            continue
        if bucket == "obvious_known_group":
            summary.obvious_known_group += 1
            continue

        # truly_ambiguous
        summary.truly_ambiguous += 1
        is_singleton_boost = (
            freq == 1
            and _has_boost_token(norm_phrase, config.singleton_boost_tokens)
        )
        if freq >= config.group_phrase_min_frequency or is_singleton_boost:
            candidates.append(GroupPhraseCandidate(
                phrase=norm_phrase,
                raw_variants=sorted(data["raw_variants"]),
                frequency=freq,
                booking_nos=data["booking_nos"],
                source_fields=sorted(data["source_fields"]),
                is_singleton_high_signal=is_singleton_boost,
                pretriage_bucket="truly_ambiguous",
            ))

    return candidates, summary


# ---------------------------------------------------------------------------
# Selector 2 — unresolved_near_text_with_no_edges
# ---------------------------------------------------------------------------

def _bookings_with_bb_edges(edges: List[dict]) -> Set[str]:
    """Return booking_nos that appear in at least one booking↔booking edge."""
    result: Set[str] = set()
    for e in edges:
        if e.get("node_a_type") == "booking" and e.get("node_b_type") == "booking":
            result.add(e["node_a"])
            result.add(e["node_b"])
    return result


def _select_near_text_no_edges(
    bookings: List[dict],
    bb_edge_booking_nos: Set[str],
) -> List[NearTextCandidate]:
    results = []
    for booking in bookings:
        bno = booking.get("booking_no", "")
        req = booking.get("request", {})
        gs = booking.get("group_signals", {})

        near_texts = (
            req.get("raw_near_texts", [])
            + gs.get("near_text_fragments", [])
        )
        near_texts = [t for t in near_texts if t.strip()]

        if near_texts and bno not in bb_edge_booking_nos:
            results.append(NearTextCandidate(
                booking_no=bno,
                full_name=booking.get("full_name", ""),
                raw_near_texts=near_texts,
                check_in=booking.get("check_in"),
                check_out=booking.get("check_out"),
            ))
    return results


# ---------------------------------------------------------------------------
# Selector 3 — weak_label_only_clusters
# ---------------------------------------------------------------------------

def _select_weak_clusters(
    clusters: List[dict],
    section_aliases: Set[str],
    group_alias_normalized: Set[str],
    config: CandidateConfig,
) -> List[WeakClusterCandidate]:
    results = []
    for c in clusters:
        if c.get("cluster_type") != "org_group":
            continue
        label = (c.get("canonical_label") or "").strip()
        if not label:
            continue
        bucket = _pretriage_phrase(
            label.lower(), section_aliases, group_alias_normalized,
            config.misspelling_threshold,
        )
        # Discard already-known groups — they don't need Gemini.
        if bucket == "obvious_known_group":
            continue
        results.append(WeakClusterCandidate(
            cluster_id=c.get("cluster_id", ""),
            canonical_label=label,
            member_booking_nos=c.get("members", []),
            pretriage_bucket=bucket,
        ))
    return results


# ---------------------------------------------------------------------------
# Selector 4 — unstructured_preference_text
# ---------------------------------------------------------------------------

def _select_preference_candidates(
    bookings: List[dict],
    config: CandidateConfig,
) -> List[PreferenceCandidate]:
    results = []
    for booking in bookings:
        missing_signals, raw_text = _find_missing_enrichment_signals(booking)
        if not missing_signals:
            continue
        if len(raw_text.strip()) < config.pref_text_min_length:
            continue
        req = booking.get("request", {})
        results.append(PreferenceCandidate(
            booking_no=booking.get("booking_no", ""),
            full_name=booking.get("full_name", ""),
            raw_text=raw_text,
            source_fields=["guest_message", "comment", "location_wish"],
            extracted_sections=req.get("preferred_sections", []),
            missing_signals=missing_signals,
        ))
    return results


# ---------------------------------------------------------------------------
# Selector 5 — subsection_detection_candidates
# ---------------------------------------------------------------------------

def _select_subsection_candidates(bookings: List[dict]) -> List[SubsectionCandidate]:
    results = []
    for booking in bookings:
        req = booking.get("request", {})
        sections = req.get("preferred_sections", [])
        if not sections:
            continue  # No section extracted — nothing to add a row to.

        patterns_found, raw_text = _find_subsection_patterns(booking)
        if not patterns_found:
            continue

        captured_rows = [r["row"] for r in req.get("preferred_section_rows", [])]
        results.append(SubsectionCandidate(
            booking_no=booking.get("booking_no", ""),
            full_name=booking.get("full_name", ""),
            raw_text=raw_text,
            extracted_section=sections[0],
            already_captured_rows=captured_rows,
            unresolved_patterns=patterns_found,
        ))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_candidates(
    bookings: List[dict],
    clusters: List[dict],
    large_clusters: List[dict],
    edges: List[dict],
    config: Optional[CandidateConfig] = None,
    *,
    aliases_path: Optional[Path] = None,
    sections_path: Optional[Path] = None,
) -> CandidateSet:
    """
    Build a CandidateSet from already-loaded Phase 1 + Phase 2 data.

    Parameters
    ----------
    bookings      : list of booking dicts (from bookings_normalized.json)
    clusters      : list of cluster dicts (from resolved_groups.json)
    large_clusters: list of large_cluster dicts (from resolved_groups.json)
    edges         : list of edge dicts (from group_links.jsonl)
    config        : threshold overrides; uses defaults when None
    aliases_path  : override path for group_aliases.yaml (for testing)
    sections_path : override path for sections.yaml (for testing)
    """
    if config is None:
        config = CandidateConfig()

    section_aliases = _load_section_aliases(sections_path)
    group_alias_normalized = _load_group_alias_normalized(aliases_path)

    # Selector 1
    group_candidates, pretriage_summary = _select_group_phrases(
        bookings, section_aliases, group_alias_normalized, config
    )

    # Selector 2
    bb_edge_bnos = _bookings_with_bb_edges(edges)
    near_text_candidates = _select_near_text_no_edges(bookings, bb_edge_bnos)

    # Selector 3
    weak_cluster_candidates = _select_weak_clusters(
        clusters, section_aliases, group_alias_normalized, config
    )

    # Selector 4
    pref_candidates = _select_preference_candidates(bookings, config)

    # Selector 5
    subsection_candidates = _select_subsection_candidates(bookings)

    return CandidateSet(
        group_phrases=group_candidates,
        near_text=near_text_candidates,
        weak_clusters=weak_cluster_candidates,
        preferences=pref_candidates,
        subsections=subsection_candidates,
        pretriage_summary=pretriage_summary,
    )


def build_candidates_from_files(
    phase1_dir: Path,
    phase2_dir: Path,
    config: Optional[CandidateConfig] = None,
) -> CandidateSet:
    """Load Phase 1 + Phase 2 output files, then call build_candidates()."""
    bookings_path = phase1_dir / "bookings_normalized.json"
    with open(bookings_path, encoding="utf-8") as fh:
        bookings: List[dict] = json.load(fh)

    resolved_path = phase2_dir / "resolved_groups.json"
    with open(resolved_path, encoding="utf-8") as fh:
        resolved = json.load(fh)
    clusters: List[dict] = resolved.get("clusters", [])
    large_clusters: List[dict] = resolved.get("large_clusters", [])

    edges: List[dict] = []
    links_path = phase2_dir / "group_links.jsonl"
    if links_path.exists():
        with open(links_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    edges.append(json.loads(line))

    return build_candidates(bookings, clusters, large_clusters, edges, config)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_candidate_summary(cs: CandidateSet, top_n: int = 10) -> None:
    """Print a human-readable candidate selection summary."""
    pt = cs.pretriage_summary
    print("\n=== Phase 2.5 Candidate Selection ===")
    print(f"  recurring_unknown_group_phrases : {len(cs.group_phrases)}"
          f"  ({pt.total_phrases_scanned} scanned, "
          f"{pt.obvious_place + pt.obvious_place_misspelling + pt.obvious_known_group} filtered)")
    print(f"  unresolved_near_text            : {len(cs.near_text)}")
    print(f"  weak_label_clusters             : {len(cs.weak_clusters)}")
    print(f"  unstructured_preferences        : {len(cs.preferences)}")
    print(f"  subsection_candidates           : {len(cs.subsections)}")
    print(f"  -----------------------------------------")
    print(f"  Total candidates for Gemini     : {cs.total}")

    print(f"\nPre-triage breakdown (group phrases):")
    print(f"  obvious_place             : {pt.obvious_place}")
    print(f"  obvious_place_misspelling : {pt.obvious_place_misspelling}")
    print(f"  obvious_known_group       : {pt.obvious_known_group}")
    print(f"  truly_ambiguous           : {pt.truly_ambiguous}  <- sent to Gemini")

    if cs.group_phrases:
        sorted_phrases = sorted(
            cs.group_phrases, key=lambda c: (-c.frequency, c.phrase)
        )
        print(f"\nTop group phrases (truly ambiguous, showing up to {top_n}):")
        for c in sorted_phrases[:top_n]:
            tag = " [singleton+boost]" if c.is_singleton_high_signal else ""
            sources = ", ".join(sorted(set(c.source_fields)))
            print(f"  {c.phrase!r:<40} freq={c.frequency}  src={sources}{tag}")

    if cs.weak_clusters:
        print(f"\nWeak label-only clusters:")
        for wc in cs.weak_clusters[:top_n]:
            print(f"  [{wc.pretriage_bucket}] {wc.canonical_label!r}"
                  f" ({len(wc.member_booking_nos)} members)")

    if cs.preferences:
        print(f"\nPreference candidates (top {min(top_n, len(cs.preferences))}):")
        for pc in cs.preferences[:top_n]:
            print(f"  {pc.booking_no} {pc.full_name!r}  signals={pc.missing_signals}")

    if cs.subsections:
        print(f"\nSubsection candidates (top {min(top_n, len(cs.subsections))}):")
        for sc in cs.subsections[:top_n]:
            print(f"  {sc.booking_no} {sc.full_name!r}  section={sc.extracted_section}"
                  f"  unresolved={sc.unresolved_patterns}")
