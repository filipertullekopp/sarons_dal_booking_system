"""Phase 2.5 — Deterministic place/city-based near-text resolver.

Scans near-text fragments for "fra <place>" patterns and cross-references the
detected place token(s) against the co-attending booking roster's City field.

This runs before (and feeds into) the Gemini near-text resolution step:
- Fragments where the place resolves to multiple co-attending city-matches are
  annotated as "place_group" references with a confidence score.
- Church/group names in the place position (Betel, Filadelfia, …) are
  flagged as "church_group" and left for city_disambiguator / Gemini.
- Campsite section names (Furulunden, Vårdalen, …) are excluded — they are
  spatial preferences, not person/group references.
- Broad geographic terms (Jæren, Rogaland) are detected and given reduced
  confidence.

Public API:
    resolve_place_refs(cand, bookings, *, sections_path, aliases_path)
        -> PlaceResolutionResult
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import ftfy
import yaml

from saronsdal.llm.city_disambiguator import _GENERIC_CHURCH_ROOTS
from saronsdal.llm.candidate_builder import NearTextCandidate

_CONFIG_ROOT = Path(__file__).parent.parent / "config"

# ---------------------------------------------------------------------------
# Place-fragment pattern
# ---------------------------------------------------------------------------

# Matches optional social prefix + "fra" + place token(s)
# Groups: (1) optional prefix word(s), (2) place string (rest of match)
_FRA_RE = re.compile(
    r"""
    (?:
        (?:familier?|de\s+andre|andre|venner?|folk(?:a)?|
           gjengen?|gruppa?|leire(?:t)?|sambygdinger?|naboer?)
        \s+
    )?
    fra\s+
    (.+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Split slash/comma-separated multi-place strings: "Moi/Lund", "Moi, Lund"
_MULTI_PLACE_SPLIT_RE = re.compile(r"[/,;]")

# ---------------------------------------------------------------------------
# Broad geographic terms that should attract reduced confidence
# ---------------------------------------------------------------------------

_BROAD_PLACES_RAW: FrozenSet[str] = frozenset({
    "jæren",
    "jaeren",
    "rogaland",
    "vestlandet",
    "sørlandet",
    "sorlandet",
    "norge",
    "norway",
    "landet",
    "regionen",
    "distriktet",
})


# ---------------------------------------------------------------------------
# Confidence rules
# ---------------------------------------------------------------------------

_CONF_ZERO_MATCHES    = 0.00
_CONF_ONE_MATCH       = 0.45   # weak — could be coincidental
_CONF_FEW_MATCHES     = 0.70   # 2–4 — probable group reference
_CONF_MANY_MATCHES    = 0.80   # 5+  — strong group reference
_BROAD_PLACE_PENALTY  = 0.6    # multiplier for broad place terms
_HIGH_CONF_THRESHOLD  = 0.65   # >= this → enrichment note in Gemini prompt


def _confidence(n_matches: int, is_broad: bool) -> float:
    if n_matches == 0:
        base = _CONF_ZERO_MATCHES
    elif n_matches == 1:
        base = _CONF_ONE_MATCH
    elif n_matches < 5:
        base = _CONF_FEW_MATCHES
    else:
        base = _CONF_MANY_MATCHES
    return round(base * (_BROAD_PLACE_PENALTY if is_broad else 1.0), 3)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PlaceRef:
    """Deterministic resolution of one near-text fragment via place/city matching."""
    raw_fragment: str
    place_tokens: List[str]          # normalized place names extracted from fragment
    match_type: str                  # "place_group" | "church_group" | "unresolved"
    matched_booking_nos: List[str]
    matched_cities: List[str]        # distinct city values of matched bookings
    confidence: float
    is_broad_place: bool
    rationale: str


@dataclass
class PlaceResolutionResult:
    """All place-based resolutions for one NearTextCandidate."""
    booking_no: str
    refs: List[PlaceRef] = field(default_factory=list)

    @property
    def has_any_resolution(self) -> bool:
        return any(r.confidence > 0 for r in self.refs)

    @property
    def all_high_confidence(self) -> bool:
        return bool(self.refs) and all(r.confidence >= _HIGH_CONF_THRESHOLD for r in self.refs)

    def as_prompt_context(self) -> List[dict]:
        """Serialisable form for injection into a Gemini prompt."""
        return [
            {
                "fragment": r.raw_fragment,
                "place_tokens": r.place_tokens,
                "match_type": r.match_type,
                "matched_booking_nos": r.matched_booking_nos,
                "matched_cities": r.matched_cities,
                "confidence": r.confidence,
                "is_broad_place": r.is_broad_place,
                "rationale": r.rationale,
            }
            for r in self.refs
        ]


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def _load_section_names(sections_path: Optional[Path] = None) -> FrozenSet[str]:
    """Return all section name variants (lowercased) from sections.yaml."""
    p = sections_path or _CONFIG_ROOT / "sections.yaml"
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    names: set = set()
    for entry in cfg.get("sections", {}).values():
        names.add(ftfy.fix_text(entry.get("canonical", "")).lower())
        for alias in entry.get("aliases", []):
            names.add(ftfy.fix_text(alias).lower())
    for name in cfg.get("section_name_set", []):
        names.add(ftfy.fix_text(name).lower())
    return frozenset(names)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()


# Pre-compute ascii-folded broad-place set so matching works regardless of diacritics.
_BROAD_PLACES_FOLDED: FrozenSet[str] = frozenset(_ascii_fold(p) for p in _BROAD_PLACES_RAW)


def _normalize_place(raw: str) -> str:
    """Strip punctuation, fix encoding, lowercase."""
    return ftfy.fix_text(raw.strip(" .,!?")).lower()


def _is_church_root(token_lower: str) -> bool:
    """True if the place token is a generic church label."""
    for root in _GENERIC_CHURCH_ROOTS:
        if token_lower == root or token_lower.startswith(root):
            return True
    return False


def _is_broad(token_lower: str) -> bool:
    return _ascii_fold(token_lower) in _BROAD_PLACES_FOLDED


def _extract_place_tokens(fra_match: str) -> List[str]:
    """
    Split a raw place string on slashes/commas and normalize each token.

    "Moi/Lund"  -> ["moi", "lund"]
    "Varhaug"   -> ["varhaug"]
    "Moi, Lund" -> ["moi", "lund"]
    """
    parts = _MULTI_PLACE_SPLIT_RE.split(fra_match)
    return [_normalize_place(p) for p in parts if _normalize_place(p)]


def _city_matches_token(city: str, token: str) -> bool:
    """Loose containment match between a booking city and a place token."""
    if not city or not token:
        return False
    city_f = _ascii_fold(city)
    token_f = _ascii_fold(token)
    return token_f in city_f or city_f in token_f


def _find_city_matches(
    tokens: List[str],
    roster: List[dict],
) -> Tuple[List[str], List[str]]:
    """
    Return (matched_booking_nos, matched_cities) for bookings whose city
    matches any of the given place tokens.
    """
    bno_set: list[str] = []
    city_set: list[str] = []
    seen_bnos: set[str] = set()
    for entry in roster:
        bno = entry.get("booking_no", "")
        city = entry.get("city", "").strip()
        if not city:
            continue
        for token in tokens:
            if _city_matches_token(city, token):
                if bno not in seen_bnos:
                    bno_set.append(bno)
                    seen_bnos.add(bno)
                    if city not in city_set:
                        city_set.append(city)
                break
    return bno_set, city_set


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------

def _resolve_fragment(
    fragment: str,
    roster: List[dict],
    section_names: FrozenSet[str],
) -> PlaceRef:
    """Resolve one near-text fragment to a PlaceRef."""
    # Extract the "fra X" part
    m = _FRA_RE.search(fragment)
    if not m:
        # Even without a "fra" pattern, flag fragments that contain a church root
        # so callers know to route them to church disambiguation instead.
        frag_lower = _normalize_place(fragment)
        for word in re.split(r'\s+', frag_lower):
            if _is_church_root(word):
                return PlaceRef(
                    raw_fragment=fragment,
                    place_tokens=[word],
                    match_type="church_group",
                    matched_booking_nos=[],
                    matched_cities=[],
                    confidence=0.0,
                    is_broad_place=False,
                    rationale=(
                        f"Fragment contains church/org label '{word}' "
                        f"— handled by church disambiguation."
                    ),
                )
        return PlaceRef(
            raw_fragment=fragment,
            place_tokens=[],
            match_type="unresolved",
            matched_booking_nos=[],
            matched_cities=[],
            confidence=0.0,
            is_broad_place=False,
            rationale="No 'fra <place>' pattern detected.",
        )

    raw_place = m.group(1).strip()
    tokens = _extract_place_tokens(raw_place)

    if not tokens:
        return PlaceRef(
            raw_fragment=fragment,
            place_tokens=[],
            match_type="unresolved",
            matched_booking_nos=[],
            matched_cities=[],
            confidence=0.0,
            is_broad_place=False,
            rationale="Empty place token after extraction.",
        )

    # Check each token: section name? church name? broad?
    section_tokens = [t for t in tokens if t in section_names]
    church_tokens  = [t for t in tokens if _is_church_root(t) and t not in section_names]
    place_tokens   = [t for t in tokens if t not in section_names and not _is_church_root(t)]
    broad_tokens   = [t for t in place_tokens if _is_broad(t)]

    # Section names are spatial preferences, not group references
    if section_tokens and not church_tokens and not place_tokens:
        return PlaceRef(
            raw_fragment=fragment,
            place_tokens=tokens,
            match_type="unresolved",
            matched_booking_nos=[],
            matched_cities=[],
            confidence=0.0,
            is_broad_place=False,
            rationale=(
                f"Place token(s) {section_tokens!r} are campsite section names "
                f"— not a near-text group reference."
            ),
        )

    # Church/group references: flag and return without roster matching
    if church_tokens:
        return PlaceRef(
            raw_fragment=fragment,
            place_tokens=tokens,
            match_type="church_group",
            matched_booking_nos=[],
            matched_cities=[],
            confidence=0.0,
            is_broad_place=False,
            rationale=(
                f"Place token(s) {church_tokens!r} are church/org labels "
                f"— handled by church disambiguation, not place resolver."
            ),
        )

    if not place_tokens:
        return PlaceRef(
            raw_fragment=fragment,
            place_tokens=tokens,
            match_type="unresolved",
            matched_booking_nos=[],
            matched_cities=[],
            confidence=0.0,
            is_broad_place=False,
            rationale="No usable place token after filtering sections and church labels.",
        )

    is_broad = bool(broad_tokens)
    matched_bnos, matched_cities = _find_city_matches(place_tokens, roster)
    conf = _confidence(len(matched_bnos), is_broad)

    if conf == 0.0:
        rationale = (
            f"No co-attending bookings found with city matching "
            f"{place_tokens!r}."
        )
        match_type = "unresolved"
    else:
        city_str = ", ".join(f"'{c}'" for c in matched_cities)
        broad_note = f" (broad region — confidence reduced)" if is_broad else ""
        rationale = (
            f"{len(matched_bnos)} co-attending booking(s) have city matching "
            f"{place_tokens!r} ({city_str}){broad_note}."
        )
        match_type = "place_group"

    return PlaceRef(
        raw_fragment=fragment,
        place_tokens=place_tokens,
        match_type=match_type,
        matched_booking_nos=matched_bnos,
        matched_cities=matched_cities,
        confidence=conf,
        is_broad_place=is_broad,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Roster builder (re-uses city field from bookings)
# ---------------------------------------------------------------------------

def _build_place_roster(
    bookings: List[dict],
    cand: NearTextCandidate,
) -> List[dict]:
    """
    Return co-attending bookings for the candidate's date range.
    Includes the city field for place matching.
    Mirrors the roster builder in gemini_client.py but adds city.
    """
    from datetime import date as _date

    def _pd(s: str) -> Optional[_date]:
        try:
            return _date.fromisoformat(s)
        except (ValueError, TypeError, AttributeError):
            return None

    cin = _pd(cand.check_in or "")
    cout = _pd(cand.check_out or "")
    if not (cin and cout):
        return []

    roster: List[dict] = []
    for b in bookings:
        if b.get("booking_no") == cand.booking_no:
            continue
        b_cin = _pd(b.get("check_in", ""))
        b_cout = _pd(b.get("check_out", ""))
        if not (b_cin and b_cout):
            continue
        if cin < b_cout and cout > b_cin:
            sections = b.get("request", {}).get("preferred_sections", [])
            roster.append({
                "booking_no": b.get("booking_no", ""),
                "full_name": b.get("full_name", ""),
                "check_in": b.get("check_in", ""),
                "check_out": b.get("check_out", ""),
                "section": sections[0] if sections else "",
                "city": b.get("city", ""),
            })
    return roster


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_place_refs(
    cand: NearTextCandidate,
    bookings: List[dict],
    *,
    sections_path: Optional[Path] = None,
    aliases_path: Optional[Path] = None,
) -> PlaceResolutionResult:
    """
    Resolve near-text fragments in *cand* using place/city matching.

    Args:
        cand:          NearTextCandidate with raw_near_texts to resolve
        bookings:      all Phase 1 Booking dicts (for roster + city lookup)
        sections_path: override path to sections.yaml (for testing)
        aliases_path:  unused; reserved for future alias expansion

    Returns:
        PlaceResolutionResult with one PlaceRef per fragment.
    """
    section_names = _load_section_names(sections_path)
    roster = _build_place_roster(bookings, cand)

    result = PlaceResolutionResult(booking_no=cand.booking_no)
    for frag in cand.raw_near_texts:
        if not frag.strip():
            continue
        ref = _resolve_fragment(frag, roster, section_names)
        result.refs.append(ref)

    return result
