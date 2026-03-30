"""Phase 2.5 — Deterministic city-aware church label disambiguation.

Before sending generic church labels (Betel, Filadelfia, Sion, Betania…) to
Gemini, we try to resolve them from the booking's City column combined with
known canonical names from group_aliases.yaml.

Three outcomes:
  high confidence (>= 0.85)  — resolved deterministically; skip Gemini
  advisory     (0.40 – 0.84) — city hint injected into Gemini prompt
  no data      (0.00)        — city absent or ambiguous; Gemini handles unaided

The full evidence is stored in CityContext and attached to the GroupSuggestion.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import ftfy
import yaml

from saronsdal.llm.schemas import CityContext

_CONFIG_ROOT = Path(__file__).parent.parent / "config"

# ---------------------------------------------------------------------------
# Generic church roots — phrases that are too broad without a city qualifier
# ---------------------------------------------------------------------------

_GENERIC_CHURCH_ROOTS: FrozenSet[str] = frozenset({
    "betania",
    "betel",
    "filadelfia",
    "filadelfiakirken",
    "sion",
    "pinsekirken",
    "pinsekirka",
    "pinsemenigheten",
    "klippen",
    "oasen",
    "oasenkirka",
    "betelgjengen",
    "betaniakirken",
    "karismakirken",
    "livskirken",
})

# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

_CONF_KNOWN_ALIAS_CONSISTENT  = 0.90   # alias list hit + single city
_CONF_KNOWN_ALIAS_MIXED       = 0.76   # alias list hit + multiple cities
_CONF_OBSERVED_CONSISTENT     = 0.80   # recurring phrase seen with a city suffix
_CONF_OBSERVED_MIXED          = 0.68   # recurring phrase, mixed cities
_CONF_CITY_ONLY_CONSISTENT    = 0.40   # root + city → constructed name, single city
_CONF_CITY_ONLY_MIXED         = 0.34   # root + city → constructed name, mixed cities
_CONF_NO_DATA                 = 0.00   # city absent or unusable

_HIGH_CONF_THRESHOLD          = 0.85   # >= this → skip Gemini


# ---------------------------------------------------------------------------
# Canonical name loader
# ---------------------------------------------------------------------------

@dataclass
class _CanonicalEntry:
    key: str            # YAML key, e.g. "betel_hommersaak"
    canonical: str      # display label, e.g. "Betel Hommersåk"
    root: str           # lowercase bare root, e.g. "betel"
    city: str           # lowercase city suffix, e.g. "hommersåk"
    aliases_lower: List[str]


def _ascii_fold(s: str) -> str:
    """Lowercase + strip diacritics for loose matching."""
    return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()


def _split_root_city(canonical: str) -> Tuple[str, str]:
    """
    Split a canonical label into (root, city) parts.

    "Betel Hommersåk"          -> ("betel", "hommersåk")
    "Filadelfiakirken Lyngdal"  -> ("filadelfiakirken", "lyngdal")
    "Pinsekirken Flekkefjord"   -> ("pinsekirken", "flekkefjord")
    "Betania Sokndal"           -> ("betania", "sokndal")
    "Venneslagruppa"            -> ("venneslagruppa", "")   # no city part
    """
    parts = canonical.strip().split()
    if len(parts) < 2:
        return canonical.lower(), ""
    # First word is the church root; the rest is the city (may be multi-word)
    root = parts[0].lower()
    city = " ".join(parts[1:]).lower()
    return root, city


def _load_canonical_entries(aliases_path: Optional[Path] = None) -> List[_CanonicalEntry]:
    """Load all canonical entries from group_aliases.yaml."""
    p = aliases_path or _CONFIG_ROOT / "group_aliases.yaml"
    with open(p, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    entries: List[_CanonicalEntry] = []
    for key, data in cfg.get("organizations", {}).items():
        canonical = ftfy.fix_text(data.get("canonical", ""))
        root, city = _split_root_city(canonical)
        aliases_lower = [
            ftfy.fix_text(a).lower() for a in data.get("aliases", [])
        ]
        entries.append(_CanonicalEntry(
            key=key,
            canonical=canonical,
            root=root,
            city=city,
            aliases_lower=aliases_lower,
        ))
    return entries


# ---------------------------------------------------------------------------
# DisambiguationMap
# ---------------------------------------------------------------------------

@dataclass
class DisambiguationMap:
    """City-context evidence for a set of group phrases, keyed by phrase (lowercase)."""
    contexts: Dict[str, CityContext] = field(default_factory=dict)

    def get(self, phrase: str) -> Optional[CityContext]:
        return self.contexts.get(phrase.lower())

    def is_high_confidence(self, phrase: str) -> bool:
        ctx = self.get(phrase)
        return ctx is not None and ctx.confidence >= _HIGH_CONF_THRESHOLD


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_generic_root(phrase_lower: str) -> bool:
    """True if the phrase is (or starts with) a known generic church root."""
    for root in _GENERIC_CHURCH_ROOTS:
        if phrase_lower == root or phrase_lower.startswith(root):
            return True
    return False


def _root_of(phrase_lower: str) -> str:
    """Extract the matching generic root from a phrase, or return the phrase."""
    for root in _GENERIC_CHURCH_ROOTS:
        if phrase_lower == root or phrase_lower.startswith(root):
            return root
    return phrase_lower


def _city_matches(city_lower: str, canonical_city: str) -> bool:
    """Loose city match: ascii-fold both sides and check containment."""
    if not city_lower or not canonical_city:
        return False
    return _ascii_fold(city_lower) in _ascii_fold(canonical_city) or \
           _ascii_fold(canonical_city) in _ascii_fold(city_lower)


def _collect_cities(phrase: str, booking_nos: List[str], bno_map: Dict[str, dict]) -> List[str]:
    """Gather all non-empty city values for the given booking_nos."""
    cities: List[str] = []
    for bno in booking_nos:
        b = bno_map.get(bno, {})
        city = b.get("city", "").strip()
        if city:
            cities.append(city)
    return cities


def _build_observed_org_phrases(
    bookings: List[dict],
    canonical_entries: List[_CanonicalEntry],
) -> Dict[str, str]:
    """
    Scan all bookings for org/group_field values that combine a generic root
    with a city-like suffix (e.g. "Betel Stavanger") but are NOT in the known
    alias list.  Returns {phrase_lower: first_observed_raw_form}.
    """
    known_lower = {alias for e in canonical_entries for alias in e.aliases_lower}
    observed: Dict[str, str] = {}
    for b in bookings:
        gs = b.get("group_signals", {})
        for raw in (gs.get("organization") or "", gs.get("group_field") or ""):
            raw = raw.strip()
            if not raw:
                continue
            fixed = ftfy.fix_text(raw)
            lower = fixed.lower()
            if lower in known_lower:
                continue
            if _is_generic_root(lower):
                observed.setdefault(lower, fixed)
    return observed


def _disambiguate_one(
    phrase: str,                    # normalized phrase (lowercase)
    raw_variants: List[str],        # observed spellings
    booking_nos: List[str],         # bookings where this phrase appears
    bno_map: Dict[str, dict],       # all bookings by booking_no
    canonical_entries: List[_CanonicalEntry],
) -> Optional[CityContext]:
    """
    Attempt to build a CityContext for one generic church phrase.

    Returns None when the phrase is not a generic church root (caller should
    skip non-generic phrases entirely).
    """
    if not _is_generic_root(phrase):
        return None

    cities = _collect_cities(phrase, booking_nos, bno_map)
    all_cities = sorted(set(cities))
    city_is_consistent = len(set(c.lower() for c in all_cities)) <= 1
    dominant_city = all_cities[0] if all_cities else ""

    root = _root_of(phrase)

    # ---------------------------------------------------------------
    # 1. Check known alias list: does any alias in the entry match
    #    one of the raw_variants?
    # ---------------------------------------------------------------
    for entry in canonical_entries:
        if entry.root != root and not _ascii_fold(entry.root).startswith(_ascii_fold(root)):
            continue
        # Alias match?
        for variant in raw_variants:
            if ftfy.fix_text(variant).lower() in entry.aliases_lower:
                # Check city consistency with the entry's city
                if dominant_city and entry.city and _city_matches(dominant_city, entry.city):
                    confidence = (
                        _CONF_KNOWN_ALIAS_CONSISTENT
                        if city_is_consistent
                        else _CONF_KNOWN_ALIAS_MIXED
                    )
                    rationale = (
                        f"Alias '{variant}' matches canonical '{entry.canonical}'; "
                        f"booking city '{dominant_city}' consistent with "
                        f"canonical city '{entry.city}'."
                    )
                elif not dominant_city:
                    # Known alias, no city data — advisory only
                    confidence = 0.60
                    rationale = (
                        f"Alias '{variant}' matches canonical '{entry.canonical}'; "
                        f"no city data in bookings to confirm."
                    )
                else:
                    # Alias match but city doesn't match the canonical's city
                    confidence = 0.50
                    rationale = (
                        f"Alias '{variant}' matches canonical '{entry.canonical}'; "
                        f"but booking city '{dominant_city}' does not match "
                        f"canonical city '{entry.city}'."
                    )
                return CityContext(
                    raw_label=phrase,
                    city=dominant_city,
                    all_cities=all_cities,
                    city_is_consistent=city_is_consistent,
                    matched_canonical=entry.canonical,
                    match_source="known_alias",
                    confidence=confidence,
                    rationale=rationale,
                )

    # ---------------------------------------------------------------
    # 2. City + root → match canonical by root + city (prefix-aware)
    # ---------------------------------------------------------------
    if dominant_city:
        for entry in canonical_entries:
            root_matches = (
                entry.root == root
                or _ascii_fold(entry.root).startswith(_ascii_fold(root))
                or _ascii_fold(root).startswith(_ascii_fold(entry.root))
            )
            if not root_matches:
                continue
            if _city_matches(dominant_city, entry.city):
                confidence = (
                    _CONF_KNOWN_ALIAS_CONSISTENT
                    if city_is_consistent
                    else _CONF_KNOWN_ALIAS_MIXED
                )
                rationale = (
                    f"Phrase root '{root}' matches canonical root '{entry.root}'; "
                    f"booking city '{dominant_city}' matches canonical city '{entry.city}'."
                )
                return CityContext(
                    raw_label=phrase,
                    city=dominant_city,
                    all_cities=all_cities,
                    city_is_consistent=city_is_consistent,
                    matched_canonical=entry.canonical,
                    match_source="observed_phrase",
                    confidence=confidence,
                    rationale=rationale,
                )

        # ---------------------------------------------------------------
        # 3. City known but no canonical found — construct a candidate name
        # ---------------------------------------------------------------
        constructed = f"{phrase.capitalize()} {dominant_city.title()}"
        confidence = (
            _CONF_CITY_ONLY_CONSISTENT if city_is_consistent else _CONF_CITY_ONLY_MIXED
        )
        rationale = (
            f"Phrase '{phrase}' is a generic church root; booking city "
            f"'{dominant_city}' suggests '{constructed}' but no known canonical found."
        )
        return CityContext(
            raw_label=phrase,
            city=dominant_city,
            all_cities=all_cities,
            city_is_consistent=city_is_consistent,
            matched_canonical=None,
            match_source="city_context_only",
            confidence=confidence,
            rationale=rationale,
        )

    # ---------------------------------------------------------------
    # 4. No city data at all
    # ---------------------------------------------------------------
    return CityContext(
        raw_label=phrase,
        city="",
        all_cities=[],
        city_is_consistent=False,
        matched_canonical=None,
        match_source="no_city_data",
        confidence=_CONF_NO_DATA,
        rationale=f"Phrase '{phrase}' is a generic church root but no city data found.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_disambiguation_map(
    phrase_booking_nos: Dict[str, List[str]],   # phrase_lower -> booking_nos
    phrase_raw_variants: Dict[str, List[str]],   # phrase_lower -> raw spellings
    bookings: List[dict],
    aliases_path: Optional[Path] = None,
) -> DisambiguationMap:
    """
    Build a DisambiguationMap for all phrases that are generic church roots.

    Non-generic phrases are ignored (their absence from the returned map
    signals that the caller should let Gemini handle them normally).

    Args:
        phrase_booking_nos:  phrase → list of booking numbers where it appears
        phrase_raw_variants: phrase → list of observed raw spellings
        bookings:            all Phase 1 Booking dicts (from bookings_normalized.json)
        aliases_path:        override path for group_aliases.yaml (for testing)

    Returns:
        DisambiguationMap with CityContext for each resolved generic phrase.
    """
    canonical_entries = _load_canonical_entries(aliases_path)
    bno_map = {b.get("booking_no", ""): b for b in bookings}

    dmap = DisambiguationMap()
    for phrase, booking_nos in phrase_booking_nos.items():
        raw_variants = phrase_raw_variants.get(phrase, [phrase])
        ctx = _disambiguate_one(
            phrase=phrase,
            raw_variants=raw_variants,
            booking_nos=booking_nos,
            bno_map=bno_map,
            canonical_entries=canonical_entries,
        )
        if ctx is not None:
            dmap.contexts[phrase] = ctx
    return dmap
