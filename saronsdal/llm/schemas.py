"""Phase 2.5 — Gemini suggestion schemas.

Dataclass definitions for all LLM request/response payloads and the aggregate
GeminiRunSummary that suggestion_writer serialises to output files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# City-aware disambiguation (deterministic pre-pass, Phase 2.5)
# ---------------------------------------------------------------------------

@dataclass
class CityContext:
    """Evidence from city-based church-label disambiguation.

    Attached to every GroupSuggestion that was processed by CityDisambiguator,
    whether the disambiguation was decisive (high confidence) or advisory.
    """
    raw_label: str              # generic phrase, e.g. "betel"
    city: str                   # dominant booking city, e.g. "Hommersåk"
    all_cities: List[str]       # all distinct cities found across bookings
    city_is_consistent: bool    # True when all bookings share the same city
    matched_canonical: Optional[str]    # e.g. "Betel Hommersåk"; None when no match
    match_source: str   # "known_alias" | "observed_phrase" | "city_context_only" | "no_city_data"
    confidence: float   # 0.0–1.0 (≥0.85 → deterministic; 0.40–0.84 → Gemini hint)
    rationale: str      # human-readable explanation


# ---------------------------------------------------------------------------
# Selector 1 + 3: group phrase / weak cluster classification
# ---------------------------------------------------------------------------

#: All valid classification values returned by Gemini for group phrases.
GROUP_CLASSIFICATIONS = (
    "new_group_alias",      # add to group_aliases.yaml under new key
    "known_group_variant",  # add as alias under an existing group entry
    "place_name",           # section or place name — add alias to sections.yaml
    "preference_language",  # placement desire, not a group (foretrekker, ønsker…)
    "person_name",          # individual person name slipped through heuristics
    "noise",                # random text; discard
)


@dataclass
class GroupSuggestion:
    """Gemini's verdict on one group phrase or weak cluster label."""
    phrase: str                         # normalized (lowercase) input phrase
    raw_variants: List[str]             # observed spellings from bookings
    classification: str                 # one of GROUP_CLASSIFICATIONS
    suggested_canonical: Optional[str]  # None for noise / person_name
    confidence: float                   # 0.0–1.0
    reasoning: str                      # one-sentence explanation
    booking_nos: List[str]              # bookings that contained this phrase
    city_context: Optional["CityContext"] = None  # set when city disambiguation ran


# ---------------------------------------------------------------------------
# Selector 2: near-text reference resolution
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRef:
    """One near-text fragment and Gemini's best match."""
    raw_fragment: str
    matched_booking_no: Optional[str]   # None = could not resolve
    match_type: str   # full_name | surname | family | first_name | group_reference | unresolved
    confidence: float


@dataclass
class NearTextSuggestion:
    """Gemini's resolution of all near-text fragments for one booking."""
    booking_no: str
    full_name: str
    resolved_refs: List[ResolvedRef]
    unresolved_fragments: List[str]     # fragments Gemini could not resolve
    notes: str
    place_refs: List[dict] = field(default_factory=list)  # evidence from place resolver


# ---------------------------------------------------------------------------
# Selector 4: preference extraction
# ---------------------------------------------------------------------------

@dataclass
class StructuredPreferences:
    """Placement preferences structured by Gemini from raw booking text."""
    # Avoidance
    avoid_river: bool = False
    avoid_noise: bool = False
    # Historical placement
    same_as_last_year: bool = False
    # Space / vehicle needs
    extra_space: bool = False
    # Positional proximity
    near_bibelskolen: bool = False
    near_hall: bool = False
    near_toilet: bool = False
    near_forest: bool = False
    # Ground conditions
    flat_ground: bool = False
    terrain_pref: str = ""          # "uphill" | "downhill" | ""
    drainage_concern: bool = False
    # Atmosphere
    quiet_spot: bool = False
    # Special needs
    accessibility: bool = False
    # Section Gemini inferred but Phase 1 missed
    inferred_section: str = ""
    # Free-text summary of any additional nuance
    notes: str = ""


@dataclass
class PreferenceSuggestion:
    """Gemini's structured extraction for one preference candidate."""
    booking_no: str
    full_name: str
    preferences: StructuredPreferences
    confidence: float
    raw_text: str                   # combined source text that triggered selection


# ---------------------------------------------------------------------------
# Selector 5: subsection / spot-range resolution
# ---------------------------------------------------------------------------

@dataclass
class SubsectionSuggestion:
    """Gemini's row and spot-ID suggestions for one subsection candidate."""
    booking_no: str
    full_name: str
    extracted_section: str
    suggested_rows: List[str]       # e.g. ["D", "E"]
    suggested_spot_ids: List[str]   # e.g. ["D25", "D26", "D27"]
    confidence: float
    notes: str


# ---------------------------------------------------------------------------
# Aggregate run result
# ---------------------------------------------------------------------------

@dataclass
class GeminiRunSummary:
    """All suggestions produced in one Phase 2.5 Gemini run."""
    group_suggestions: List[GroupSuggestion] = field(default_factory=list)
    near_text_suggestions: List[NearTextSuggestion] = field(default_factory=list)
    preference_suggestions: List[PreferenceSuggestion] = field(default_factory=list)
    subsection_suggestions: List[SubsectionSuggestion] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)    # parse / API errors logged
    model_used: str = ""
    candidates_processed: int = 0
    candidates_capped: int = 0      # existed but skipped due to --cap
