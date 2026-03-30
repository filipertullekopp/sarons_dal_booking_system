"""Phase 2.5 — Gemini API client.

Sends batched JSON-mode prompts for each candidate type and returns a
GeminiRunSummary containing all structured suggestions.

Requires:
  pip install google-genai
  GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable set

Model default: gemini-2.5-flash
  Override per-run: --gemini-model MODEL
  Override globally: GEMINI_MODEL environment variable
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from typing import List, Optional

from google import genai
from google.genai import types as genai_types

from saronsdal.llm.candidate_builder import (
    CandidateSet,
    GroupPhraseCandidate,
    NearTextCandidate,
    PreferenceCandidate,
    SubsectionCandidate,
    WeakClusterCandidate,
)
from saronsdal.llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_group_phrase_prompt,
    build_near_text_prompt,
    build_preference_prompt,
    build_subsection_prompt,
)
from saronsdal.llm.city_disambiguator import build_disambiguation_map
from saronsdal.llm.place_resolver import resolve_place_refs
from saronsdal.llm.schemas import (
    CityContext,
    GeminiRunSummary,
    GroupSuggestion,
    NearTextSuggestion,
    PreferenceSuggestion,
    ResolvedRef,
    StructuredPreferences,
    SubsectionSuggestion,
)

logger = logging.getLogger(__name__)

_PREF_BATCH = 5      # preference candidates per Gemini call
_SUBSEC_BATCH = 5    # subsection candidates per Gemini call
_ROSTER_MAX = 40     # max co-attending guests included in near-text prompt


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "gemini-2.5-flash"


def create_client(
    model_name: str = _DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> "GeminiClient":
    """Create a GeminiClient, reading credentials and model from env if not provided.

    Environment variables:
        GEMINI_API_KEY or GOOGLE_API_KEY — required for live runs
        GEMINI_MODEL — overrides model_name when set
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError(
            "Gemini API key not found.  "
            "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    model = os.environ.get("GEMINI_MODEL", model_name)
    return GeminiClient(api_key=key, model_name=model)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Wraps google-genai for Phase 2.5 enrichment."""

    def __init__(self, api_key: str, model_name: str = _DEFAULT_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._gen_cfg = genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            temperature=0.1,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_all(
        self,
        cs: CandidateSet,
        bookings: List[dict],
        cap: Optional[int] = None,
    ) -> GeminiRunSummary:
        """Process all candidate types; return aggregate GeminiRunSummary."""
        run = GeminiRunSummary(model_used=self._model_name)

        # Selector 1 + 3: group phrases + weak clusters (one batched call)
        gp = cs.group_phrases[:cap] if cap else cs.group_phrases
        wc = cs.weak_clusters[:cap] if cap else cs.weak_clusters
        run.candidates_capped += max(0, len(cs.group_phrases) - len(gp))
        run.candidates_capped += max(0, len(cs.weak_clusters) - len(wc))
        if gp or wc:
            run.group_suggestions = self._classify_group_phrases(gp, wc, bookings, run)
            run.candidates_processed += len(gp) + len(wc)

        # Selector 2: near-text (one call per candidate — roster varies per booking)
        nt = cs.near_text[:cap] if cap else cs.near_text
        run.candidates_capped += max(0, len(cs.near_text) - len(nt))
        if nt:
            run.near_text_suggestions = self._resolve_near_texts(nt, bookings, run)
            run.candidates_processed += len(nt)

        # Selector 4: preferences (batched)
        pref = cs.preferences[:cap] if cap else cs.preferences
        run.candidates_capped += max(0, len(cs.preferences) - len(pref))
        if pref:
            run.preference_suggestions = self._extract_preferences(pref, run)
            run.candidates_processed += len(pref)

        # Selector 5: subsections (batched)
        sub = cs.subsections[:cap] if cap else cs.subsections
        run.candidates_capped += max(0, len(cs.subsections) - len(sub))
        if sub:
            run.subsection_suggestions = self._resolve_subsections(sub, run)
            run.candidates_processed += len(sub)

        logger.info(
            "Gemini run complete: %d processed, %d capped, %d errors",
            run.candidates_processed, run.candidates_capped, len(run.errors),
        )
        return run

    # ------------------------------------------------------------------
    # Selector 1 + 3
    # ------------------------------------------------------------------

    def _classify_group_phrases(
        self,
        phrase_cands: List[GroupPhraseCandidate],
        cluster_cands: List[WeakClusterCandidate],
        bookings: List[dict],
        run: GeminiRunSummary,
    ) -> List[GroupSuggestion]:
        # Build lookups used both for disambiguation and result assembly
        phrase_bno: dict[str, List[str]] = {c.phrase: c.booking_nos for c in phrase_cands}
        phrase_variants: dict[str, List[str]] = {
            c.phrase: c.raw_variants for c in phrase_cands
        }
        cluster_bno: dict[str, List[str]] = {
            wc.canonical_label.lower(): wc.member_booking_nos for wc in cluster_cands
        }

        # ---------------------------------------------------------------
        # Deterministic city disambiguation pass
        # ---------------------------------------------------------------
        dmap = build_disambiguation_map(
            phrase_booking_nos=phrase_bno,
            phrase_raw_variants=phrase_variants,
            bookings=bookings,
        )

        # Phrases that are deterministically resolved skip Gemini
        skip_set: set[str] = set()
        deterministic_results: List[GroupSuggestion] = []
        for phrase, ctx in dmap.contexts.items():
            if dmap.is_high_confidence(phrase):
                booking_nos = phrase_bno.get(phrase, [])
                raw_variants = phrase_variants.get(phrase, [phrase])
                deterministic_results.append(GroupSuggestion(
                    phrase=phrase,
                    raw_variants=raw_variants,
                    classification="known_group_variant" if ctx.matched_canonical else "new_group_alias",
                    suggested_canonical=ctx.matched_canonical,
                    confidence=ctx.confidence,
                    reasoning=ctx.rationale,
                    booking_nos=booking_nos,
                    city_context=ctx,
                ))
                skip_set.add(phrase)
                logger.debug(
                    "City disambiguation resolved '%s' -> '%s' (conf=%.2f)",
                    phrase, ctx.matched_canonical, ctx.confidence,
                )

        # Build city_hints dict for Gemini (advisory tier only)
        city_hints: dict[str, dict] = {}
        for phrase, ctx in dmap.contexts.items():
            if phrase not in skip_set and ctx.confidence >= 0.40:
                city_hints[phrase] = {
                    "city": ctx.city,
                    "matched_canonical": ctx.matched_canonical,
                    "confidence": ctx.confidence,
                    "rationale": ctx.rationale,
                }

        # Filter out deterministically resolved candidates before sending to Gemini
        gemini_phrase_cands = [c for c in phrase_cands if c.phrase not in skip_set]

        logger.info(
            "City disambiguation: %d deterministic, %d advisory hints, "
            "%d sent to Gemini",
            len(deterministic_results), len(city_hints), len(gemini_phrase_cands),
        )

        if not gemini_phrase_cands and not cluster_cands:
            return deterministic_results

        # ---------------------------------------------------------------
        # Gemini call for remaining candidates
        # ---------------------------------------------------------------
        prompt = build_group_phrase_prompt(
            gemini_phrase_cands, cluster_cands, bookings, city_hints=city_hints
        )
        raw = self._call(prompt, "group_phrases")
        if raw is None:
            return deterministic_results

        results: List[GroupSuggestion] = list(deterministic_results)
        for item in raw if isinstance(raw, list) else [raw]:
            try:
                item_id = item.get("id") or item.get("phrase", "")
                booking_nos = phrase_bno.get(item_id) or cluster_bno.get(item_id, [])
                raw_variants = phrase_variants.get(item_id, [item_id])
                city_ctx = dmap.get(item_id)   # attach advisory context if present
                results.append(GroupSuggestion(
                    phrase=item_id,
                    raw_variants=raw_variants,
                    classification=item.get("classification", "noise"),
                    suggested_canonical=item.get("suggested_canonical"),
                    confidence=float(item.get("confidence", 0.0)),
                    reasoning=item.get("reasoning", ""),
                    booking_nos=booking_nos,
                    city_context=city_ctx,
                ))
            except Exception as exc:
                run.errors.append(f"group_phrase parse: {exc} item={item!r:.120}")
        logger.info("Group phrase classification: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Selector 2
    # ------------------------------------------------------------------

    def _resolve_near_texts(
        self,
        candidates: List[NearTextCandidate],
        bookings: List[dict],
        run: GeminiRunSummary,
    ) -> List[NearTextSuggestion]:
        results: List[NearTextSuggestion] = []
        for cand in candidates:
            # -----------------------------------------------------------
            # Deterministic place/city pass
            # -----------------------------------------------------------
            place_result = resolve_place_refs(cand, bookings)
            place_context = place_result.as_prompt_context() if place_result.has_any_resolution else None

            # If every fragment is resolved with high confidence, skip Gemini
            if place_result.all_high_confidence:
                resolved = [
                    ResolvedRef(
                        raw_fragment=r.raw_fragment,
                        matched_booking_no=(
                            r.matched_booking_nos[0] if len(r.matched_booking_nos) == 1
                            else None
                        ),
                        match_type=r.match_type,
                        confidence=r.confidence,
                    )
                    for r in place_result.refs
                ]
                unresolved = [
                    r.raw_fragment for r in place_result.refs if r.confidence == 0.0
                ]
                results.append(NearTextSuggestion(
                    booking_no=cand.booking_no,
                    full_name=cand.full_name,
                    resolved_refs=resolved,
                    unresolved_fragments=unresolved,
                    notes="Resolved deterministically by place/city matcher.",
                    place_refs=place_result.as_prompt_context(),
                ))
                logger.debug(
                    "near_text/%s: all fragments resolved by place matcher — skipped Gemini",
                    cand.booking_no,
                )
                continue

            # -----------------------------------------------------------
            # Gemini call — enriched with place context when available
            # -----------------------------------------------------------
            roster = _build_roster(bookings, cand)
            prompt = build_near_text_prompt(cand, roster, place_context=place_context)
            raw = self._call(prompt, f"near_text/{cand.booking_no}")
            if raw is None:
                continue
            try:
                d = raw if isinstance(raw, dict) else (raw[0] if raw else {})
                resolved = [
                    ResolvedRef(
                        raw_fragment=r.get("raw_fragment", ""),
                        matched_booking_no=r.get("matched_booking_no"),
                        match_type=r.get("match_type", "unresolved"),
                        confidence=float(r.get("confidence", 0.0)),
                    )
                    for r in d.get("resolved_refs", [])
                ]
                results.append(NearTextSuggestion(
                    booking_no=cand.booking_no,
                    full_name=cand.full_name,
                    resolved_refs=resolved,
                    unresolved_fragments=d.get("unresolved_fragments", []),
                    notes=d.get("notes", ""),
                    place_refs=place_result.as_prompt_context(),
                ))
            except Exception as exc:
                run.errors.append(f"near_text parse ({cand.booking_no}): {exc}")
        logger.info("Near-text resolution: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Selector 4
    # ------------------------------------------------------------------

    def _extract_preferences(
        self,
        candidates: List[PreferenceCandidate],
        run: GeminiRunSummary,
    ) -> List[PreferenceSuggestion]:
        results: List[PreferenceSuggestion] = []
        for i in range(0, len(candidates), _PREF_BATCH):
            batch = candidates[i : i + _PREF_BATCH]
            prompt = build_preference_prompt(batch)
            raw = self._call(prompt, f"preferences/batch_{i // _PREF_BATCH}")
            if raw is None:
                continue
            bno_map = {c.booking_no: c for c in batch}
            for item in raw if isinstance(raw, list) else [raw]:
                try:
                    bno = str(item.get("booking_no", ""))
                    cand = bno_map.get(bno)
                    if cand is None:
                        continue
                    pd = item.get("preferences", {})
                    results.append(PreferenceSuggestion(
                        booking_no=bno,
                        full_name=cand.full_name,
                        preferences=StructuredPreferences(
                            avoid_river=bool(pd.get("avoid_river", False)),
                            avoid_noise=bool(pd.get("avoid_noise", False)),
                            same_as_last_year=bool(pd.get("same_as_last_year", False)),
                            extra_space=bool(pd.get("extra_space", False)),
                            near_bibelskolen=bool(pd.get("near_bibelskolen", False)),
                            near_hall=bool(pd.get("near_hall", False)),
                            flat_ground=bool(pd.get("flat_ground", False)),
                            terrain_pref=str(pd.get("terrain_pref", "")),
                            near_toilet=bool(pd.get("near_toilet", False)),
                            near_forest=bool(pd.get("near_forest", False)),
                            quiet_spot=bool(pd.get("quiet_spot", False)),
                            drainage_concern=bool(pd.get("drainage_concern", False)),
                            accessibility=bool(pd.get("accessibility", False)),
                            inferred_section=str(pd.get("inferred_section", "")),
                            notes=str(pd.get("notes", "")),
                        ),
                        confidence=float(item.get("confidence", 0.0)),
                        raw_text=cand.raw_text,
                    ))
                except Exception as exc:
                    run.errors.append(f"preference parse: {exc} item={item!r:.120}")
        logger.info("Preference extraction: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Selector 5
    # ------------------------------------------------------------------

    def _resolve_subsections(
        self,
        candidates: List[SubsectionCandidate],
        run: GeminiRunSummary,
    ) -> List[SubsectionSuggestion]:
        results: List[SubsectionSuggestion] = []
        for i in range(0, len(candidates), _SUBSEC_BATCH):
            batch = candidates[i : i + _SUBSEC_BATCH]
            prompt = build_subsection_prompt(batch)
            raw = self._call(prompt, f"subsections/batch_{i // _SUBSEC_BATCH}")
            if raw is None:
                continue
            bno_map = {c.booking_no: c for c in batch}
            for item in raw if isinstance(raw, list) else [raw]:
                try:
                    bno = str(item.get("booking_no", ""))
                    cand = bno_map.get(bno)
                    if cand is None:
                        continue
                    results.append(SubsectionSuggestion(
                        booking_no=bno,
                        full_name=cand.full_name,
                        extracted_section=cand.extracted_section,
                        suggested_rows=item.get("suggested_rows", []),
                        suggested_spot_ids=item.get("suggested_spot_ids", []),
                        confidence=float(item.get("confidence", 0.0)),
                        notes=item.get("notes", ""),
                    ))
                except Exception as exc:
                    run.errors.append(f"subsection parse: {exc} item={item!r:.120}")
        logger.info("Subsection resolution: %d results", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, prompt: str, context: str = "") -> Optional[object]:
        """Send one prompt; return parsed JSON or None on any error."""
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._gen_cfg,
            )
            text = response.text
        except Exception as exc:
            logger.warning("Gemini API error (%s): %s", context, exc)
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Gemini JSON parse error (%s): %s — response=%r", context, exc, text[:200]
            )
            return None


# ---------------------------------------------------------------------------
# Roster builder (used by near-text resolver)
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> Optional[date]:
    try:
        return date.fromisoformat(s)
    except (ValueError, TypeError, AttributeError):
        return None


def _build_roster(bookings: List[dict], cand: NearTextCandidate) -> List[dict]:
    """Return co-attending bookings (date-overlapping) for the near-text prompt."""
    cin = _parse_date(cand.check_in or "")
    cout = _parse_date(cand.check_out or "")
    if not (cin and cout):
        return []
    roster: List[dict] = []
    for b in bookings:
        if b.get("booking_no") == cand.booking_no:
            continue
        b_cin = _parse_date(b.get("check_in", ""))
        b_cout = _parse_date(b.get("check_out", ""))
        if not (b_cin and b_cout):
            continue
        if cin < b_cout and cout > b_cin:   # ranges overlap
            sections = b.get("request", {}).get("preferred_sections", [])
            roster.append({
                "booking_no": b.get("booking_no", ""),
                "full_name": b.get("full_name", ""),
                "check_in": b.get("check_in", ""),
                "check_out": b.get("check_out", ""),
                "section": sections[0] if sections else "",
            })
            if len(roster) >= _ROSTER_MAX:
                break
    return roster
