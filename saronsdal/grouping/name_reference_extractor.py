"""Extract person/family name references from free-text booking strings.

Produces ExtractedReference objects — these are NOT yet matched to bookings.
Matching against the booking roster happens in affinity_graph.py.

Reference types (conservative priority):
  family        — "fam. Husvik", "familien Lode"  (surname only)
  full_name     — "Otto Husvik", "Camilla Ødegård"  (first + last)
  first_name_only — single capitalised word after a proximity trigger
  organization  — multi-word capitalised string that looks like an org name
  ambiguous     — matched a pattern but cannot be classified

False-positive guards:
  - Stop-word filter: first token of a potential name must not be a known Norwegian
    stop word or preposition (even if capitalised at sentence start).
  - Section-name guard: extracted candidate is checked against the GroupNormalizer's
    section name set; if matched, the reference is discarded.
  - Minimum token length: surname / first-name tokens must be ≥ 3 characters.
  - Single capitalised word without a proximity trigger → first_name_only only
    (low confidence, never produces a hard edge on its own).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Set

import ftfy
import yaml

from saronsdal.models.grouping import ExtractedReference
from saronsdal.grouping.group_normalizer import GroupNormalizer


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_CONFIG_ROOT = Path(__file__).parent.parent / "config"


def _load_rules(rules_path: Optional[Path] = None) -> dict:
    p = rules_path or _CONFIG_ROOT / "group_rules.yaml"
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# NameReferenceExtractor
# ---------------------------------------------------------------------------

class NameReferenceExtractor:
    """
    Compile patterns once; apply to many text fragments.

    Usage:
        extractor = NameReferenceExtractor()
        refs = extractor.extract("bo med Otto Husvik og familien Lode", "guest_message")
    """

    def __init__(
        self,
        normalizer: GroupNormalizer,
        rules_path: Optional[Path] = None,
    ) -> None:
        self._normalizer = normalizer
        rules = _load_rules(rules_path)

        # Stop words (lowercased set for fast lookup)
        self._stop_words: Set[str] = {w.lower() for w in rules.get("stop_words", [])}

        # Family patterns — each must have one capture group (the surname)
        self._family_patterns = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in rules.get("family_patterns", [])
        ]

        # Full-name pattern
        fn_pat = rules.get("full_name_pattern", "")
        self._full_name_re = re.compile(fn_pat, re.UNICODE) if fn_pat else None

        # Proximity triggers (joined into one OR pattern)
        triggers = rules.get("proximity_triggers", [])
        if triggers:
            trigger_pat = "(?:" + "|".join(triggers) + ")"
            # After a trigger: capture one or two capitalised tokens
            self._trigger_re = re.compile(
                trigger_pat
                + r"\s+"
                + r"("
                + r"(?:[A-ZÆØÅ][a-zæøå]+(?:-[A-ZÆØÅ][a-zæøå]+)?)"
                + r"(?:\s+[A-ZÆØÅ][a-zæøå]+(?:-[A-ZÆØÅ][a-zæøå]+)?)?"
                + r")",
                re.IGNORECASE | re.UNICODE,
            )
        else:
            self._trigger_re = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str, source_field: str) -> List[ExtractedReference]:
        """
        Extract all name references from `text`.

        Returns a list of ExtractedReference objects.  Duplicates (same
        normalized_candidate from the same source) are deduplicated.
        """
        if not text or not text.strip():
            return []

        text = ftfy.fix_text(text).strip()
        refs: List[ExtractedReference] = []
        seen: Set[str] = set()

        def _add(raw: str, candidate: str, ref_type, confidence: float) -> None:
            key = f"{ref_type}:{candidate}"
            if key in seen:
                return
            # Section-name guard.
            if self._normalizer.is_section_name(raw):
                return
            # Stop-word guard on first token.
            first_tok = candidate.split()[0].lower() if candidate else ""
            if first_tok in self._stop_words:
                return
            # Minimum token length.
            tokens = candidate.split()
            if any(len(t) < 3 for t in tokens):
                return
            seen.add(key)
            refs.append(ExtractedReference(
                raw_text=raw,
                normalized_candidate=candidate.lower(),
                ref_type=ref_type,
                confidence=confidence,
                source_field=source_field,
            ))

        # 1. Family patterns (highest priority — most specific).
        for pattern in self._family_patterns:
            for m in pattern.finditer(text):
                surname = m.group(1).strip()
                _add(m.group(0), surname, "family", 0.85)

        # 2. Full-name pairs (after proximity trigger — high confidence).
        if self._trigger_re:
            for m in self._trigger_re.finditer(text):
                captured = m.group(1).strip()
                tokens = captured.split()
                if len(tokens) == 2:
                    # Two-token → full_name
                    _add(captured, captured, "full_name", 0.82)
                elif len(tokens) == 1:
                    # Single token after trigger → first_name_only
                    _add(captured, captured, "first_name_only", 0.40)

        # 3. Standalone full-name pairs anywhere in text (lower confidence).
        if self._full_name_re:
            for m in self._full_name_re.finditer(text):
                first_tok = m.group(1)
                last_tok  = m.group(2)
                full = f"{first_tok} {last_tok}"
                # Only emit if first token not a stop word.
                if first_tok.lower() in self._stop_words:
                    continue
                # Skip pairs that were already captured as family-pattern surnames.
                if f"family:{last_tok.lower()}" in seen:
                    continue
                _add(full, full, "full_name", 0.65)

        return refs
