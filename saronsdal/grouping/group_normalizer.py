"""Normalize noisy organization/group label strings.

Normalization pipeline (in order):
  1. ftfy encoding repair
  2. Strip leading/trailing whitespace
  3. Reject empty / too-short strings
  4. Reject private_labels (from group_aliases.yaml)
  5. Reject org_blocklist generics (from group_rules.yaml)
  6. Reject section names (from sections.yaml) → flag "group_field_is_section"
  7. Exact alias lookup (case-insensitive) on the raw form
  8. Noise-word stripping, then exact alias lookup again
  9. If no alias hit, return a normalized free-form label with lower confidence

Returns None for inputs that should never produce a group node (empty, private,
blocklisted, or section name).

Thread-safety: GroupNormalizer instances are immutable after __init__.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import ftfy
import yaml


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class NormalizedLabel:
    canonical: str          # the canonical form to use as graph node id
    confidence: float       # 1.0 = exact alias hit; 0.70 = free-form normalized
    alias_key: Optional[str]  # group_aliases.yaml key, or None for free-form


# ---------------------------------------------------------------------------
# GroupNormalizer
# ---------------------------------------------------------------------------

class GroupNormalizer:
    """
    Stateful normalizer that caches config on first use.

    Usage:
        normalizer = GroupNormalizer()
        label = normalizer.normalize("Betel HommersÃ¥k gjengen")
        # → NormalizedLabel(canonical="Betel Hommersåk", confidence=1.0, alias_key="betel_hommersaak")
    """

    _CONFIG_ROOT = Path(__file__).parent.parent / "config"

    def __init__(
        self,
        aliases_path: Optional[Path] = None,
        sections_path: Optional[Path] = None,
        rules_path: Optional[Path] = None,
    ) -> None:
        ap = aliases_path or self._CONFIG_ROOT / "group_aliases.yaml"
        sp = sections_path or self._CONFIG_ROOT / "sections.yaml"
        rp = rules_path   or self._CONFIG_ROOT / "group_rules.yaml"

        with open(ap, encoding="utf-8") as fh:
            alias_cfg = yaml.safe_load(fh)
        with open(sp, encoding="utf-8") as fh:
            section_cfg = yaml.safe_load(fh)
        with open(rp, encoding="utf-8") as fh:
            rules_cfg = yaml.safe_load(fh)

        # private_labels: should be treated as no-org
        self._private: Set[str] = {
            v.lower().strip()
            for v in alias_cfg.get("private_labels", [])
        }

        # org_blocklist from rules
        self._blocklist: Set[str] = {
            w.lower().strip()
            for w in rules_cfg.get("org_blocklist", [])
        }

        # section names (canonical + all aliases, lowercased)
        self._section_names: Set[str] = set()
        for entry in section_cfg.get("sections", {}).values():
            self._section_names.add(entry["canonical"].lower())
            for alias in entry.get("aliases", []):
                self._section_names.add(alias.lower())
        # also the explicit section_name_set
        for s in section_cfg.get("section_name_set", []):
            self._section_names.add(s.lower())

        # alias lookup: normalized alias string → (canonical, alias_key)
        self._alias_lookup: Dict[str, tuple] = {}
        for key, entry in alias_cfg.get("organizations", {}).items():
            canonical = entry["canonical"]
            for alias in entry.get("aliases", []):
                repaired = ftfy.fix_text(alias).strip().lower()
                self._alias_lookup[repaired] = (canonical, key)

        # noise words for stripping
        self._noise_words: List[str] = [
            w.lower() for w in rules_cfg.get("noise_words", [])
        ]
        # Build a single regex that strips any noise word at word boundaries
        if self._noise_words:
            pattern = r"\b(?:" + "|".join(re.escape(w) for w in self._noise_words) + r")\b"
            self._noise_re = re.compile(pattern, re.IGNORECASE)
        else:
            self._noise_re = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, raw: str, source_field: str = "") -> Optional[NormalizedLabel]:
        """
        Normalize a raw label string.

        Returns None if the label is:
          - empty / whitespace only
          - a private label ("privat", "-", etc.)
          - in the org blocklist ("familie", "venner", etc.)
          - a section name ("Fjellterrassen", "Furulunden", etc.)
        """
        if not raw or not raw.strip():
            return None

        repaired = ftfy.fix_text(raw).strip()
        lower = repaired.lower()

        # Reject private and blocklisted labels.
        if lower in self._private or lower in self._blocklist:
            return None

        # Reject section names.
        if lower in self._section_names:
            return None

        # Try exact alias lookup on the repaired form.
        result = self._alias_lookup.get(lower)
        if result:
            canonical, key = result
            return NormalizedLabel(canonical=canonical, confidence=1.0, alias_key=key)

        # Strip noise words and try again.
        if self._noise_re:
            stripped = self._noise_re.sub("", repaired).strip()
            stripped_lower = stripped.lower()
            if stripped_lower and stripped_lower != lower:
                result = self._alias_lookup.get(stripped_lower)
                if result:
                    canonical, key = result
                    return NormalizedLabel(canonical=canonical, confidence=0.95, alias_key=key)

        # No alias hit → return a free-form normalized label.
        # Normalize: title-case, collapse whitespace, preserve original ftfy output.
        free_form = " ".join(repaired.split())
        if len(free_form) < 2:
            return None
        return NormalizedLabel(canonical=free_form, confidence=0.70, alias_key=None)

    def is_section_name(self, raw: str) -> bool:
        """Return True if raw (after ftfy repair) matches a known section name."""
        if not raw:
            return False
        return ftfy.fix_text(raw).strip().lower() in self._section_names
