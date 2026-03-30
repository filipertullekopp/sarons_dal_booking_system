"""Robust dimension parser for messy Sirvoy export strings.

All output lengths and widths are in METRES.

Handles observed real-world formats:
  "7,5"                              → length=7.5
  "9,4 m"                            → length=9.4
  "753"                              → length=7.53  (3-digit no-unit → assumed cm)
  "8650"                             → length=8.65  (4-digit no-unit → assumed mm)
  "7530 m/draget"                    → length=7.53  (strip "m/draget", then assumed mm)
  "770 cm, pluss telt 3,10 ut fra vogn" → length=7.70, fortelt_width=3.10
  "4.10x7"                           → length=7.0, width=4.10 (tent: larger=length)
  "Lenge 8m bredde 6m"               → length=8.0, width=6.0
  "Lengde 7,5 bredde 4"              → length=7.5, width=4.0
  "Camp-let"                         → length=None, flag=camplet_no_dimensions
  "9,51( med drag)"                  → length=9.51
  "Telt 640 cm"                      → length=6.40  (strip 'Telt' prefix)
  "ca 7m"                            → length=7.0,  flag=approximate
  "826m"                             → length=8.26  (>20 with unit 'm' → assumed cm typo)

Unit inference rules (applied when no explicit unit is given):
  val ≥ 1000  → divide by 1000  (mm), flag=assumed_mm
  val ≥ 100   → divide by 100   (cm), flag=assumed_cm
  val < 100   → use as-is (m)

Sanity rescue: if the parsed value ends up > 2×max_plausible (i.e. > 30 m), one
more rescaling attempt is made (/10 then /100) before giving up.  Values that are
merely above the max_plausible threshold (e.g. 17 m) are NOT auto-rescaled — they
stay with the "too_long" flag for manual review.

The parser returns a ParsedDimensions object.  Callers must check `flags` and
`confidence` before using values in hard constraints.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

import ftfy

from saronsdal.models.normalized import ParsedDimensions


# ---------------------------------------------------------------------------
# Module-level sanity bounds (must stay in sync with equipment.yaml)
# ---------------------------------------------------------------------------

_MIN_PLAUSIBLE_M: float = 1.5   # feasibility.min_plausible_length_m
_MAX_PLAUSIBLE_M: float = 15.0  # feasibility.max_plausible_length_m


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Numeric literal: digits, optional decimal separator (comma or period), more digits.
_NUM = r"\d+(?:[,\.]\d+)?"

# Unit patterns (case-insensitive).
_UNIT_M = r"(?:meter|m)\b"
_UNIT_CM = r"cm\b"
_UNIT_MM = r"mm\b"
_UNIT_ANY = rf"(?:{_UNIT_CM}|{_UNIT_MM}|{_UNIT_M})"

# Combined: number optionally followed by a unit.
_NUM_WITH_OPT_UNIT = rf"({_NUM})\s*({_UNIT_ANY})?"

# Norwegian trailer-annotation patterns where "m" is NOT a unit marker.
# Examples: "7530 m/draget", "8.5 m. drag", "m/drag"
_TRAILER_ANNOTATION = re.compile(
    r"\bm[/\.]\s*drag(?:et)?\b",
    re.IGNORECASE,
)


def _to_float(s: str) -> float:
    """Parse a numeric string that may use comma as decimal separator."""
    return float(s.replace(",", "."))


def _apply_unit(val: float, unit: str) -> Tuple[float, str]:
    """
    Convert ``val`` to metres based on ``unit`` string.

    Unit inference rules (no explicit unit):
      val ≥ 1000  → mm  (e.g. 8650 → 8.65 m)
      val ≥ 100   → cm  (e.g. 753  → 7.53 m)
      val < 100   → m   (used as-is)

    Returns:
        (value_in_metres, flag_name)  — flag_name is "" when no inference was needed.
    """
    u = unit.lower().strip()

    if "mm" in u:
        return val / 1000.0, ""
    if "cm" in u:
        return val / 100.0, ""
    if u in ("m", "meter"):
        # Stated as metres but suspiciously large → almost certainly a cm typo.
        if val > 20.0:
            return val / 100.0, "assumed_cm_typo"
        return val, ""

    # No unit: infer from order of magnitude.
    if val >= 1000:
        # 4+ digit bare value (e.g. 8650) → millimetres
        return val / 1000.0, "assumed_mm"
    if val >= 100:
        # 3-digit bare value (e.g. 753) → centimetres
        return val / 100.0, "assumed_cm"
    return val, ""


def _parse_single_value(text: str) -> Tuple[Optional[float], list]:
    """
    Extract one numeric value from text and convert to metres.

    Returns:
        (value_in_metres or None, flags_list)
    """
    flags = []

    # Strip common label words that are not units.
    cleaned = re.sub(
        r"\b(telt|bredde|lengde|lenge|lang|fortelt|drag|med drag|"
        r"totalt|inkl\.?|ca\.?|tror|omtrent)\b",
        " ", text, flags=re.IGNORECASE,
    ).strip()

    # Find first number + optional unit.
    m = re.search(rf"({_NUM})\s*({_UNIT_ANY})?", cleaned, re.IGNORECASE)
    if not m:
        return None, ["parse_failed"]

    try:
        val = _to_float(m.group(1))
    except ValueError:
        return None, ["parse_failed"]

    unit = (m.group(2) or "").strip()
    val_m, inferred_flag = _apply_unit(val, unit)

    if inferred_flag:
        flags.append(inferred_flag)

    return val_m, flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_dimensions(raw_length: str, raw_width: str = "") -> ParsedDimensions:
    """
    Parse vehicle/tent dimension strings from Sirvoy booking exports.

    Args:
        raw_length: The contents of the length/dimension field.
        raw_width:  The contents of the separate width field (if present).
                    May be empty string when the export format lacks a width column.

    Returns:
        ParsedDimensions with all values in metres.  Caller must inspect
        `.confidence` and `.flags` before using values in hard constraints.
    """
    flags: list = []
    confidence = 1.0

    # 1. Encoding repair and normalise whitespace.
    raw_length = ftfy.fix_text(str(raw_length)).strip() if raw_length else ""
    raw_width = ftfy.fix_text(str(raw_width)).strip() if raw_width else ""

    # 2. Treat zero / empty as missing.
    if not raw_length or raw_length in ("0", "0.0", "0,0", "-", "nan", "none"):
        return ParsedDimensions(
            length_m=None,
            width_m=None,
            fortelt_width_m=None,
            confidence=0.0,
            flags=["missing_dimensions"],
        )

    # 3. Camp-let: a named equipment type with no numeric dimensions.
    if re.search(r"\bcamp.?let\b", raw_length, re.IGNORECASE):
        return ParsedDimensions(
            length_m=None,
            width_m=None,
            fortelt_width_m=None,
            confidence=0.5,
            flags=["camplet_no_dimensions"],
        )

    length_m: Optional[float] = None
    width_m: Optional[float] = None
    fortelt_width_m: Optional[float] = None

    # 4. Strip Norwegian trailer annotations ("m/draget", "m. drag") BEFORE unit
    #    matching.  These abbreviations for "med draget" (with hitch) contain a
    #    leading "m" that the unit regex would otherwise misread as "metres".
    working = _TRAILER_ANNOTATION.sub("", raw_length).strip()

    # 5. Extract fortelt lateral width from patterns like:
    #    "770 cm, pluss telt 3,10 ut fra vogn"
    #    "770cm + fortelt 3.1m"
    fortelt_pattern = re.search(
        rf"(?:pluss|[+])\s*(?:telt|fortelt)\s*({_NUM})\s*({_UNIT_ANY})?",
        working, re.IGNORECASE,
    )
    if fortelt_pattern:
        try:
            fw_raw = _to_float(fortelt_pattern.group(1))
            fw_unit = (fortelt_pattern.group(2) or "").strip()
            fortelt_width_m, _ = _apply_unit(fw_raw, fw_unit)
            flags.append("fortelt_dimension_parsed")
        except ValueError:
            pass
        # Remove the fortelt portion so it does not confuse the main parse.
        working = working[: fortelt_pattern.start()].rstrip(" ,+")

    # 6. Try labeled format: "Lenge Xm bredde Ym" / "Lengde X bredde Y"
    labeled = re.search(
        rf"(?:leng(?:e|de)?|l)\s*[:\s]*({_NUM})\s*({_UNIT_ANY})?"
        rf"[\s,;]*(?:bredde|b)\s*[:\s]*({_NUM})\s*({_UNIT_ANY})?",
        working, re.IGNORECASE,
    )
    if labeled:
        try:
            lv = _to_float(labeled.group(1))
            lu = (labeled.group(2) or "").strip()
            wv = _to_float(labeled.group(3))
            wu = (labeled.group(4) or "").strip()
            length_m, lf = _apply_unit(lv, lu)
            width_m, wf = _apply_unit(wv, wu)
            if lf:
                flags.append(lf)
            if wf:
                flags.append(wf)
            flags.append("labeled_dimensions")
            confidence -= 0.05
        except ValueError:
            pass

    # 7. Try cross-dimension format: "4.10x7", "7 x 5m", "4,1 x 7 m"
    if length_m is None:
        cross = re.search(
            rf"({_NUM})\s*({_UNIT_ANY})?\s*[xX×]\s*({_NUM})\s*({_UNIT_ANY})?",
            working,
        )
        if cross:
            try:
                v1 = _to_float(cross.group(1))
                u1 = (cross.group(2) or "").strip()
                v2 = _to_float(cross.group(3))
                u2 = (cross.group(4) or "").strip()
                m1, f1 = _apply_unit(v1, u1)
                m2, f2 = _apply_unit(v2, u2)
                if f1:
                    flags.append(f1)
                if f2:
                    flags.append(f2)
                # For tents: smaller = width, larger = length (tent can be rotated).
                length_m = max(m1, m2)
                width_m = min(m1, m2)
                flags.append("two_dimensions_parsed")
                confidence -= 0.05
            except ValueError:
                pass

    # 8. Parse the separate width field if present and width not yet determined.
    if raw_width and raw_width not in ("0", "0.0", "-", "nan", "none"):
        w_val, w_flags = _parse_single_value(raw_width)
        if w_val is not None and width_m is None:
            width_m = w_val
            flags.extend(w_flags)

    # 9. Parse primary length from the working string if still missing.
    if length_m is None:
        # Strip type-hint prefixes (e.g. "Telt 640 cm" → "640 cm").
        stripped = re.sub(
            r"^\s*(?:telt|bobil|campingvogn|vogn|fortelt)\s*",
            "", working, flags=re.IGNORECASE,
        ).strip()
        # Strip parenthetical labels ("(med drag)", "(inkl. fortelt)").
        stripped = re.sub(r"\([^)]*\)", " ", stripped).strip()

        val, pf = _parse_single_value(stripped)
        flags.extend(pf)
        if val is not None:
            length_m = val
        else:
            # Last resort: try the original working string.
            val2, pf2 = _parse_single_value(working)
            flags.extend(pf2)
            length_m = val2

    # 10. Approximate qualifier.
    if re.search(r"\bca\.?\b|\btror\b|\bomtrent\b|\bungefær\b", raw_length, re.IGNORECASE):
        flags.append("approximate")
        confidence -= 0.10

    # 11. Sanity checks on final values.
    if length_m is not None:
        if length_m < _MIN_PLAUSIBLE_M:
            flags.append("too_short")
            confidence -= 0.20
        elif length_m > _MAX_PLAUSIBLE_M:
            flags.append("too_long")
            confidence -= 0.20

    # 11b. Sanity rescue: if the value is *clearly* unrealistic (more than 2× the
    #      maximum plausible length), attempt rescaling by /10 then /100.
    #      Values that are merely above the threshold (e.g. 17 m) are intentionally
    #      left untouched — they carry a "too_long" review flag for manual inspection.
    if length_m is not None and length_m > 2 * _MAX_PLAUSIBLE_M:
        for divisor, rflag in ((10, "rescaled_div10"), (100, "rescaled_div100")):
            candidate = round(length_m / divisor, 3)
            if _MIN_PLAUSIBLE_M <= candidate <= _MAX_PLAUSIBLE_M:
                length_m = candidate
                # "too_long" is no longer accurate after rescaling.
                flags = [f for f in flags if f != "too_long"]
                flags.extend([rflag, "rescaled_to_plausible"])
                confidence -= 0.25
                break

    if "parse_failed" in flags:
        flags.append("complex_dimension_string")
        confidence -= 0.30

    # 12. Complex raw string hint (many non-numeric characters).
    non_numeric_ratio = len(re.sub(r"[\d,\. ]", "", raw_length)) / max(len(raw_length), 1)
    if non_numeric_ratio > 0.4 and "complex_dimension_string" not in flags:
        flags.append("complex_dimension_string")
        confidence -= 0.10

    confidence = round(max(0.0, min(1.0, confidence)), 2)

    return ParsedDimensions(
        length_m=round(length_m, 3) if length_m is not None else None,
        width_m=round(width_m, 3) if width_m is not None else None,
        fortelt_width_m=round(fortelt_width_m, 3) if fortelt_width_m is not None else None,
        confidence=confidence,
        flags=flags,
    )
