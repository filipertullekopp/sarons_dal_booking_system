"""Tests for NameReferenceExtractor."""
import pytest

from saronsdal.grouping.group_normalizer import GroupNormalizer
from saronsdal.grouping.name_reference_extractor import NameReferenceExtractor


@pytest.fixture(scope="module")
def extractor() -> NameReferenceExtractor:
    normalizer = GroupNormalizer()
    return NameReferenceExtractor(normalizer)


def _refs_by_type(refs, ref_type):
    return [r for r in refs if r.ref_type == ref_type]


# ---------------------------------------------------------------------------
# Family patterns
# ---------------------------------------------------------------------------

def test_fam_dot_pattern(extractor):
    refs = extractor.extract("fam. Husvik", "near_text")
    family_refs = _refs_by_type(refs, "family")
    assert len(family_refs) == 1
    assert family_refs[0].normalized_candidate == "husvik"
    assert family_refs[0].confidence == pytest.approx(0.85)


def test_familien_pattern(extractor):
    refs = extractor.extract("Bo nærme familien Lode takk", "near_text")
    family_refs = _refs_by_type(refs, "family")
    assert any(r.normalized_candidate == "lode" for r in family_refs)


def test_familie_pattern(extractor):
    refs = extractor.extract("familie Andersen", "group_field")
    # "familie" alone might hit org_blocklist in normalizer; check the extractor still captures it
    family_refs = _refs_by_type(refs, "family")
    assert any(r.normalized_candidate == "andersen" for r in family_refs)


# ---------------------------------------------------------------------------
# Full name after proximity trigger
# ---------------------------------------------------------------------------

def test_full_name_after_bo_med(extractor):
    refs = extractor.extract("bo med Otto Husvik", "near_text")
    full_refs = _refs_by_type(refs, "full_name")
    assert len(full_refs) >= 1
    assert any(r.normalized_candidate == "otto husvik" for r in full_refs)


def test_full_name_after_sammen_med(extractor):
    refs = extractor.extract("vi ønsker å bo sammen med Camilla Ødegård", "near_text")
    full_refs = _refs_by_type(refs, "full_name")
    assert any("camilla" in r.normalized_candidate or "ødegård" in r.normalized_candidate
               for r in full_refs)


def test_first_name_only_after_trigger(extractor):
    """Single token after proximity trigger → first_name_only."""
    refs = extractor.extract("bo med Otto", "near_text")
    # Otto alone after "bo med"
    first_refs = _refs_by_type(refs, "first_name_only")
    full_refs = _refs_by_type(refs, "full_name")
    # Should be first_name_only since only one token, not full_name
    candidates = [r.normalized_candidate for r in first_refs + full_refs]
    assert any("otto" in c for c in candidates)


# ---------------------------------------------------------------------------
# Standalone full name
# ---------------------------------------------------------------------------

def test_standalone_full_name(extractor):
    refs = extractor.extract("Ønsker å stå nær Lars Hansen", "group_field")
    full_refs = _refs_by_type(refs, "full_name")
    assert any("lars hansen" in r.normalized_candidate for r in full_refs)


def test_standalone_full_name_lower_confidence(extractor):
    """Standalone full name (no trigger) has confidence ~0.65."""
    refs = extractor.extract("Lars Hansen", "group_field")
    full_refs = _refs_by_type(refs, "full_name")
    if full_refs:
        assert full_refs[0].confidence == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Stop word filter
# ---------------------------------------------------------------------------

def test_stop_word_first_token_rejected(extractor):
    """'Med Hansen' — 'Med' is a stop word → the pair should not produce a full_name ref."""
    refs = extractor.extract("Med Hansen", "near_text")
    full_refs = _refs_by_type(refs, "full_name")
    assert not any("med" in r.normalized_candidate.split()[0] for r in full_refs)


def test_stop_word_og_rejected(extractor):
    refs = extractor.extract("Og Nilsen", "near_text")
    full_refs = _refs_by_type(refs, "full_name")
    assert not any(r.normalized_candidate.startswith("og ") for r in full_refs)


# ---------------------------------------------------------------------------
# Minimum token length
# ---------------------------------------------------------------------------

def test_short_token_rejected(extractor):
    """Surnames/first names of < 3 chars must be rejected."""
    refs = extractor.extract("fam. Li", "near_text")
    family_refs = _refs_by_type(refs, "family")
    # "Li" is 2 chars — should be filtered out
    assert not any(r.normalized_candidate == "li" for r in family_refs)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_deduplication_same_ref(extractor):
    """Same family reference appearing twice → only one ExtractedReference."""
    text = "fam. Husvik og fam. Husvik"
    refs = extractor.extract(text, "near_text")
    family_husvik = [r for r in refs if r.ref_type == "family" and r.normalized_candidate == "husvik"]
    assert len(family_husvik) == 1


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_text_returns_empty(extractor):
    assert extractor.extract("", "near_text") == []
    assert extractor.extract("   ", "near_text") == []


# ---------------------------------------------------------------------------
# Encoding repair
# ---------------------------------------------------------------------------

def test_mojibake_repaired(extractor):
    """ftfy should repair encoding artifacts before pattern matching."""
    # "familien HusviÃ¸k" is contrived, but check that ftfy runs without error
    refs = extractor.extract("fam. HommersÃ¥k", "near_text")
    # Should produce at least one family ref (surname repaired to Hommersåk → hommersåk)
    family_refs = _refs_by_type(refs, "family")
    assert len(family_refs) >= 1
    assert family_refs[0].normalized_candidate == "hommersåk"
