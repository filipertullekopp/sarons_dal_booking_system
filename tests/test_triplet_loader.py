"""Tests for saronsdal.ingestion.triplet_loader and saronsdal.models.triplet."""
import io
import re
import textwrap
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import patch

import pytest

from saronsdal.ingestion.triplet_loader import (
    CAMPING_SECTIONS,
    load_triplets,
    _parse_room_id,
    _build_triplet,
)
from saronsdal.models.normalized import Spot
from saronsdal.models.triplet import Triplet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spot(section: str, spot_id: str, length_m: float = 8.0) -> Spot:
    row = re.match(r"^([A-Z]+)", spot_id).group(1)
    pos = int(re.match(r"[A-Z]+(\d+)", spot_id).group(1))
    return Spot(
        spot_id=spot_id,
        section=section,
        row=row,
        position=pos,
        length_m=length_m,
        hilliness=0,
        is_end_of_row=False,
        is_not_spot=False,
        is_reserved=False,
        no_motorhome=False,
        no_caravan_nor_motorhome=False,
        width_m=3.0,
    )


def _lookup(*spots: Spot) -> Dict[Tuple[str, str], Spot]:
    return {(s.section, s.spot_id): s for s in spots}


def _csv_file(rows: str, tmp_path: Path) -> Path:
    """Write a minimal sirvoy_room_ids.csv to a temp file."""
    content = "Rom;Fullstendig bookinginformasjon;Gjester;Rengjøring\n" + rows
    p = tmp_path / "sirvoy_room_ids.csv"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _parse_room_id: standard triplets
# ---------------------------------------------------------------------------

class TestParseRoomId:
    def test_standard_triplet_with_length(self):
        lut = _lookup(
            _spot("Furulunden", "A16", 5.0),
            _spot("Furulunden", "A17", 5.0),
            _spot("Furulunden", "A18", 5.0),
        )
        t = _parse_room_id("Furulunden A16-18 (5m)", lut)
        assert t is not None
        assert t.room_id == "Furulunden A16-18 (5m)"
        assert t.section == "Furulunden"
        assert t.row == "A"
        assert t.spot_ids == ["A16", "A17", "A18"]
        assert t.is_allocatable is True

    def test_standard_triplet_no_length_annotation(self):
        lut = _lookup(*[_spot("Bedehuset", f"A{i}") for i in range(1, 4)])
        t = _parse_room_id("Bedehuset A01-03", lut)
        assert t is not None
        assert t.spot_ids == ["A1", "A2", "A3"]
        assert t.review_flags == []

    def test_repeated_row_letter(self):
        """Bibelskolen A01-A03 format — row letter repeated before end number."""
        lut = _lookup(*[_spot("Bibelskolen", f"A{i}") for i in range(1, 4)])
        t = _parse_room_id("Bibelskolen A01-A03", lut)
        assert t is not None
        assert t.spot_ids == ["A1", "A2", "A3"]
        assert t.is_allocatable is True

    def test_space_in_range(self):
        """Vårdalen A01- 03 (15,7m) — space after hyphen."""
        lut = _lookup(*[_spot("Vårdalen", f"A{i}", 15.7) for i in range(1, 4)])
        t = _parse_room_id("Vårdalen A01- 03 (15,7m)", lut)
        assert t is not None
        assert t.spot_ids == ["A1", "A2", "A3"]
        assert t.is_allocatable is True

    def test_comma_decimal_length_annotation(self):
        """Annotation like (7,8m) is handled correctly (not parsed for length — ignored)."""
        lut = _lookup(*[_spot("Fjellterrassen", f"D{i}", 7.8) for i in range(28, 31)])
        t = _parse_room_id("Fjellterrassen D28-30 (7,8m)", lut)
        assert t is not None
        assert t.spot_ids == ["D28", "D29", "D30"]

    def test_extra_annotations_after_length(self):
        """Room strings with extra text after the length (telt, siste i rekka, etc.)"""
        lut = _lookup(*[_spot("Elvebredden", f"C{i}") for i in range(34, 37)])
        t = _parse_room_id("Elvebredden C34-36 TELT (5 m)", lut)
        assert t is not None
        assert t.spot_ids == ["C34", "C35", "C36"]


# ---------------------------------------------------------------------------
# _parse_room_id: non-standard (flagged)
# ---------------------------------------------------------------------------

class TestParseNonStandard:
    def test_two_spot_room_flagged(self):
        lut = _lookup(
            _spot("Bibelskolen", "A10"),
            _spot("Bibelskolen", "A11"),
        )
        t = _parse_room_id("Bibelskolen A10-11 ( bobil)", lut)
        assert t is not None
        assert t.is_allocatable is False
        assert any("non_triplet_spot_count_2" in f for f in t.review_flags)

    def test_four_spot_room_flagged(self):
        lut = _lookup(*[_spot("Elvebredden", f"C{i}") for i in range(37, 41)])
        t = _parse_room_id("Elvebredden C37-40 TELT (5m)", lut)
        assert t is not None
        assert t.is_allocatable is False
        assert any("non_triplet_spot_count_4" in f for f in t.review_flags)

    def test_single_spot_room_flagged(self):
        lut = _lookup(_spot("Bibelskolen", "D22"))
        t = _parse_room_id("Bibelskolen D22", lut)
        assert t is not None
        assert t.is_allocatable is False
        assert any("non_triplet_spot_count_1" in f for f in t.review_flags)


# ---------------------------------------------------------------------------
# _parse_room_id: non-camping skipped
# ---------------------------------------------------------------------------

class TestNonCampingSkipped:
    def test_campinghytte_returns_none(self):
        assert _parse_room_id("Campinghytte 1 seng1", {}) is None

    def test_internatet_returns_none(self):
        assert _parse_room_id("Internatet 101 (seng1)", {}) is None

    def test_sovesal_returns_none(self):
        assert _parse_room_id("Sovesal gutter seng01", {}) is None

    def test_helsebua_returns_none(self):
        assert _parse_room_id("Helsebua1 seng1", {}) is None

    def test_bare_row_range_returns_none(self):
        """'B01-02' has no section name — cannot be parsed as camping."""
        assert _parse_room_id("B01-02", {}) is None


# ---------------------------------------------------------------------------
# First-spot resolution from spot_lookup
# ---------------------------------------------------------------------------

class TestFirstSpotResolution:
    def test_first_spot_resolved_from_lookup(self):
        anchor = _spot("Furulunden", "A16", 5.0)
        lut = _lookup(
            anchor,
            _spot("Furulunden", "A17"),
            _spot("Furulunden", "A18"),
        )
        t = _parse_room_id("Furulunden A16-18 (5m)", lut)
        assert t.first_spot is anchor
        assert t.first_spot_length_m == pytest.approx(5.0)

    def test_length_from_spot_lookup_not_annotation(self):
        """spots.csv length takes precedence over the room_id annotation."""
        # Room says 5m, but spots.csv has 6.0m
        anchor = _spot("Furulunden", "A16", 6.0)
        lut = _lookup(anchor, _spot("Furulunden", "A17"), _spot("Furulunden", "A18"))
        t = _parse_room_id("Furulunden A16-18 (5m)", lut)
        assert t.first_spot_length_m == pytest.approx(6.0)

    def test_first_spot_missing_from_lookup(self):
        """Only A17 and A18 in lookup — first spot A16 is missing."""
        lut = _lookup(_spot("Furulunden", "A17"), _spot("Furulunden", "A18"))
        t = _parse_room_id("Furulunden A16-18 (5m)", lut)
        assert t.first_spot is None
        assert t.first_spot_length_m == 0.0
        assert t.is_allocatable is False
        assert "first_spot_missing" in t.review_flags

    def test_first_spot_id_correct(self):
        lut = _lookup(*[_spot("Egelandsletta", f"A{i}", 8.0) for i in range(1, 4)])
        t = _parse_room_id("Egelandsletta A01-03 (8m)", lut)
        assert t.first_spot_id == "A1"


# ---------------------------------------------------------------------------
# load_triplets: CSV parsing via temp file
# ---------------------------------------------------------------------------

class TestLoadTriplets:
    def test_load_two_camping_rooms(self, tmp_path):
        spots = [_spot("Furulunden", f"A{i}", 5.0) for i in range(16, 19)] + \
                [_spot("Furulunden", f"B{i}", 7.5) for i in range(1, 4)]
        lut = _lookup(*spots)
        p = _csv_file(
            "Furulunden A16-18 (5m);;;\n"
            "Furulunden B01-03 (7,5m);;;\n",
            tmp_path,
        )
        triplets = load_triplets(p, lut)
        assert len(triplets) == 2
        assert all(t.is_allocatable for t in triplets)

    def test_non_camping_rows_excluded(self, tmp_path):
        lut = _lookup(*[_spot("Furulunden", f"A{i}") for i in range(1, 4)])
        p = _csv_file(
            "Furulunden A01-03;;;\n"
            "Campinghytte 1 seng1;;;\n"
            "Internatet 101 (seng1);;;\n",
            tmp_path,
        )
        triplets = load_triplets(p, lut)
        assert len(triplets) == 1
        assert triplets[0].section == "Furulunden"

    def test_room_id_preserved_exactly(self, tmp_path):
        lut = _lookup(*[_spot("Fjellterrassen", f"D{i}", 7.8) for i in range(28, 31)])
        p = _csv_file("Fjellterrassen D28-30 (7,8m);;;\n", tmp_path)
        triplets = load_triplets(p, lut)
        assert triplets[0].room_id == "Fjellterrassen D28-30 (7,8m)"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_triplets(tmp_path / "nonexistent.csv", {})

    def test_flagged_rooms_included_but_not_allocatable(self, tmp_path):
        lut = _lookup(_spot("Bibelskolen", "A10"), _spot("Bibelskolen", "A11"))
        p = _csv_file("Bibelskolen A10-11 ( bobil);;;\n", tmp_path)
        triplets = load_triplets(p, lut)
        assert len(triplets) == 1
        assert triplets[0].is_allocatable is False


# ---------------------------------------------------------------------------
# Triplet model properties
# ---------------------------------------------------------------------------

class TestTripletModel:
    def _make(self, spot_ids, section="Furulunden", row="A", length_m=8.0):
        anchor = _spot(section, spot_ids[0], length_m)
        return Triplet(
            room_id=f"{section} {spot_ids[0]}-{spot_ids[-1]}",
            section=section,
            row=row,
            spot_ids=spot_ids,
            first_spot=anchor,
        )

    def test_is_allocatable_true_for_three_spots(self):
        t = self._make(["A1", "A2", "A3"])
        assert t.is_allocatable is True

    def test_is_allocatable_false_for_two_spots(self):
        anchor = _spot("Furulunden", "A1")
        t = Triplet(
            room_id="Furulunden A1-2",
            section="Furulunden",
            row="A",
            spot_ids=["A1", "A2"],
            first_spot=anchor,
        )
        assert t.is_allocatable is False

    def test_is_allocatable_false_when_first_spot_none(self):
        t = Triplet(
            room_id="Furulunden A1-3",
            section="Furulunden",
            row="A",
            spot_ids=["A1", "A2", "A3"],
            first_spot=None,
        )
        assert t.is_allocatable is False

    def test_first_spot_id_returns_first_element(self):
        t = self._make(["D28", "D29", "D30"])
        assert t.first_spot_id == "D28"

    def test_first_spot_length_m_from_spot(self):
        t = self._make(["A1", "A2", "A3"], length_m=7.5)
        assert t.first_spot_length_m == pytest.approx(7.5)

    def test_first_spot_length_m_zero_when_no_spot(self):
        t = Triplet(
            room_id="Furulunden A1-3",
            section="Furulunden",
            row="A",
            spot_ids=["A1", "A2", "A3"],
            first_spot=None,
        )
        assert t.first_spot_length_m == 0.0
