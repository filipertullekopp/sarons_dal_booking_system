"""Phase 3 — Topology loader.

Parses all topology grid CSV files into spatial indices.

Grid coordinate convention:
  (x = column_index, y = row_index), both 0-based.
  Every cell in the grid occupies exactly one unit of space.
  EMPTY cells are real physical space — they are NOT missing data.
  Distances between entities are computed from their (x, y) grid coordinates,
  so blank rows and EMPTY cells correctly contribute to the distance.

Section name handling:
  Bibelskolen (topology name) is normalised to Internatet (canonical name).
  All other section names are preserved as-is.

Spot ID normalisation:
  Topology files use zero-padded numbers ("C01", "D22").
  The spots.csv convention is un-padded ("C1", "D22").
  This loader strips leading zeros so IDs match the spots.csv format:
      "C01" → "C1",  "B38" → "B38",  "E07" → "E7"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import ftfy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Known section names as they appear in the topology CSVs
# ---------------------------------------------------------------------------

_KNOWN_SECTIONS: Tuple[str, ...] = (
    "Elvebredden",
    "Furulunden",
    "Furutoppen",
    "Fjellterrassen",
    "Bibelskolen",
    "Bedehuset",
    "Vårdalen",
    "Egelandsletta",
    # Encoding-fallback variants (latin-1 decoded as "wrong" UTF-8, repaired by ftfy)
    "V\u00e5rdalen",   # same as Vårdalen after ftfy
)

# Canonical section name mapping (topology name → canonical name)
_SECTION_NORMALISE: Dict[str, str] = {
    "Bibelskolen": "Internatet",
}

# ---------------------------------------------------------------------------
# Landmark recognition and normalisation
# ---------------------------------------------------------------------------

# Maps lower-cased, stripped cell text → canonical landmark type
_LANDMARK_MAP: Dict[str, str] = {
    "river":                    "river",
    "road":                     "road",
    "main road":                "main_road",
    "fire safety zone (3m)":    "fire_safety",
    "fire safety (3m)":         "fire_safety",
    "fire safety":              "fire_safety",
    "toilet":                   "toilet",
    "toilets":                  "toilet",
    "shower":                   "shower",
}

# ---------------------------------------------------------------------------
# Default topology file names
# ---------------------------------------------------------------------------

#: Maps a short grid identifier to the CSV filename.
#: Vårdalen and Egelandsletta share one file ("vaardalen_egeland") —
#: spots from both sections appear in the same coordinate space.
TOPOLOGY_FILENAMES: Dict[str, str] = {
    "elvebredden":        "topology_grid_elvebredden.csv",
    "furulunden":         "topology_grid_furulunden.csv",
    "fjellterrassen":     "topology_grid_fjellterrassen.csv",
    "furutoppen":         "topology_grid_furutoppen.csv",
    "bibelskolen":        "topology_grid_bibelskolen.csv",
    "bedehuset":          "topology_grid_bedehuset.csv",
    "vaardalen_egeland":  "topology_grid_v\u00e5rdalen_egeland.csv",  # topology_grid_vårdalen_egeland.csv
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridCoord:
    """An entity's position in a topology grid."""
    x: int    # column index, 0-based
    y: int    # row index, 0-based
    grid: str # grid identifier (key from TOPOLOGY_FILENAMES)


@dataclass
class Topology:
    """Spatial index built from all topology CSV files.

    spot_index:      (section, spot_id) → GridCoord
    landmark_index:  landmark_type      → [GridCoord, ...]
    """
    spot_index: Dict[Tuple[str, str], GridCoord] = field(default_factory=dict)
    landmark_index: Dict[str, List[GridCoord]] = field(default_factory=dict)
    # (grid_id → (width, height)) for debugging
    grid_dimensions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # Cells whose content was not recognised (grid, text, x, y)
    unrecognised_cells: List[Tuple[str, str, int, int]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_spot_coord(self, section: str, spot_id: str) -> Optional[GridCoord]:
        """Return the GridCoord for a spot, or None if not in topology."""
        return self.spot_index.get((section, spot_id))

    def get_landmark_coords(self, landmark_type: str) -> List[GridCoord]:
        """Return all GridCoords for a landmark type (empty list if none)."""
        return self.landmark_index.get(landmark_type, [])

    def spot_keys(self) -> List[Tuple[str, str]]:
        """All (section, spot_id) keys in the topology."""
        return list(self.spot_index.keys())

    def spots_in_section(self, section: str) -> List[Tuple[str, GridCoord]]:
        """All (spot_id, GridCoord) pairs for a section."""
        return [
            (sid, coord)
            for (sec, sid), coord in self.spot_index.items()
            if sec == section
        ]

    def sections(self) -> List[str]:
        """All distinct section names present in the topology."""
        return sorted({sec for sec, _ in self.spot_index})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SPOT_ID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def normalise_spot_id(raw: str) -> str:
    """Strip leading zeros from the numeric suffix to match spots.csv convention.

    Examples:
        "C01"  → "C1"
        "B38"  → "B38"
        "E07"  → "E7"
        "D22"  → "D22"
        "A1"   → "A1"   (already un-padded)
    """
    m = _SPOT_ID_RE.match(raw.strip())
    if not m:
        return raw
    letters = m.group(1).upper()
    number = int(m.group(2))   # int() strips leading zeros
    return f"{letters}{number}"


def _classify_cell(cell: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Classify a cell string.

    Returns:
        ("spot",     section, spot_id)  — spot cell
        ("landmark", lm_type, None)     — landmark cell
        ("empty",    None, None)        — empty / EMPTY / EMTPY cell
        ("unknown",  cell_text, None)   — unrecognised content
    """
    if not cell or cell.upper() in ("EMPTY", "EMTPY"):
        return ("empty", None, None)

    # Spot: starts with a known section name followed by a space
    for section in _KNOWN_SECTIONS:
        if cell.startswith(section + " "):
            raw_id = cell[len(section):].strip()
            if raw_id:
                canonical_section = _SECTION_NORMALISE.get(section, section)
                return ("spot", canonical_section, normalise_spot_id(raw_id))

    # Landmark
    lm_type = _LANDMARK_MAP.get(cell.lower())
    if lm_type:
        return ("landmark", lm_type, None)

    return ("unknown", cell, None)


def _read_grid(path: Path) -> List[List[str]]:
    """Read a topology CSV as a raw grid (list of rows of stripped cell strings).

    Tries UTF-8-sig first (handles BOM), then latin-1 (handles Windows-1252).
    Applies ftfy encoding repair to each cell so mojibake section names are
    recovered (e.g. "V\u00e5rdalen" from a latin-1 encoded 'å').
    """
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as fh:
                content = fh.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Cannot decode {path} with any supported encoding")

    rows = []
    for line in content.splitlines():
        cells = [ftfy.fix_text(c.strip()) for c in line.split(";")]
        rows.append(cells)
    return rows


def _parse_grid_file(
    path: Path,
    grid_name: str,
    spot_index: Dict[Tuple[str, str], GridCoord],
    landmark_index: Dict[str, List[GridCoord]],
    unrecognised: List[Tuple[str, str, int, int]],
) -> Tuple[int, int]:
    """Parse one grid CSV file, mutating spot_index and landmark_index.

    Returns:
        (max_x, max_y) — grid bounds (number of columns and rows encountered).
    """
    rows = _read_grid(path)
    max_x = 0
    max_y = 0

    for row_idx, cells in enumerate(rows):
        for col_idx, cell in enumerate(cells):
            max_x = max(max_x, col_idx)
            max_y = max(max_y, row_idx)

            kind, a, b = _classify_cell(cell)

            if kind == "empty":
                continue

            coord = GridCoord(x=col_idx, y=row_idx, grid=grid_name)

            if kind == "spot":
                section, spot_id = a, b
                key = (section, spot_id)
                if key in spot_index:
                    existing = spot_index[key]
                    logger.warning(
                        "Duplicate spot key %s in grid %s at (%d,%d) — "
                        "already registered in grid %s at (%d,%d); keeping first",
                        key, grid_name, col_idx, row_idx,
                        existing.grid, existing.x, existing.y,
                    )
                else:
                    spot_index[key] = coord

            elif kind == "landmark":
                lm_type = a
                landmark_index.setdefault(lm_type, []).append(coord)

            else:  # unknown
                unrecognised.append((grid_name, a, col_idx, row_idx))
                logger.debug(
                    "Unrecognised cell in grid %s at (%d,%d): %r",
                    grid_name, col_idx, row_idx, a,
                )

    return max_x, max_y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_topology(
    data_dir: Path,
    filenames: Optional[Dict[str, str]] = None,
) -> Topology:
    """Load all topology CSV files and return a Topology spatial index.

    Args:
        data_dir:  Directory that contains the topology CSV files.
        filenames: Override the default {grid_id → filename} mapping.
                   Useful for testing with smaller fixture files.

    Returns:
        Topology populated with spot and landmark coordinates.

    Notes:
        - Files that do not exist are skipped with a warning.
        - Unrecognised cell content is logged at DEBUG level and accumulated
          in Topology.unrecognised_cells for inspection.
        - spot_metadata.csv (if present) is NOT loaded automatically; it
          duplicates topology_grid_elvebredden.csv content.
    """
    fn_map = filenames if filenames is not None else TOPOLOGY_FILENAMES
    topo = Topology()

    for grid_name, filename in fn_map.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning("Topology file not found, skipping: %s", path)
            continue

        try:
            max_x, max_y = _parse_grid_file(
                path, grid_name,
                topo.spot_index, topo.landmark_index, topo.unrecognised_cells,
            )
            topo.grid_dimensions[grid_name] = (max_x + 1, max_y + 1)
            n_spots = sum(1 for c in topo.spot_index.values() if c.grid == grid_name)
            logger.info(
                "Grid %-22s loaded from %-50s  spots=%d  dims=%dx%d",
                grid_name, filename, n_spots, max_x + 1, max_y + 1,
            )
        except Exception as exc:
            logger.error("Failed to load grid %s (%s): %s", grid_name, filename, exc)

    if topo.unrecognised_cells:
        logger.warning(
            "%d unrecognised cell(s) across all grids; "
            "inspect Topology.unrecognised_cells for details",
            len(topo.unrecognised_cells),
        )

    logger.info(
        "Topology loaded: %d spots across %d section(s), %d landmark type(s)",
        len(topo.spot_index),
        len(topo.sections()),
        len(topo.landmark_index),
    )
    return topo
