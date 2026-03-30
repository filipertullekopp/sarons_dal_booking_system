"""Diagnostic CLI for the near-ref / group-context allocation pipeline.

Loads the same inputs as the allocate CLI but does NOT run allocation.
Instead it prints a detailed snapshot of the social-signal pipeline for
one or more probe booking numbers, so you can trace exactly where a
near-ref or cluster link is present, filtered, or missing.

Usage:
    python -m saronsdal.cli.diagnose \\
        --bookings  output/bookings_normalized.json \\
        --groups    output/resolved_groups.json \\
        --refs      output/llm_suggestions/reference_resolutions.jsonl \\
        --probe     30030 30058

Optional (for group-context detail):
        --subsec    output/llm_suggestions/subsection_resolutions.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _path_status(path, loaded, label: str) -> None:
    """Print one load-summary line for an optional input file."""
    if path is None:
        print(f"  {label} : (not provided)")
    elif not path.exists():
        print(f"  {label} : FILE NOT FOUND — {path}")
    else:
        print(f"  {label} : {len(loaded)} entries from {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose near-ref / cluster pipeline for specific bookings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bookings", required=True, type=Path)
    parser.add_argument("--groups",   type=Path, default=None)
    parser.add_argument("--refs",     type=Path, default=None)
    parser.add_argument("--subsec",   type=Path, default=None)
    parser.add_argument("--probe",    nargs="+", required=True,
                        help="One or more booking_nos to inspect")
    parser.add_argument("--debug",    action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Reuse the loaders from the allocate CLI
    from saronsdal.cli.allocate import (
        _load_bookings,
        _load_clusters,
        _load_near_refs,
        _load_subsection_suggestions,
    )
    from saronsdal.allocation.allocator import diagnose_pipeline

    bookings  = _load_bookings(args.bookings)
    clusters  = _load_clusters(args.groups)
    near_refs = _load_near_refs(args.refs)
    subsecs   = _load_subsection_suggestions(args.subsec)

    # ── Load summary (visible before any probe output) ──────────────────────
    print("=" * 70)
    print("LOAD SUMMARY")
    print("=" * 70)
    print(f"  bookings  : {len(bookings)} loaded from {args.bookings}")
    _path_status(args.groups,  clusters,  "--groups")
    _path_status(args.refs,    near_refs, "--refs  ")
    _path_status(args.subsec,  subsecs,   "--subsec")
    probe_bnos_in_refs = {s.booking_no for s in near_refs}
    missing_from_refs  = [b for b in args.probe if b not in probe_bnos_in_refs]
    if near_refs and missing_from_refs:
        print(f"\n  WARNING: probe booking(s) {missing_from_refs} "
              f"have NO entry in the near-refs file.")
    elif not near_refs and args.refs:
        print(f"\n  WARNING: --refs was provided but loaded 0 entries. "
              f"Check that the path is correct.")
    print()

    report = diagnose_pipeline(
        bookings=bookings,
        probe_booking_nos=args.probe,
        clusters=clusters or None,
        subsection_suggestions=subsecs or None,
        near_refs=near_refs or None,
    )

    # Pretty-print the report for each probe booking
    for bno, data in report.items():
        _print_probe(bno, data)


def _print_probe(bno: str, data: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"PROBE: booking {bno}  —  {data['full_name']}")
    print(f"  constraint_strength : {data['constraint_strength']}")
    print(sep)

    # ── [1] Phase 2 cluster membership ─────────────────────────────────────
    gm = data["group_map_entry"]
    print(f"\n[1] group_map[\"{bno}\"] ({len(gm)} co-member(s)):")
    if gm:
        for m in gm:
            print(f"      {m}")
    else:
        print("      (empty — not in any Phase 2 cluster)")

    # ── [2] near_ref_map (what _build_near_ref_map produced) ───────────────
    nr = data["near_ref_map_entry"]
    print(f"\n[2] near_ref_map[\"{bno}\"] ({len(nr)} target(s)):")
    if nr:
        for t in nr:
            print(f"      → {t}")
    else:
        print("      (empty — no targets resolved from resolved_refs or place_refs)")

    # ── [2b] Is this booking a target of someone else? ──────────────────────
    is_target_of = data["is_near_ref_target_of"]
    if is_target_of:
        print(f"      (is ITSELF a near-ref target of: {is_target_of})")

    # ── [3] Raw NearTextSuggestion data ────────────────────────────────────
    raw = data["raw_near_ref_suggestion"]
    if isinstance(raw, str):
        print(f"\n[3] NearTextSuggestion: {raw}")
    else:
        print(f"\n[3] NearTextSuggestion for {bno}:")
        rr = raw.get("resolved_refs", [])
        print(f"      resolved_refs ({len(rr)} entry(s)):")
        for r in rr:
            status = "OK" if r["matched_booking_no"] else "null"
            print(f"        [{status}] fragment={r['raw_fragment']!r:40s}  "
                  f"type={r['match_type']:18s}  conf={r['confidence']:.2f}  "
                  f"matched={r['matched_booking_no']}")

        pc = raw.get("place_refs_count", 0)
        prs = raw.get("place_refs", [])
        print(f"      place_refs ({pc} entry(s)):")
        if prs:
            for pr in prs:
                if not isinstance(pr, dict):
                    print(f"        [BAD ENTRY — not a dict: {pr!r}]")
                    continue
                nos = pr.get("matched_booking_nos", [])
                print(f"        type={pr.get('match_type'):18s}  "
                      f"conf={pr.get('confidence', '?'):.2f}  "
                      f"targets({len(nos)}): {nos}  "
                      f"frag={pr.get('fragment', '')!r}")
        else:
            print("        (none)")

    # ── [4] Group context ───────────────────────────────────────────────────
    print(f"\n[4] group_context_map.get(\"{bno}\"):")
    if not data["group_context_present"]:
        print("      NOT PRESENT — seeding will not activate for this booking")
        # Diagnose why
        if not data["group_map_entry"] and not data["near_ref_map_entry"]:
            print("      REASON: no links in group_map or near_ref_map")
        elif not data["near_ref_map_entry"] and not data["group_map_entry"]:
            print("      REASON: linked members have no usable preference data")
        else:
            print("      REASON: linked members have no section/row/spot preferences")
    else:
        ctx = data["group_context"]
        contribs = ctx["contributing_booking_nos"]
        sources  = ctx["contributor_sources"]
        print(f"      PRESENT — {len(contribs)} contributing co-member(s):")
        for c in contribs:
            src = sources.get(c, "?")
            print(f"        {c}  (source: {src})")
        print(f"      section_weights : {ctx['section_weights']}")
        print(f"      row_weights     : {ctx['row_weights']}")
        print(f"      spot_id_weights : {ctx['spot_id_weights']}")
        if not contribs:
            print("      WARNING: context is present but contributing_booking_nos is empty")
            print("               (context has weights but no contributing member logged —")
            print("                check _build_group_context_map contributing logic)")

    print()


if __name__ == "__main__":
    main()
