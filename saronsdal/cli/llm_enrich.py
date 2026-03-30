"""CLI entry point for Phase 2.5: LLM-assisted enrichment.

Usage — dry-run (no API calls, free):
    python -m saronsdal.cli.llm_enrich \\
        --phase1-dir  output/ \\
        --phase2-dir  output/ \\
        --dry-run

Usage — capped live run (API costs, bounded):
    python -m saronsdal.cli.llm_enrich \\
        --phase1-dir  output/ \\
        --phase2-dir  output/ \\
        --output-dir  llm_suggestions/ \\
        --cap 5

Usage — full live run:
    python -m saronsdal.cli.llm_enrich \\
        --phase1-dir  output/ \\
        --phase2-dir  output/ \\
        --output-dir  llm_suggestions/

Environment variables:
    GEMINI_API_KEY or GOOGLE_API_KEY  — required for live runs
    GEMINI_MODEL                      — override the default model globally

This CLI is intentionally separate from main.py — Phase 2.5 is optional,
has API costs, and must be triggered manually.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import sys
from pathlib import Path

from saronsdal.llm.candidate_builder import (
    CandidateConfig,
    build_candidates_from_files,
    print_candidate_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def _parse_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sarons Dal Phase 2.5 — LLM enrichment (candidate selection + Gemini).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--phase1-dir",
        required=True,
        metavar="DIR",
        help="Directory containing bookings_normalized.json (Phase 1 output)",
    )
    p.add_argument(
        "--phase2-dir",
        default=None,
        metavar="DIR",
        help="Directory containing resolved_groups.json and group_links.jsonl "
             "(Phase 2 output; defaults to --phase1-dir)",
    )
    p.add_argument(
        "--output-dir",
        default="llm_suggestions",
        metavar="DIR",
        help="Directory for LLM suggestion output files (default: llm_suggestions/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print candidate summary only; do not call Gemini",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N candidates of each type (cost control for testing)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=15,
        metavar="N",
        help="Number of top group phrases to show in summary (default: 15)",
    )
    p.add_argument(
        "--group-phrase-min-freq",
        type=int,
        default=2,
        metavar="N",
        help="Minimum frequency for a group phrase to be selected (default: 2)",
    )
    p.add_argument(
        "--gemini-model",
        default="gemini-2.0-flash",
        metavar="MODEL",
        help="Gemini model name (default: gemini-2.0-flash)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = _parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    phase1_dir = Path(args.phase1_dir)
    phase2_dir = Path(args.phase2_dir) if args.phase2_dir else phase1_dir

    if not (phase1_dir / "bookings_normalized.json").exists():
        logger.error("bookings_normalized.json not found in %s", phase1_dir)
        return 1
    if not (phase2_dir / "resolved_groups.json").exists():
        logger.error("resolved_groups.json not found in %s", phase2_dir)
        return 1

    config = CandidateConfig(
        group_phrase_min_frequency=args.group_phrase_min_freq,
    )

    logger.info("=== Phase 2.5: candidate selection ===")
    cs = build_candidates_from_files(phase1_dir, phase2_dir, config)
    print_candidate_summary(cs, top_n=args.top_n)

    if args.dry_run:
        print("\n[dry-run] Stopping before Gemini.  "
              "Re-run without --dry-run to call the API.")
        return 0

    # -----------------------------------------------------------------------
    # Live Gemini run
    # -----------------------------------------------------------------------
    from saronsdal.llm.gemini_client import create_client
    from saronsdal.llm.suggestion_writer import write_suggestions

    try:
        client = create_client(model_name=args.gemini_model)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    # Load bookings so the client can build near-text rosters and phrase context.
    bookings_path = phase1_dir / "bookings_normalized.json"
    with open(bookings_path, encoding="utf-8") as fh:
        bookings = json.load(fh)

    cap_msg = f" (capped at {args.cap} per type)" if args.cap else ""
    logger.info("=== Phase 2.5: calling Gemini%s ===", cap_msg)

    run = client.run_all(cs, bookings, cap=args.cap)

    output_dir = Path(args.output_dir)
    write_suggestions(output_dir, run)

    print(f"\n=== Phase 2.5 complete ===")
    print(f"  Processed : {run.candidates_processed} candidates")
    if run.candidates_capped:
        print(f"  Capped    : {run.candidates_capped} (use --cap to adjust)")
    print(f"  Errors    : {len(run.errors)}")
    print(f"  Output    : {output_dir}/")
    print(f"    group_aliases_suggested.yaml")
    print(f"    reference_resolutions.jsonl")
    print(f"    preference_enrichments.jsonl")
    print(f"    subsection_resolutions.jsonl")
    print(f"    gemini_run_summary.json")
    if run.errors:
        print(f"\nErrors (see gemini_run_summary.json for details):")
        for err in run.errors[:5]:
            print(f"  {err}")
        if len(run.errors) > 5:
            print(f"  ... and {len(run.errors) - 5} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
