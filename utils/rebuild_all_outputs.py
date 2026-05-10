import argparse
import json
from pathlib import Path
from typing import Any


def load_sample_objects(run_dir: Path, pattern: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(run_dir.glob(pattern)):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend([x for x in obj if isinstance(x, dict)])
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Deduplicate by (problem_name, sample_index). Keep last seen.
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    fallback: list[dict[str, Any]] = []

    for r in records:
        problem_name = r.get("problem_name")
        sample_index = r.get("sample_index")
        if isinstance(problem_name, str) and isinstance(sample_index, int):
            by_key[(problem_name, sample_index)] = r
        else:
            fallback.append(r)

    deduped = list(by_key.values()) + fallback
    deduped.sort(
        key=lambda x: (
            str(x.get("problem_name", "")),
            int(x.get("sample_index", -1)) if isinstance(x.get("sample_index"), int) else -1,
        )
    )
    return deduped


def maybe_write_protocol_metadata(
    run_dir: Path,
    records: list[dict[str, Any]],
    model_path_override: str | None,
    force: bool,
) -> Path | None:
    out = run_dir / "protocol_metadata.json"
    if out.exists() and not force:
        return None

    # Build minimal metadata needed by utils scripts.
    gm = None
    for r in records:
        cand = r.get("generation_metadata")
        if isinstance(cand, dict):
            gm = cand
            break

    max_tokens = None
    if isinstance(gm, dict):
        mt = gm.get("max_tokens")
        if isinstance(mt, int):
            max_tokens = mt

    model_path = model_path_override
    if model_path is None and isinstance(gm, dict):
        # generation_metadata in this project usually lacks model_path,
        # but keep the fallback for compatibility with older runs.
        mp = gm.get("model_path")
        if isinstance(mp, str) and mp.strip():
            model_path = mp.strip()

    metadata = {
        "reconstructed": True,
        "model_path": model_path or "",
        "max_tokens": max_tokens if isinstance(max_tokens, int) else 0,
        "note": "Reconstructed from individual sample JSON files.",
    }
    out.write_text(json.dumps(metadata, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild all_outputs.json from individual *_sample*.json files."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing individual sample JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="*_sample*.json",
        help="Glob pattern for individual sample files.",
    )
    parser.add_argument(
        "--output_name",
        default="all_outputs.json",
        help="Output filename for the rebuilt combined file.",
    )
    parser.add_argument(
        "--write_protocol_metadata",
        action="store_true",
        help="Also write minimal protocol_metadata.json if missing.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Optional model path to include in reconstructed protocol_metadata.json.",
    )
    parser.add_argument(
        "--force_protocol_metadata",
        action="store_true",
        help="Overwrite protocol_metadata.json if it already exists.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    records = load_sample_objects(run_dir, args.pattern)
    if not records:
        raise SystemExit(f"No records found in {run_dir} with pattern '{args.pattern}'.")

    deduped = dedupe_records(records)
    out = run_dir / args.output_name
    out.write_text(json.dumps(deduped, indent=2))

    print(f"run_dir: {run_dir}")
    print(f"pattern: {args.pattern}")
    print(f"raw_records: {len(records)}")
    print(f"deduped_records: {len(deduped)}")
    print(f"wrote: {out}")

    if args.write_protocol_metadata:
        pm = maybe_write_protocol_metadata(
            run_dir=run_dir,
            records=deduped,
            model_path_override=args.model_path,
            force=args.force_protocol_metadata,
        )
        if pm is None:
            print("protocol_metadata.json exists; skipped (use --force_protocol_metadata to overwrite).")
        else:
            print(f"wrote: {pm}")
            print("NOTE: protocol_metadata.json is minimal reconstruction.")


if __name__ == "__main__":
    main()
