"""Print a table comparing eval results against OPSD paper baselines."""
import argparse
import json
import sys
from pathlib import Path

BASELINES = {
    # (model_name_fragment, mode): {dataset: avg@12}
    ("Qwen3-1.7B", "thinking"): {"aime24": 51.5, "aime25": 36.7, "hmmt25": 23.1},
    ("Qwen3-1.7B", "nonthinking"): {"aime24": 11.9, "aime25": 9.2, "hmmt25": 5.0},
    ("Qwen3-4B", "nonthinking"): {"aime24": 23.1, "aime25": 21.4, "hmmt25": 10.8},
    ("Qwen3-8B", "nonthinking"): {"aime24": 26.4, "aime25": 19.7, "hmmt25": 10.8},
}


def load_result(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--model", default="")
    parser.add_argument("--mode", default="")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob("*.json"))

    if not json_files:
        print(f"No result files found in {results_dir}")
        sys.exit(0)

    # Filter by model/mode if specified
    if args.model or args.mode:
        model_name = Path(args.model).name if args.model else ""
        filtered = [f for f in json_files
                    if (not model_name or model_name in f.name)
                    and (not args.mode or args.mode in f.name)]
        json_files = filtered if filtered else json_files

    # Find matching baseline
    baseline = None
    for (model_frag, mode_key), vals in BASELINES.items():
        if model_frag in args.model and mode_key == args.mode:
            baseline = vals
            break

    datasets = ["aime24", "aime25", "hmmt25", "math500", "amc23"]

    # Collect results by dataset
    by_dataset = {}
    for f in json_files:
        data = load_result(f)
        if data is None:
            continue
        ds = data.get("dataset", "unknown")
        by_dataset[ds] = data

    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY — {args.model or 'unknown model'}  [{args.mode}]")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Avg@N':>8} {'Pass@N':>8} {'N':>4} {'Baseline':>10} {'Delta':>8}")
    print("-" * 70)

    for ds in datasets:
        if ds not in by_dataset:
            continue
        d = by_dataset[ds]
        avg = d.get("avg_at_n", 0)
        passk = d.get("pass_at_n_pct", 0)
        n = d.get("val_n", "?")
        ref = (baseline or {}).get(ds, None)
        ref_str = f"{ref:.1f}%" if ref is not None else "N/A"
        delta_str = f"{avg - ref:+.1f}%" if ref is not None else ""
        flag = ""
        if ref is not None:
            if abs(avg - ref) > 5:
                flag = " ← CHECK"
        print(f"{ds:<12} {avg:>7.2f}% {passk:>7.2f}% {n:>4}  {ref_str:>9}  {delta_str:>7}{flag}")

    print("=" * 70)
    print("Baseline source: OPSD paper (Zhao et al. 2026), same eval protocol")
    print("Note: ±3-5% variation is expected due to sampling randomness")
    if baseline:
        print(f"Expecting results close to: {baseline}")
    print()


if __name__ == "__main__":
    main()
