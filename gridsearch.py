#!/usr/bin/env python3
"""
gridsearch.py – Grid search over LoRA / training hyperparameters.

For each combination, train.py is called as a subprocess so that each
run gets a fresh process (important for GPU memory management across runs).

Results from all runs are aggregated into a single CSV for easy comparison.

Example usage:
  python gridsearch.py \\
      --dataset emodb \\
      --data_dir /ssd/4/u/baz9201/data/EMODB \\
      --output_root /stor/u/baz9201/ser_results/gridsearch_emodb \\
      --lora_r 8 16 32 \\
      --lora_alpha 16 32 \\
      --lr 1e-4 5e-5 \\
      --epochs 3 5 \\
      --grad_accum 4 \\
      --seeds 42 123

This will run 3 × 2 × 2 × 2 × 1 = 24 combinations, each with 2 seeds
→ 48 runs total.

Tip: on Berta, wrap this in a SLURM job:
  sbatch --partition=gpu --gres=gpu:4 --time=48:00:00 \\
      --wrap "python gridsearch.py ..."
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Grid search over SER hyperparameters"
    )

    # ---- Required ----
    p.add_argument("--dataset",      required=True, choices=["emodb", "ravdess"])
    p.add_argument("--data_dir",     required=True)
    p.add_argument("--output_root",  required=True,
                   help="Root directory; one sub-dir is created per run")
    p.add_argument("--model_path",   default="Qwen/Qwen2-Audio-7B-Instruct")

    # ---- Grid axes (one or more values each) ----
    p.add_argument("--lora_r",       type=int,   nargs="+", default=[16])
    p.add_argument("--lora_alpha",   type=int,   nargs="+", default=[32])
    p.add_argument("--lora_dropout", type=float, nargs="+", default=[0.05])
    p.add_argument("--lr",           type=float, nargs="+", default=[1e-4])
    p.add_argument("--epochs",       type=int,   nargs="+", default=[3])
    p.add_argument("--grad_accum",   type=int,   nargs="+", default=[4])
    p.add_argument("--train_prompt", type=str,   nargs="+", default=["normal"])
    p.add_argument("--seeds",        type=int,   nargs="+", default=[123])

    # ---- Execution ----
    p.add_argument("--python",    default=sys.executable,
                   help="Python interpreter to use (default: same as current)")
    p.add_argument("--dry_run",   action="store_true",
                   help="Print commands without executing them")
    p.add_argument("--skip_done", action="store_true", default=True,
                   help="Skip runs whose output directory already has summary.json")
    p.add_argument("--no_skip_done", dest="skip_done", action="store_false")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Build all combinations
# ---------------------------------------------------------------------------

def build_grid(args) -> List[Dict[str, Any]]:
    keys = [
        "lora_r", "lora_alpha", "lora_dropout",
        "lr", "epochs", "grad_accum", "train_prompt", "seeds",
    ]
    values = [
        args.lora_r, args.lora_alpha, args.lora_dropout,
        args.lr, args.epochs, args.grad_accum, args.train_prompt, args.seeds,
    ]
    combos = []
    for combo in itertools.product(*values):
        d = dict(zip(keys, combo))
        d["seed"] = d.pop("seeds")   # rename for train.py
        combos.append(d)
    return combos


def run_name_for(params: Dict[str, Any]) -> str:
    return (
        f"R{params['lora_r']}"
        f"_A{params['lora_alpha']}"
        f"_DO{params['lora_dropout']}"
        f"_lr{params['lr']}"
        f"_ep{params['epochs']}"
        f"_AS{params['grad_accum']}"
        f"_p{params['train_prompt']}"
        f"_s{params['seed']}"
    )


# ---------------------------------------------------------------------------
# Load an existing summary (if the run already finished)
# ---------------------------------------------------------------------------

def load_summary(run_dir: str):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Aggregate all run summaries into one CSV
# ---------------------------------------------------------------------------

def write_master_csv(results: List[Dict], output_path: str) -> None:
    """
    Write one row per (run, prompt) with hyperparameters + metrics.
    """
    if not results:
        return

    # Collect all prompt names seen
    prompt_names = set()
    for r in results:
        if r.get("summary"):
            prompt_names.update(r["summary"].keys())
    prompt_names = sorted(prompt_names)

    param_keys = [
        "dataset", "lora_r", "lora_alpha", "lora_dropout",
        "lr", "epochs", "grad_accum", "train_prompt", "seed",
    ]
    metric_keys = ["mean_uar", "std_uar", "mean_acc", "std_acc", "mean_wf1", "std_wf1"]

    fieldnames = param_keys + ["eval_prompt"] + metric_keys

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            if not r.get("summary"):
                continue
            base_row = {k: r["params"].get(k, "") for k in param_keys}
            for pname in prompt_names:
                if pname not in r["summary"]:
                    continue
                m = r["summary"][pname]
                row = dict(base_row)
                row["eval_prompt"] = pname
                for mk in metric_keys:
                    row[mk] = f"{m[mk]:.4f}" if isinstance(m[mk], float) else m[mk]
                writer.writerow(row)

    print(f"\nMaster CSV written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)

    grid = build_grid(args)
    print(f"\n{'='*60}")
    print(f"Grid search: {len(grid)} combinations")
    print(f"Dataset:     {args.dataset}")
    print(f"Output root: {args.output_root}")
    print(f"{'='*60}\n")

    results = []
    failed  = []

    for idx, params in enumerate(grid):
        rname   = run_name_for(params)
        run_dir = os.path.join(args.output_root, rname)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n[{idx+1}/{len(grid)}] {rname}")

        # Check if already done
        if args.skip_done and load_summary(run_dir) is not None:
            print("  → already done, skipping.")
            summary = load_summary(run_dir)
            results.append({"params": {**params, "dataset": args.dataset}, "summary": summary})
            continue

        # Build command
        cmd = [
            args.python, "train.py",
            "--dataset",      args.dataset,
            "--data_dir",     args.data_dir,
            "--output_dir",   run_dir,
            "--model_path",   args.model_path,
            "--lora_r",       str(params["lora_r"]),
            "--lora_alpha",   str(params["lora_alpha"]),
            "--lora_dropout", str(params["lora_dropout"]),
            "--lr",           str(params["lr"]),
            "--epochs",       str(params["epochs"]),
            "--grad_accum",   str(params["grad_accum"]),
            "--train_prompt", str(params["train_prompt"]),
            "--seed",         str(params["seed"]),
        ]

        print("  Command:", " ".join(cmd))

        if args.dry_run:
            print("  [dry run] skipping execution")
            continue

        ret = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        if ret.returncode != 0:
            print(f"  [ERROR] run failed with return code {ret.returncode}")
            failed.append(rname)
        else:
            summary = load_summary(run_dir)
            results.append({"params": {**params, "dataset": args.dataset}, "summary": summary})
            if summary:
                # Quick peek: print train-prompt UAR
                tp = params["train_prompt"]
                if tp in summary:
                    m = summary[tp]
                    print(
                        f"  → {tp}: UAR={m['mean_uar']*100:.2f}% ± {m['std_uar']*100:.2f}%"
                    )

    # ------------------------------------------------------------------
    # Write master CSV
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_csv = os.path.join(args.output_root, f"gridsearch_results_{ts}.csv")
    write_master_csv(results, master_csv)

    if failed:
        print(f"\n[WARNING] {len(failed)} runs failed:")
        for name in failed:
            print(f"  - {name}")

    print(f"\nGrid search complete. {len(results)}/{len(grid)} runs succeeded.")


if __name__ == "__main__":
    main()
