#!/usr/bin/env python3
import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path


def parse_profiling_csv(csv_path: Path):
    totals = []
    ants = []
    evap = []
    update = []
    render = []
    first_food_iter = None
    final_food = None

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            totals.append(float(row["t_total_s"]))
            ants.append(float(row["t_ants_s"]))
            evap.append(float(row["t_evaporation_s"]))
            update.append(float(row["t_update_s"]))
            render.append(float(row["t_render_s"]))

            if first_food_iter is None and int(row["first_food_arrived"]) == 1:
                first_food_iter = int(row["iteration"])
            final_food = int(row["food_quantity"])

    if not totals:
        raise RuntimeError(f"No profiling rows in {csv_path}")

    return {
        "tp_ms": statistics.fmean(totals) * 1e3,
        "ants_ms": statistics.fmean(ants) * 1e3,
        "evap_ms": statistics.fmean(evap) * 1e3,
        "update_ms": statistics.fmean(update) * 1e3,
        "render_ms": statistics.fmean(render) * 1e3,
        "first_food_iter": first_food_iter,
        "final_food": final_food,
        "iterations": len(totals),
    }


def median_of(values):
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser(description="Run OpenMP scaling campaign and build summary_openmp.csv")
    parser.add_argument("--threads", default="1,2,4,8,12,14", help="Comma-separated thread counts")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per thread count")
    parser.add_argument("--iters", type=int, default=15000, help="Iterations per run")
    parser.add_argument("--binary", default="./ant_simu.exe", help="Path to simulation binary")
    parser.add_argument("--out", default="summary_openmp.csv", help="Summary CSV output path")
    parser.add_argument("--results-dir", default="results_openmp", help="Directory for per-run CSV files")
    parser.add_argument("--no-render", action="store_true", default=True, help="Run with --no-render (default true)")
    args = parser.parse_args()

    threads = [int(item.strip()) for item in args.threads.split(",") if item.strip()]
    if not threads:
        raise ValueError("No thread count provided")

    workspace = Path.cwd()
    binary_path = (workspace / args.binary).resolve() if not Path(args.binary).is_absolute() else Path(args.binary)
    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    results_dir = workspace / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    aggregated = []

    for t in threads:
        per_run = []
        print(f"[summary_openmp] threads={t}")
        for r in range(1, args.repeats + 1):
            run_csv = results_dir / f"profiling_t{t}_r{r}.csv"
            cmd = [str(binary_path), "--max-iters", str(args.iters), "--profile", str(run_csv)]
            if args.no_render:
                cmd.insert(1, "--no-render")

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(t)

            print(f"  - run {r}/{args.repeats}: {run_csv.name}")
            proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                print(proc.stdout)
                raise RuntimeError(f"Simulation failed for threads={t}, run={r}")

            per_run.append(parse_profiling_csv(run_csv))

        tp_med = median_of([row["tp_ms"] for row in per_run])
        ants_med = median_of([row["ants_ms"] for row in per_run])
        evap_med = median_of([row["evap_ms"] for row in per_run])
        update_med = median_of([row["update_ms"] for row in per_run])
        render_med = median_of([row["render_ms"] for row in per_run])

        first_food_values = [row["first_food_iter"] for row in per_run if row["first_food_iter"] is not None]
        first_food_med = int(median_of(first_food_values)) if first_food_values else "NA"
        final_food_med = int(median_of([row["final_food"] for row in per_run]))

        aggregated.append({
            "threads": t,
            "tp_ms": tp_med,
            "ants_ms": ants_med,
            "evap_ms": evap_med,
            "update_ms": update_med,
            "render_ms": render_med,
            "first_food_iter": first_food_med,
            "final_food": final_food_med,
            "iterations": per_run[0]["iterations"],
        })

    t1 = next((row["tp_ms"] for row in aggregated if row["threads"] == 1), None)
    if t1 is None:
        raise RuntimeError("Thread count 1 is required to compute speedup")

    for row in aggregated:
        speedup = t1 / row["tp_ms"]
        efficiency = 100.0 * speedup / row["threads"]
        row["speedup"] = speedup
        row["efficiency_pct"] = efficiency

    out_path = workspace / args.out
    fieldnames = [
        "threads",
        "iterations",
        "tp_ms",
        "speedup",
        "efficiency_pct",
        "ants_ms",
        "evap_ms",
        "update_ms",
        "render_ms",
        "first_food_iter",
        "final_food",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(aggregated, key=lambda x: x["threads"]):
            writer.writerow({
                "threads": row["threads"],
                "iterations": row["iterations"],
                "tp_ms": f"{row['tp_ms']:.6f}",
                "speedup": f"{row['speedup']:.6f}",
                "efficiency_pct": f"{row['efficiency_pct']:.3f}",
                "ants_ms": f"{row['ants_ms']:.6f}",
                "evap_ms": f"{row['evap_ms']:.6f}",
                "update_ms": f"{row['update_ms']:.6f}",
                "render_ms": f"{row['render_ms']:.6f}",
                "first_food_iter": row["first_food_iter"],
                "final_food": row["final_food"],
            })

    print(f"[summary_openmp] Summary written to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[summary_openmp] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
