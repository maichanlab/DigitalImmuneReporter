"""
Batch-run infer.py (the CRC AI prediction + spatial-feature pipeline) over
multiple slides/images listed in a CSV file, in parallel across GPUs.

CSV format (one row per slide):
    slide_path             (required) Path to the slide/image.
    mpp                    (optional) Microns-per-pixel override.
    use_malignant_region   (optional) true/false (also accepts 1/0, yes/no). Defaults to true.
    cell_type_method       (optional) morphology_based/miphei_multiplex/both. Defaults to morphology_based.

Every slide's output goes to <output_folder>/<slide_name> (--output_folder,
defaults to "output") - there is no per-slide output directory override; all
outputs for a batch live under one root.

Example CSV:
    slide_path,mpp,use_malignant_region,cell_type_method
    /data/slideA.svs,,true,morphology_based
    /data/slideB.svs,0.25,false,miphei_multiplex

Batch-wide overrides: --mpp/--use_malignant_region/--no-use_malignant_region/
--cell_type_method can also be passed on the command line to apply a
non-default value to every slide in the batch, instead of setting it per-row
in the CSV. Mutually exclusive with the CSV column at the batch level: if
--mpp (say) is passed, the CSV's 'mpp' column must be entirely empty (no row
may set it) - if any row does, the script refuses to start and reports the
conflict so you can clear the column or drop the CLI flag and re-run.

Parallelism: each slide runs as its own `infer.py` subprocess, pinned to one
GPU via CUDA_VISIBLE_DEVICES so Trident/mmdet/mmseg all land on that device
(subprocess isolation also means one slide crashing/OOMing doesn't take down
the batch). By default one slide runs per GPU at a time (--num_workers
defaults to the number of GPUs detected via torch); pass --gpu_ids to
restrict which GPUs are used, or --num_workers to control concurrency
directly (workers round-robin across --gpu_ids, so multiple workers can
share a GPU if you have enough VRAM for it).

Usage:
    python batch_infer.py --csv_path slides.csv
    python batch_infer.py --csv_path slides.csv --output_folder /data/batch_run_1
    python batch_infer.py --csv_path slides.csv --gpu_ids 0,1,2 --num_workers 3
    python batch_infer.py --csv_path slides.csv --cell_type_method miphei_multiplex --mpp 0.25
"""
import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parent
INFER_SCRIPT = REPO_ROOT / "infer.py"

TRUE_STRINGS = {"true", "1", "yes", "y"}
FALSE_STRINGS = {"false", "0", "no", "n"}


def parse_bool(value, default=True):
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return default
    value = str(value).strip().lower()
    if value in TRUE_STRINGS:
        return True
    if value in FALSE_STRINGS:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r} (expected true/false, 1/0, or yes/no)")


def _column_in_use(df, column):
    """True if `column` exists in the CSV and at least one row sets a non-blank value for it."""
    if column not in df.columns:
        return False
    return df[column].apply(lambda v: pd.notna(v) and str(v).strip() != "").any()


def load_jobs(csv_path, mpp_override=None, use_malignant_region_override=None, cell_type_method_override=None):
    df = pd.read_csv(csv_path, dtype=str)
    if "slide_path" not in df.columns:
        raise ValueError("CSV must have a 'slide_path' column.")

    # Batch-level mutual exclusivity: a CLI override and the CSV column for the same field can't
    # both be in use at once (which should win is ambiguous) - checked once, up front, so the
    # whole batch aborts before any subprocess starts rather than partway through.
    conflicts = []
    if mpp_override is not None and _column_in_use(df, "mpp"):
        conflicts.append(f"  'mpp': --mpp {mpp_override!r} was passed, but the CSV's 'mpp' column also has values set.")
    if use_malignant_region_override is not None and _column_in_use(df, "use_malignant_region"):
        conflicts.append("  'use_malignant_region': --use_malignant_region/--no-use_malignant_region was passed, but the CSV's 'use_malignant_region' column also has values set.")
    if cell_type_method_override is not None and _column_in_use(df, "cell_type_method"):
        conflicts.append(f"  'cell_type_method': --cell_type_method {cell_type_method_override!r} was passed, but the CSV's 'cell_type_method' column also has values set.")
    if conflicts:
        raise ValueError(
            "Conflicting options between the CSV and command-line overrides "
            "(ambiguous which should win) - fix the CSV (clear the column) or drop the CLI flag, then re-run:\n"
            + "\n".join(conflicts)
        )

    jobs = []
    for _, row in df.iterrows():
        slide_path = row["slide_path"]
        if pd.isna(slide_path) or not str(slide_path).strip():
            continue
        slide_path = str(slide_path).strip()

        csv_mpp = row.get("mpp") if "mpp" in df.columns else None
        csv_mpp = csv_mpp if csv_mpp is not None and pd.notna(csv_mpp) and str(csv_mpp).strip() else None
        mpp = csv_mpp if csv_mpp is not None else mpp_override

        csv_use_malignant_region_raw = row.get("use_malignant_region") if "use_malignant_region" in df.columns else None
        csv_use_malignant_region_set = csv_use_malignant_region_raw is not None and pd.notna(csv_use_malignant_region_raw) and str(csv_use_malignant_region_raw).strip() != ""
        if csv_use_malignant_region_set:
            use_malignant_region = parse_bool(csv_use_malignant_region_raw)
        elif use_malignant_region_override is not None:
            use_malignant_region = use_malignant_region_override
        else:
            use_malignant_region = True

        csv_cell_type_method = row.get("cell_type_method") if "cell_type_method" in df.columns else None
        csv_cell_type_method = csv_cell_type_method if csv_cell_type_method is not None and pd.notna(csv_cell_type_method) and str(csv_cell_type_method).strip() else None
        cell_type_method = csv_cell_type_method or cell_type_method_override or "morphology_based"
        if cell_type_method not in {"morphology_based", "miphei_multiplex", "both"}:
            raise ValueError(f"Invalid cell_type_method: {cell_type_method!r} (expected morphology_based, miphei_multiplex, or both)")

        jobs.append({
            "slide_path": slide_path,
            "mpp": mpp,
            "use_malignant_region": use_malignant_region,
            "cell_type_method": cell_type_method,
        })

    return jobs


def run_one_slide(job, gpu_id, output_folder):
    slide_name = Path(job["slide_path"]).stem
    output_dir = os.path.join(output_folder, slide_name)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, str(INFER_SCRIPT),
        "--slide_path", job["slide_path"],
        "--output_dir", output_dir,
        "--gpu", "0",  # CUDA_VISIBLE_DEVICES below remaps this to the assigned physical GPU
    ]
    if job["mpp"] is not None:
        cmd += ["--mpp", str(job["mpp"])]
    if not job["use_malignant_region"]:
        cmd += ["--no-use_malignant_region"]
    cmd += ["--cell_type_method", job["cell_type_method"]]

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))

    log_path = os.path.join(output_dir, "batch_run.log")
    logging.info(f"[GPU {gpu_id}] Starting {job['slide_path']} -> {output_dir}")
    start = time.perf_counter()
    with open(log_path, "w") as log_file:
        result = subprocess.run(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start

    status = "success" if result.returncode == 0 else "failed"
    if status == "success":
        logging.info(f"[GPU {gpu_id}] Done: {job['slide_path']} ({elapsed / 60:.1f} min)")
    else:
        logging.error(f"[GPU {gpu_id}] FAILED: {job['slide_path']} (exit code {result.returncode}); see {log_path}")

    return {
        "slide_path": job["slide_path"],
        "output_dir": output_dir,
        "status": status,
        "elapsed_seconds": round(elapsed, 1),
        "log_path": log_path,
    }


def worker(gpu_id, jobs, output_folder, results, results_lock):
    for job in jobs:
        result = run_one_slide(job, gpu_id, output_folder)
        with results_lock:
            results.append(result)


def main():
    parser = argparse.ArgumentParser(description="Batch-run infer.py over multiple slides listed in a CSV file.")
    parser.add_argument("--csv_path", required=True, help="CSV file with a 'slide_path' column (optional: mpp, use_malignant_region, cell_type_method).")
    parser.add_argument("--output_folder", default="output", help="Root directory for all slides' outputs; each slide is written to <output_folder>/<slide_name>. Defaults to 'output'.")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated GPU indices to use, e.g. '0,1,2'. Defaults to all GPUs detected.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of concurrent slides to process. Defaults to the number of --gpu_ids (one slide per GPU); set higher to share GPUs across workers.")
    parser.add_argument("--results_csv", default=None, help="Where to write the summary results CSV. Defaults to '<csv_path stem>_results.csv'.")
    parser.add_argument("--mpp", type=float, default=None, help="Batch-wide microns-per-pixel override, applied to every slide whose CSV row leaves 'mpp' blank. Conflicts with a row that sets its own 'mpp'.")
    parser.add_argument(
        "--use_malignant_region", action=argparse.BooleanOptionalAction, default=None,
        help="Batch-wide default for whether to run malignant-region identification (Step 2), applied to every slide whose CSV row leaves 'use_malignant_region' blank. Conflicts with a row that sets its own 'use_malignant_region'.",
    )
    parser.add_argument(
        "--cell_type_method", choices=["morphology_based", "miphei_multiplex", "both"], default=None,
        help="Batch-wide default cell-typing method, applied to every slide whose CSV row leaves 'cell_type_method' blank. Conflicts with a row that sets its own 'cell_type_method'.",
    )
    args = parser.parse_args()

    jobs = load_jobs(
        args.csv_path,
        mpp_override=args.mpp,
        use_malignant_region_override=args.use_malignant_region,
        cell_type_method_override=args.cell_type_method,
    )
    if not jobs:
        logging.warning("No slides found in CSV; nothing to do.")
        return
    logging.info(f"Loaded {len(jobs)} slide(s) from {args.csv_path}")

    if args.gpu_ids:
        gpu_ids = [g.strip() for g in args.gpu_ids.split(",") if g.strip()]
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs detected; the pipeline requires a GPU. Pass --gpu_ids to override.")
        gpu_ids = [str(i) for i in range(num_gpus)]

    num_workers = args.num_workers or len(gpu_ids)
    logging.info(f"Using GPUs {gpu_ids} with {num_workers} worker(s)")

    results = []
    results_lock = threading.Lock()
    partitions = [jobs[i::num_workers] for i in range(num_workers)]
    threads = [
        threading.Thread(target=worker, args=(gpu_ids[i % len(gpu_ids)], partitions[i], args.output_folder, results, results_lock))
        for i in range(num_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    results_csv = args.results_csv or f"{Path(args.csv_path).stem}_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)

    num_success = sum(1 for r in results if r["status"] == "success")
    logging.info(f"Batch complete: {num_success}/{len(results)} succeeded. Summary saved to {results_csv}")


if __name__ == "__main__":
    main()
