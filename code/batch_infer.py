"""
Batch-run infer.py (the CRC AI prediction + spatial-feature pipeline) over
multiple slides/images listed in a CSV file, in parallel across GPUs.

CSV format (one row per slide):
    slide_path             (required) Path to the slide/image.
    output_dir             (optional) Defaults to output/<slide_name>, same as infer.py.
    mpp                    (optional) Microns-per-pixel override.
    use_malignant_region   (optional) true/false (also accepts 1/0, yes/no). Defaults to true.

Example CSV:
    slide_path,output_dir,mpp,use_malignant_region
    /data/slideA.svs,,,true
    /data/slideB.svs,output/slideB_run2,0.25,false

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
    python batch_infer.py --csv_path slides.csv --gpu_ids 0,1,2 --num_workers 3
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


def load_jobs(csv_path):
    df = pd.read_csv(csv_path, dtype=str)
    if "slide_path" not in df.columns:
        raise ValueError("CSV must have a 'slide_path' column.")

    jobs = []
    for _, row in df.iterrows():
        slide_path = row["slide_path"]
        if pd.isna(slide_path) or not str(slide_path).strip():
            continue

        output_dir = row.get("output_dir") if "output_dir" in df.columns else None
        output_dir = output_dir if output_dir is not None and pd.notna(output_dir) and str(output_dir).strip() else None

        mpp = row.get("mpp") if "mpp" in df.columns else None
        mpp = mpp if mpp is not None and pd.notna(mpp) and str(mpp).strip() else None

        use_malignant_region = parse_bool(row.get("use_malignant_region") if "use_malignant_region" in df.columns else None)

        jobs.append({
            "slide_path": str(slide_path).strip(),
            "output_dir": output_dir,
            "mpp": mpp,
            "use_malignant_region": use_malignant_region,
        })

    return jobs


def run_one_slide(job, gpu_id):
    slide_name = Path(job["slide_path"]).stem
    output_dir = job["output_dir"] or os.path.join("output", slide_name)
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


def worker(gpu_id, jobs, results, results_lock):
    for job in jobs:
        result = run_one_slide(job, gpu_id)
        with results_lock:
            results.append(result)


def main():
    parser = argparse.ArgumentParser(description="Batch-run infer.py over multiple slides listed in a CSV file.")
    parser.add_argument("--csv_path", required=True, help="CSV file with a 'slide_path' column (optional: output_dir, mpp, use_malignant_region).")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated GPU indices to use, e.g. '0,1,2'. Defaults to all GPUs detected.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of concurrent slides to process. Defaults to the number of --gpu_ids (one slide per GPU); set higher to share GPUs across workers.")
    parser.add_argument("--results_csv", default=None, help="Where to write the summary results CSV. Defaults to '<csv_path stem>_results.csv'.")
    args = parser.parse_args()

    jobs = load_jobs(args.csv_path)
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
        threading.Thread(target=worker, args=(gpu_ids[i % len(gpu_ids)], partitions[i], results, results_lock))
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
