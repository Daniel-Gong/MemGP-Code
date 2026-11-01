"""Run the manuscript simulations on this machine, several tasks at a time.

    python scripts/run_tasks.py                # everything, full scale
    python scripts/run_tasks.py --smoke        # a few trials per task (pipeline check)
    python scripts/run_tasks.py --jobs 8 --only setsize-main

Stages: stage-1 tasks, stage-1 merges, stage-2 tasks (per-subject model
fits, which need the merged set-size data), stage-2 merge. Logs go to
results/logs/. Set MEMGP_RESULTS to write somewhere other than results/.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import registry  # noqa: E402
from memgp.paths import RESULTS_DIR  # noqa: E402


def run(name, argv, log_dir):
    log = log_dir / f"{name}.log"
    env = dict(os.environ, TORCH_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, "-m", *argv], cwd=ROOT,
                             stdout=fh, stderr=subprocess.STDOUT, env=env)
    return name, rc


def run_many(tasks, jobs, log_dir):
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run, n, a, log_dir) for n, a, _ in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            name, rc = fut.result()
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"[{i}/{len(tasks)}] {name}: {status}", flush=True)
            if rc != 0:
                failed.append(name)
    return failed


def run_merges(merges, log_dir):
    for argv in merges:
        name = "merge-" + "-".join(a for a in argv if not a.startswith("--"))
        _, rc = run(name.replace("experiments.", ""), argv, log_dir)
        print(f"{name}: {'ok' if rc == 0 else 'FAILED'}", flush=True)
        if rc != 0:
            raise SystemExit(f"merge failed; see {log_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--only", default=None, help="Run stage-1 tasks whose name starts with this.")
    ap.add_argument("--merge", type=int, choices=(1, 2), default=None,
                    help="Only run the merges of this stage (after a Modal or Slurm run).")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    ap.add_argument("--index", type=int, default=None,
                    help="Run only task number INDEX of --stage (Slurm array task).")
    ap.add_argument("--count", action="store_true", help="Print the number of tasks in --stage.")
    args = ap.parse_args()

    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stage_tasks = (registry.stage1 if args.stage == 1 else registry.stage2)(smoke=args.smoke)
    if args.count:
        print(len(stage_tasks))
        return
    if args.index is not None:
        name, argv, _ = stage_tasks[args.index]
        print(f"task {args.index}: {name}: python -m {' '.join(argv)}", flush=True)
        raise SystemExit(subprocess.call([sys.executable, "-m", *argv], cwd=ROOT))

    if args.merge:
        run_merges(registry.MERGES_STAGE1 if args.merge == 1 else registry.MERGES_STAGE2,
                   log_dir)
        return

    tasks = registry.stage1(smoke=args.smoke)
    if args.only:
        tasks = [t for t in tasks if t[0].startswith(args.only)]
        failed = run_many(tasks, args.jobs, log_dir)
        raise SystemExit(1 if failed else 0)

    failed = run_many(tasks, args.jobs, log_dir)
    if failed:
        raise SystemExit(f"{len(failed)} stage-1 tasks failed: {failed[:5]} ... see {log_dir}")
    run_merges(registry.MERGES_STAGE1, log_dir)
    failed = run_many(registry.stage2(smoke=args.smoke), args.jobs, log_dir)
    if failed:
        raise SystemExit(f"stage-2 tasks failed: {failed}")
    run_merges(registry.MERGES_STAGE2, log_dir)
    print("All simulations done.")


if __name__ == "__main__":
    main()
