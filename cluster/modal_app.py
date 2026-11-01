"""Run the manuscript simulations on Modal (https://modal.com) CPU containers.

Every task in experiments/registry.py runs in its own container; the shards it
writes are copied back into results/ on this machine. The merges are cheap and
run locally.

    modal run cluster/modal_app.py --stage 1          # ~410 tasks
    python scripts/run_tasks.py --merge 1
    modal run cluster/modal_app.py --stage 2          # per-subject model fits
    python scripts/run_tasks.py --merge 2

Options: --smoke (a few trials per task), --only <task-name prefix>,
--skip-done (leave out tasks that already have a log in results/logs/).
Requires `pip install modal` and `modal token new` once.
"""
import os
import pathlib
import shutil
import subprocess
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = "/root/code"
LOCK = "bias/adaptive_bias_locked_design.json"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("gpytorch==1.13", "numpy==2.0.2", "scipy==1.13.1", "pandas==2.3.3",
                 "matplotlib==3.9.4", "tqdm==4.67.3", "PyYAML==6.0.1")
    .add_local_dir(ROOT, REMOTE, ignore=[
        "**/__pycache__/**", "figures/output/**", "results/**/shards/**",
        "results/logs/**", "results/**/*.csv", "results/**/*.npz", ".venv/**",
    ])
)
app = modal.App("memgp-rerun", image=image)


def _execute(name, argv, inputs):
    out = pathlib.Path("/tmp/results")
    shutil.rmtree(out, ignore_errors=True)
    (out / "bias").mkdir(parents=True)
    shutil.copy(pathlib.Path(REMOTE) / "results" / LOCK, out / LOCK)
    for rel, data in inputs.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_bytes(data)
    env = dict(os.environ, MEMGP_RESULTS=str(out), PYTHONUNBUFFERED="1",
               TORCH_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, "-m", *argv], cwd=REMOTE, env=env,
                          capture_output=True, text=True)
    skip = {LOCK, *inputs}
    files = {str(p.relative_to(out)): p.read_bytes()
             for p in out.rglob("*") if p.is_file() and str(p.relative_to(out)) not in skip}
    return name, proc.returncode, proc.stdout[-6000:] + proc.stderr[-6000:], files


@app.function(cpu=1.0, memory=2048, timeout=6 * 3600, retries=1)
def run_1cpu(name, argv, inputs):
    return _execute(name, argv, inputs)


@app.function(cpu=8.0, memory=8192, timeout=6 * 3600, retries=1)
def run_8cpu(name, argv, inputs):
    return _execute(name, argv, inputs)


@app.local_entrypoint()
def main(stage: int = 1, smoke: bool = False, only: str = "", skip_done: bool = False):
    sys.path.insert(0, str(ROOT))
    from experiments import registry

    tasks = registry.stage1(smoke) if stage == 1 else registry.stage2(smoke)
    if only:
        tasks = [t for t in tasks if t[0].startswith(only)]
    results_dir = pathlib.Path(os.environ.get("MEMGP_RESULTS", ROOT / "results"))
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if skip_done:
        kept, skipped = [], []
        for t in tasks:
            if (log_dir / f"{t[0]}.log").exists():
                skipped.append(t[0])
            else:
                kept.append(t)
        print(f"skip-done: {len(skipped)} already have logs, {len(kept)} remaining")
        tasks = kept
        if not tasks:
            print(f"stage {stage} already complete; now run: python scripts/run_tasks.py --merge {stage}")
            return

    inputs = {}
    if stage == 2:
        rel = "setsize/setsize_main_trials.csv"
        inputs[rel] = (results_dir / rel).read_bytes()
    small = [(n, a, inputs) for n, a, c in tasks if c <= 1]
    large = [(n, a, inputs) for n, a, c in tasks if c > 1]
    print(f"stage {stage}: {len(small)} 1-cpu tasks, {len(large)} 8-cpu tasks")
    calls = []
    if small:
        calls.append(run_1cpu.starmap(small, order_outputs=False, return_exceptions=True,
                                     wrap_returned_exceptions=False))
    if large:
        calls.append(run_8cpu.starmap(large, order_outputs=False, return_exceptions=True,
                                     wrap_returned_exceptions=False))

    failed, done = [], 0
    for gen in calls:
        for res in gen:
            done += 1
            if isinstance(res, BaseException):
                failed.append(repr(res))
                print(f"[{done}/{len(tasks)}] exception: {res!r}", flush=True)
                continue
            name, rc, log, files = res
            (log_dir / f"{name}.log").write_text(log)
            for rel, data in files.items():
                path = results_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            if rc != 0:
                failed.append(name)
            print(f"[{done}/{len(tasks)}] {name}: {status}, {len(files)} files", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} tasks failed: {failed[:10]}")
    print(f"stage {stage} complete; now run: python scripts/run_tasks.py --merge {stage}")
