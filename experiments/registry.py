"""Every simulation behind the manuscript, as independent tasks.

Each task is one command (``python -m <module> <args>``) that writes a shard
under results/. Stage 1 tasks are independent of each other; stage 2 needs the
merged stage-1 set-size data. ``MERGES`` combines the shards into the files the
figures read. ``scripts/run_tasks.py``, ``cluster/modal_app.py`` and the Slurm
scripts all take their work from here.

    python -m experiments.registry            # list the tasks
"""
# Number of parameter levels per set-size design (experiments/setsize_subjects.py
# checks this against its DESIGNS table). Kept here so the registry imports no
# simulation code and can be read by job launchers without PyTorch.
DESIGN_LEVELS = {"main": 1, "M": 4, "beta": 4, "lambda": 4, "misbinding": 1}

N_SUBJECTS = 20
N_BIAS_PRECISION_SUBJECTS = 24


def stage1(smoke=False):
    """List of (name, argv, cpus)."""
    n_subj = 2 if smoke else N_SUBJECTS
    def t(small):
        return ["--n_trials", str(small)] if smoke else []

    out = []
    for design, n_levels in DESIGN_LEVELS.items():
        for s in range(n_subj):
            for li in range(n_levels):
                out.append((f"setsize-{design}-s{s}-l{li}",
                            ["experiments.setsize_subjects", design,
                             "--subject_idx", str(s), "--level", str(li)] + t(1), 1))
    for size in (2, 4):
        for s in range(n_subj):
            out.append((f"retrocue-N{size}-s{s}",
                        ["experiments.retrocue_subjects", "retrocue", "--set_size", str(size),
                         "--subject_idx", str(s)] + t(2), 1))
    for s in range(n_subj):
        out.append((f"flexibility-s{s}",
                    ["experiments.retrocue_subjects", "flexibility",
                     "--subject_idx", str(s)] + t(1), 1))
        out.append((f"bias-s{s}",
                    ["experiments.bias_adaptive", "confirm_shard", "--figure_only",
                     "--subject_idx", str(s), "--workers", "8"] + t(1), 8))
    for s in range(2 if smoke else N_BIAS_PRECISION_SUBJECTS):
        out.append((f"bias-precision-s{s}",
                    ["experiments.bias_precision", "--subject_idx", str(s)] + t(2), 1))
    out += [
        ("dynamics",
         ["experiments.retrocue_dynamics", "--n_trials", "2" if smoke else "100",
          "--workers", "8"], 8),
        ("circuit-equivalence",
         ["experiments.circuit_equivalence", "--trials", "2" if smoke else "100"], 1),
        ("circuit-bump-drift",
         ["experiments.circuit_bump_drift", "--trials", "3" if smoke else "100"], 1),
        ("circuit-decoded-uncertainty",
         ["experiments.circuit_decoded_uncertainty", "--trials", "3" if smoke else "100",
          "--set-size", "4"] + (["--n-mc", "10"] if smoke else []), 1),
        ("circuit-cued-precision",
         ["experiments.circuit_cued_precision", "--trials", "3" if smoke else "100",
          "--workers", "8"], 8),
    ]
    return out


def stage2(smoke=False):
    n_subj = 2 if smoke else N_SUBJECTS
    extra = ["--de_maxiter", "3"] if smoke else []
    return [(f"model-comparison-s{s}",
             ["experiments.setsize_model_comparison", "--subject_idx", str(s)] + extra, 1)
            for s in range(n_subj)]


MERGES_STAGE1 = (
    [["experiments.setsize_subjects", d, "--merge"] for d in DESIGN_LEVELS]
    + [["experiments.retrocue_subjects", "retrocue", "--set_size", "2", "--merge"],
       ["experiments.retrocue_subjects", "retrocue", "--set_size", "4", "--merge"],
       ["experiments.retrocue_subjects", "aggregate"],
       ["experiments.retrocue_subjects", "flexibility", "--merge"],
       ["experiments.bias_adaptive", "confirm_merge"],
       ["experiments.bias_precision", "--merge"]]
)
MERGES_STAGE2 = [["experiments.setsize_model_comparison", "--merge"]]


if __name__ == "__main__":
    for stage, tasks in (("stage 1", stage1()), ("stage 2", stage2())):
        print(f"{stage}: {len(tasks)} tasks")
        for name, argv, cpus in tasks[:3]:
            print(f"  {name}: python -m {' '.join(argv)}  [{cpus} cpu]")
        print("  ...")
