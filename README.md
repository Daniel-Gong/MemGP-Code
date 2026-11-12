# MemGP

Code and simulation results for

> Dongyu Gong, Mario Belledonne, Ilker Yildirim. *A single inference process
> underlying the behavior and neural circuits of working memory.*

MemGP treats visual working memory as inference over a continuous function on
the location × color torus. A sparse variational Gaussian process encodes noisy
sensory samples, maintains the posterior through the delay under a KL
forgetting term and optional attention weights, and reports a color by
sampling the posterior and applying divisive normalization. Inducing points
start on a 10 × 10 grid; their locations are learned. The recurrent network in
Figure 5 is the same maintenance update with the geometry held fixed.

Every figure, table and statistic in the paper can be regenerated from the
files in `results/` in a few minutes. Rerunning the simulations that fill
`results/` takes about 140 CPU-hours; `cluster/` runs them on Modal or Slurm.

## Repository layout

```
memgp/            the model
  gp_model.py       WorkingMemoryGP (GPyTorch sparse variational GP)
  generator.py      stimulus displays and noisy sensory samples
  attention.py      spatial attention gain a(l)
  simulation.py     encoding, maintenance, normalized retrieval
  circuit.py        explicit recurrent-network form of maintenance (Supp. S3)
  tracking.py       trial that records the representation at every epoch
  paths.py          where results, data and figures live
  configs/          one YAML file per experiment family (Supp. Table S1)
experiments/      simulations; each writes shards and merged files to results/<family>/
  registry.py       every simulation task of the paper, used by all runners
figures/          one script per figure, plus tables.py and stats_report.py
results/          simulation outputs used by the figures (shipped)
  setsize/  retrocue/  bias/  circuit/
data/             published human data replotted in Figs. 2-4 (see data/README.md)
cluster/          Modal app and Slurm scripts for the full-scale runs
scripts/          run_tasks.py, make_figures.sh, smoke_test.sh
```

## Setup

Python 3.9 was used for all results.

```bash
git clone https://github.com/Daniel-Gong/MemGP-Code.git
cd MemGP-Code
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run every command from the repository root. Scripts are run as modules
(`python -m experiments.<name>`, `python -m figures.<name>`), which puts the
root on the import path; nothing needs to be installed.

## The model as simulated

- **Inducing points** start on a 10 × 10 grid over the torus (M = 100).
  `learn_inducing_locations` is true, so the points move during encoding and
  during the delay.
- **Encoding** fits the variational mean m and covariance S together with the
  inducing locations, kernel lengthscales, output scale and constant prior
  mean. Likelihood noise stays at its configured value. The lengthscales in
  Supplementary Table S1 are initial values.
- **Maintenance** continues that fit on the live inducing grid: m, S, inducing
  locations, kernel hyperparameters and the prior mean. The KL term relaxes
  the posterior toward the prior. After cue onset, attention reweights the
  objective. Likelihood noise stays fixed.
- **Circuit.** Figure 5 runs the delay as an explicit network. Inducing
  locations, the kernel and the likelihood are pinned at their encoding-end
  values, and Adam updates m and the covariance factor under
  W = A/(σ²M) + β K_ZZ⁻¹. In the matched comparison (Fig. 5c) the GP keeps
  updating through the delay; the network holds the geometry fixed.
- **Retrieval** draws one joint posterior sample of f at every occupied item
  location and reports the argmax of the divisively normalized response at the
  probed location (`memgp.simulation.retrieve_color` and `normalized_reports`;
  Supplementary S1.5). Every experiment uses this readout.

## Reproduce the figures and tables

```bash
bash scripts/make_figures.sh
```

This reads `results/` and `data/` and writes to `figures/output/`: the figures
as PDF and PNG (SVG and HTML for most), source-data CSVs for Figures 4 and 5,
`table_S1_parameters.tex`, `table_S2_information_criteria.{tex,csv}`, and
`stats_report.txt`, which recomputes every statistic quoted in the Results.

## Units and error bars

Behavioral panels use **20 simulated subjects** (24 in Fig. 4g, the size of the
human cohort it is compared with). Subject s draws every display from its own
seed range, and error bars are SEM across subject means. Pooled distributions
pool all trials of all subjects, as the human panels do. Figure 5 compares
with single-neuron and decoding studies, so the trial is the unit there.

| Panel | Simulated design | Plotted value | Error bar or band |
|---|---|---|---|
| 2f | 20 subjects × 100 displays per N ∈ {1,2,4,6,8}, one item probed per display | KDE of pooled signed errors | none |
| 2g | same data | circular SD per subject, averaged | SEM across subjects |
| 2h | same data | precision per subject, 1/SD − 1/SD_chance (rad⁻¹, Bays et al. 2009), averaged; power law fitted to the means | SEM across subjects |
| 2i | same data, N = 4 | histogram of pooled non-target-relative errors | none |
| 2j–l | 20 subjects × 50 displays per N per level of M, β or initial λ | mean absolute error per subject, averaged | SEM across subjects |
| 2m | 20 subjects × 30 displays per N ∈ {2,4,6,8}, 100 reports per probe | pooled misbinding rate per (N, distance) | none (per-cell SEM in source data) |
| 3f | 20 subjects × 100 trials per condition, N = 4 | mean absolute error, neutral vs cued | SEM across subjects |
| 3g | same, N = 2 and 4 | cue benefit (neutral − cued) | SEM across subjects |
| 3h | 20 subjects × 100 redirect trials | error for items A and B after each cue | SEM across subjects |
| 3i–j | the trials of 3f–g | KDE of pooled signed errors | none |
| 3k–l | 100 trials, N = 4, one maintenance each | change from cue onset in posterior mean (k) and variance (l), cued item versus the mean of the other items | SEM across trials |
| 4b–e (top) | 20 subjects × 100 trials per condition | bias % = 100·P(away) − 50 per subject, averaged | SEM across subjects |
| 4b–e (bottom) | same | histogram of pooled oriented errors | none |
| 4g | 24 subjects, σ ~ U[10°, 40°], 7 distances × 50 trials | circular SD and bias per subject, averaged over distances | none; Pearson r with least-squares line |
| 5c | 100 displays × {neutral, cued} | mean absolute error over 100 reports, GP vs network | none; Pearson r |
| 5d | 100 trials, N = 1 | absolute peak displacement per trial | trial-bootstrap 95% CI of the mean |
| 5e | 100 trials, N = 4 (400 items) | circular SD of reported peaks vs absolute error | item-bootstrap 95% CI of a local-linear fit |
| 5f–g | 100 paired trials, N = 4 | change from cue onset in posterior mean and variance | trial-bootstrap 95% CI of the mean |
| S2 | the 2f–h subjects | mixture w and CSD per subject, averaged; model predictions per subject, averaged | SEM across subjects |

Fig. 4g plots the mean oriented error in degrees, with trials of absolute
error above 40° excluded.

## Where each figure comes from

| Figure | Script | Inputs (`results/` unless noted) | Produced by |
|---|---|---|---|
| 1, 2a, 3a–b, 4a | figure scripts | none (schematics) | — |
| 2b–e | `figures/fig2_setsize.py` | `data/bays2009/bays2009_fig2_human.npz` | `data/bays2009/build_cache.py` |
| 2f–i | | `setsize/setsize_main_trials.csv` | `experiments.setsize_subjects main` |
| 2j, 2k, 2l | | `setsize/setsize_{M,beta,lambda}_trials.csv` | `experiments.setsize_subjects {M,beta,lambda}` |
| 2m | | `setsize/setsize_misbinding_trials.csv` | `experiments.setsize_subjects misbinding` |
| 3c–e | `figures/fig3_retrocue.py` | `data/souza2014/`, `data/vanmoorselaar2015/` | published data |
| 3f–g | | `retrocue/retrocue_setsize_{results,comparison}.csv` | `experiments.retrocue_subjects retrocue` + `aggregate` |
| 3h | | `retrocue/cue_flexibility_subjects.csv` | `experiments.retrocue_subjects flexibility` |
| 3i–j | | `retrocue/retrocue_trials_N{2,4}.csv` | `experiments.retrocue_subjects retrocue` |
| 3k–l | | `retrocue/retrocue_dynamics.npz` | `experiments.retrocue_dynamics` |
| 4b–e | `figures/fig4_bias.py` | `bias/adaptive_bias_confirm_{trials,summary}.csv` | `experiments.bias_adaptive confirm_shard --figure_only` |
| 4f | | `data/chunharas2022/` | published data |
| 4g | | `bias/bias_precision_trials.csv` | `experiments.bias_precision` |
| 5a–b | `figures/fig5_circuit.py` | computed from the kernel | — |
| 5c | | `circuit/rnn_equiv.npz` | `experiments.circuit_equivalence` |
| 5d | | `circuit/bump_diffusion.npz` | `experiments.circuit_bump_drift` |
| 5e | | `circuit/decoded_uncertainty.npz` | `experiments.circuit_decoded_uncertainty` |
| 5f–g | | `circuit/cued_precision.npz` | `experiments.circuit_cued_precision` |
| S1 | `figures/figS_gp_prior.py` | none (prior samples drawn in the script) | — |
| S2, Table S2 | `figures/figS_model_comparison.py` | `setsize/model_comparison_subjects_{ic,curves}.csv` | `experiments.setsize_model_comparison` |
| Table S1 | `figures/tables.py` | `memgp/configs/*.yaml` | — |

## Rerun the simulations

`experiments/registry.py` lists every simulation as an independent task (one
simulated subject, sweep level or circuit analysis): 389 tasks in stage 1, then
20 per-subject model fits in stage 2, which need the merged set-size data.
`python -m experiments.registry` prints them. Each task seeds every trial from
the subject and trial index, so tasks can run in any order on any machine.

**Modal** (used for the paper's results):

```bash
pip install modal && modal token new          # once
modal run cluster/modal_app.py --stage 1
python scripts/run_tasks.py --merge 1
modal run cluster/modal_app.py --stage 2
python scripts/run_tasks.py --merge 2
```

The local `modal run` process collects the results, so keep the machine awake
until it finishes.

**Slurm:**

```bash
bash cluster/setup_env.sh                     # once, on a login node
export MEMGP_ACCOUNT=<account> MEMGP_PARTITION=<cpu partition>
bash cluster/run_all.sh                       # both stages and merges, chained
```

**One machine:** `python scripts/run_tasks.py --jobs <cores>` runs both stages
and the merges. A single task: `python scripts/run_tasks.py --stage 1 --index 0`.

To write somewhere other than `results/`, set `MEMGP_RESULTS=/path`; figure
scripts read from the same variable, and `MEMGP_FIGURES` redirects their
output.

### Quick check

```bash
bash scripts/smoke_test.sh
```

runs every task with two subjects and a trial or two into a temporary
directory, merges, and renders all figures from it (about 15 minutes). It does
not touch `results/`.

### Fig. 4 design

The levels in Fig. 4b–e were fixed from pilot runs before the confirmatory run
and recorded in `results/bias/adaptive_bias_locked_design.json`. The pilot data
are in `results/bias/adaptive_bias_*pilot*`, and the `pilot`, `lock`,
`encoding_pilot` and `encoding_lock` modes of `experiments/bias_adaptive.py`
regenerate them.
