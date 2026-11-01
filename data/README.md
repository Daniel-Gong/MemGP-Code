# Human data

The human panels of Figures 2–4 replot published data. Nothing in this folder
was collected by us; cite the original papers when you reuse it, and check each
source's license before redistributing.

| Folder | Source | Used in | Files |
|---|---|---|---|
| `bays2009/` | Bays, Catalao & Husain (2009), *J. Vis.* 9(10):7, [doi:10.1167/9.10.7](https://doi.org/10.1167/9.10.7); trial data from [OSF c2yx5](https://osf.io/c2yx5/) | Fig. 2b–e | `Exp_data.mat` (raw trials), `bays2009_fig2_human.npz` (processed), `build_cache.py` |
| `souza2014/` | Souza, Rerko, Lin & Oberauer (2014), *Atten. Percept. Psychophys.* 76, 2085–2104, [doi:10.3758/s13414-014-0687-2](https://doi.org/10.3758/s13414-014-0687-2), Experiment 2 | Fig. 3c–d | trial-level CSVs for 2 and 4 items |
| `vanmoorselaar2015/` | van Moorselaar et al. (2015), *J. Exp. Psychol. Learn. Mem. Cogn.* 41(6), 1755–1763, [doi:10.1037/xlm0000124](https://doi.org/10.1037/xlm0000124), Experiment 2 | Fig. 3e | `fig2b_capacity.csv` |
| `chunharas2022/` | Chunharas, Rademaker, Brady & Serences (2022), *J. Exp. Psychol. Gen.*, Experiment 4; file from the authors' analysis code release (R folder) | Fig. 4f | `EXP4_subj_encd_coldis_val_circSTD.csv` |

## Notes per dataset

**Bays et al. (2009).** `Exp_data.mat` holds 7,271 trials from 12 subjects at
set sizes 1, 2, 4 and 6. Ma, Husain & Bays (2014) Fig. 1b–c replot the same
data. `build_cache.py` turns the raw trials into the arrays that
`figures/fig2_setsize.py` reads, and downloads `Exp_data.mat` from OSF if it is
missing:

```bash
python data/bays2009/build_cache.py
```

Fields of `bays2009_fig2_human.npz`:

- `err_deg_N*`: target-relative recall errors (deg), panel b
- `recall_sd_deg_{mean,sem}`: circular SD vs set size, panel c
- `precision_{mean,sem}`: precision (rad⁻¹; 1/SD minus chance), panel d
- `nt_err_deg_N4`, `nt_hist_*`: non-target-relative errors at N = 4, panel e

**Souza et al. (2014).** One row per trial; errors and colors in radians.
`condition_index` 0 is immediate report and 1 is delayed report after a
location cue (focused attention). The figure plots them as neutral and cued.
The repository does not record where this copy of the CSVs was downloaded
from.

**van Moorselaar et al. (2015).** Change-detection Cowan's *K* at load 8,
digitized from their Figure 2b. The header of the CSV documents the
digitization and how the four bars map onto the cue-A / cue-B axes of Fig. 3e.
The SEMs are read from plotted error bars and are approximate.

**Chunharas et al. (2022).** Columns (no header): subject, encoding time (s),
color distance, `val` = P(report away from distractor) − 0.5, circular SD
(deg). `figures/fig4_bias.py` averages over encoding × distance cells per
subject, which reproduces the subject means behind their reported
*r* = −0.81.
