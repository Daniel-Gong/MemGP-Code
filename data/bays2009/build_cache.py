#!/usr/bin/env python3
"""Rebuild bays2009_fig2_human.npz from the Bays et al. (2009) OSF release.

Reads ``data/bays2009/Exp_data.mat`` (downloads from https://osf.io/c2yx5/ if
missing), then writes subject-level precision / recall-SD summaries and
trial-level error arrays used by Figure 2 panels b–e.

Run:  python data/bays2009/build_cache.py
"""
import os
import urllib.request
import zipfile

import numpy as np
from scipy.io import loadmat

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = _HERE
_MAT = os.path.join(_SRC, "Exp_data.mat")
_ZIP = os.path.join(_SRC, "osf_c2yx5.zip")
_OUT = os.path.join(_HERE, "bays2009_fig2_human.npz")
_OSF_ZIP = "https://files.osf.io/v1/resources/c2yx5/providers/osfstorage/?zip="


def _ensure_mat():
    if os.path.isfile(_MAT):
        return
    os.makedirs(_SRC, exist_ok=True)
    print("Downloading Bays 2009 data from OSF (https://osf.io/c2yx5/)…")
    urllib.request.urlretrieve(_OSF_ZIP, _ZIP)
    with zipfile.ZipFile(_ZIP) as zf:
        zf.extractall(_SRC)
    if not os.path.isfile(_MAT):
        for root, _, files in os.walk(_SRC):
            for f in files:
                if f.endswith(".mat"):
                    src = os.path.join(root, f)
                    if os.path.abspath(src) != os.path.abspath(_MAT):
                        os.rename(src, _MAT)
                    break
    if not os.path.isfile(_MAT):
        raise FileNotFoundError(f"Could not find Exp_data.mat under {_SRC}")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def circ_sd(x):
    R = np.clip(np.abs(np.mean(np.exp(1j * np.asarray(x, float)))), 1e-12, 1.0)
    return float(np.sqrt(-2.0 * np.log(R)))


def main():
    _ensure_mat()
    d = loadmat(_MAT, squeeze_me=True, struct_as_record=False)
    err = np.asarray(d["error"], float)
    resp = np.asarray(d["response"], float)
    nt = np.asarray(d["nontarget"], float)
    n_items = np.asarray(d["n_items"], int)
    subj = np.asarray(d["subject"], int)

    SD_CHANCE = np.pi / np.sqrt(3.0)
    Ns = np.array([1, 2, 4, 6])
    subjects = np.unique(subj)

    prec_sub = np.zeros((len(subjects), len(Ns)))
    sd_deg_sub = np.zeros((len(subjects), len(Ns)))
    mae_deg_sub = np.zeros((len(subjects), len(Ns)))
    for si, s in enumerate(subjects):
        for ni, N in enumerate(Ns):
            m = (subj == s) & (n_items == N)
            e = err[m]
            sd = circ_sd(e)
            prec_sub[si, ni] = 1.0 / sd - 1.0 / SD_CHANCE
            sd_deg_sub[si, ni] = np.rad2deg(sd)
            mae_deg_sub[si, ni] = float(np.mean(np.abs(np.rad2deg(e))))

    def mean_sem(a):
        return a.mean(0), a.std(0, ddof=1) / np.sqrt(a.shape[0])

    prec_mean, prec_sem = mean_sem(prec_sub)
    sd_mean, sd_sem = mean_sem(sd_deg_sub)
    mae_mean, mae_sem = mean_sem(mae_deg_sub)

    errs_deg = {int(N): np.rad2deg(err[n_items == N]) for N in Ns}

    m4 = n_items == 4
    nt_err_deg = []
    for i in np.where(m4)[0]:
        for j in range(3):
            nt_err_deg.append(np.rad2deg(wrap(resp[i] - nt[i, j])))
    nt_err_deg = np.asarray(nt_err_deg, float)

    edges = np.arange(-180, 181, 12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    hist_sub = []
    for s in subjects:
        m = (subj == s) & (n_items == 4)
        e = []
        for i in np.where(m)[0]:
            for j in range(3):
                e.append(np.rad2deg(wrap(resp[i] - nt[i, j])))
        h, _ = np.histogram(e, bins=edges, density=True)
        hist_sub.append(h)
    hist_sub = np.asarray(hist_sub)

    np.savez_compressed(
        _OUT,
        Ns=Ns,
        precision_mean=prec_mean,
        precision_sem=prec_sem,
        recall_sd_deg_mean=sd_mean,
        recall_sd_deg_sem=sd_sem,
        mae_deg_mean=mae_mean,
        mae_deg_sem=mae_sem,
        err_deg_N1=errs_deg[1],
        err_deg_N2=errs_deg[2],
        err_deg_N4=errs_deg[4],
        err_deg_N6=errs_deg[6],
        nt_err_deg_N4=nt_err_deg,
        nt_hist_edges=edges,
        nt_hist_centers=centers,
        nt_hist_mean=hist_sub.mean(0),
        nt_hist_sem=hist_sub.std(0, ddof=1) / np.sqrt(len(subjects)),
        source=("Bays, Catalao & Husain (2009) OSF c2yx5; "
                "Ma et al. (2014) Fig 1b–c replot this dataset"),
        mat_path=_MAT,
    )
    print(f"Wrote {_OUT}")
    print(f"  from {_MAT}")
    print(f"  precision: {prec_mean}")
    print(f"  recall SD (deg): {sd_mean}")


if __name__ == "__main__":
    main()
