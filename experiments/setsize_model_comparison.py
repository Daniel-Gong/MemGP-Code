"""
Set-size experiment + comparison to Item-Limit (IL), Equal-Precision (EP),
Slots-plus-Averaging (SA), and Variable-Precision (VP) models from
van den Berg, Shin, Chou, George, & Ma (PNAS 2012;
doi:10.1073/pnas.1117465109), using the same Fisher-information mapping
J = κ I₁(κ)/I₀(κ), κ = Φ(J) as in the supplement.

For each set size we fit a uniform + Von Mises mixture to signed estimation
errors (Zhang & Luck, 2008): w = mixture weight on the Von Mises component;
CSD = circular SD of that component, sqrt(-2 ln(I₁(κ)/I₀(κ))).

Empirical summaries are compared to predictions from each model after
maximum-likelihood parameter estimation on pooled circular errors.
Figure layout follows Figure 4A of the PNAS paper (2×4: w and CSD × four models).

The models are fitted separately to each simulated subject's errors from the
main set-size run (results/setsize/setsize_main_trials.csv), as van den Berg
et al. fitted each human subject.

Outputs (results/setsize/):
  model_comparison_subjects_ic.csv      AIC / BIC / AICc per subject and model (Table S2)
  model_comparison_subjects_curves.csv  per subject, N and model: mixture w and CSD,
                                        empirical and predicted (Fig. S2)

Run all subjects:            python -m experiments.setsize_model_comparison
One subject, then merge:     python -m experiments.setsize_model_comparison --subject_idx 4
                             python -m experiments.setsize_model_comparison --merge
"""
from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq, differential_evolution
from scipy.special import i0e, i1e
from scipy.stats import vonmises, gamma as gamma_dist

from memgp.paths import results_dir


# --- Fisher information ↔ Von Mises concentration (Eq. 2, supplement) ---


def J_from_kappa(kappa: float) -> float:
    if kappa <= 0:
        return 0.0
    # I₁/I₀ = i₁e/i₀e avoids overflow for large κ.
    return float(kappa * (i1e(kappa) / i0e(kappa)))


def kappa_from_J(J: float) -> float:
    """κ = Φ(J), inverse of J = κ I₁(κ)/I₀(κ)."""
    J = float(np.asarray(J).flat[0])
    if not np.isfinite(J) or J <= 0:
        return 1e-8
    lo = 1e-12
    flo = J_from_kappa(lo) - J
    hi = 50.0
    while hi < 1e8:
        fhi = J_from_kappa(hi) - J
        if flo * fhi <= 0:
            break
        hi *= 2
    else:
        return max(hi / 2, 1e-8)
    return float(brentq(lambda k: J_from_kappa(k) - J, lo, hi, xtol=1e-9, rtol=1e-7))


def csd_from_kappa(kappa: float) -> float:
    """Circular SD from Von Mises concentration (Mardia & Jupp)."""
    if kappa <= 0:
        return np.sqrt(2 * np.log(2))  # undefined limit; avoid NaN
    r = float(i1e(kappa) / i0e(kappa))
    r = max(r, 1e-12)
    return float(np.sqrt(-2.0 * np.log(r)))


def vm_convolution_density(err: np.ndarray, kappa_mem: float, kappa_r: float) -> np.ndarray:
    """
    Angular error x = r - s density when estimate ~ VM(κ_mem) and response noise ~ VM(κ_r)
    (Eq. S11 / convolution form in supplement).
    """
    kappa_mem = max(kappa_mem, 1e-10)
    kappa_r = max(kappa_r, 1e-10)
    err = np.asarray(err, dtype=float)
    rho = np.sqrt(
        kappa_mem**2 + kappa_r**2 + 2.0 * kappa_mem * kappa_r * np.cos(err)
    )
    # Use_log unstable at large κ; ratio via exponentials cancels.
    num = i0e(rho) * np.exp(rho - kappa_mem - kappa_r)
    den = (2.0 * np.pi) * i0e(kappa_mem) * i0e(kappa_r)
    return num / den


# --- Model log-densities for MLE ---


def log_density_il(err: np.ndarray, N: int, K: float, kappa_mem: float, kappa_r: float) -> np.ndarray:
    w = min(1.0, max(0.0, K / N))
    vm_d = vm_convolution_density(err, kappa_mem, kappa_r)
    if w >= 1.0 - 1e-12:
        return np.log(np.clip(vm_d, 1e-300, None))
    mix = w * vm_d + (1.0 - w) / (2.0 * np.pi)
    return np.log(np.clip(mix, 1e-300, None))


def log_density_ep(
    err: np.ndarray,
    N: int,
    J1: float,
    alpha: float,
    kappa_r: float,
) -> np.ndarray:
    """EP-A: all items always remembered; κ_mem = Φ(J₁ N^α)."""
    J = J1 * (N ** alpha)
    kappa_mem = kappa_from_J(max(J, 1e-12))
    vm_d = vm_convolution_density(err, kappa_mem, kappa_r)
    return np.log(np.clip(vm_d, 1e-300, None))


def log_density_sa(
    err: np.ndarray,
    N: int,
    K: float,
    J1: float,
    kappa_r: float,
) -> np.ndarray:
    kappa1 = kappa_from_J(max(J1, 1e-12))
    Kn = int(np.clip(round(K), 1, 64))

    if Kn <= N:
        w = min(1.0, Kn / N)
        vm_d = vm_convolution_density(err, kappa1, kappa_r)
        if w >= 1.0 - 1e-12:
            return np.log(np.clip(vm_d, 1e-300, None))
        mix = w * vm_d + (1.0 - w) / (2.0 * np.pi)
        return np.log(np.clip(mix, 1e-300, None))

    n_high = Kn % N
    n_low = N - n_high
    base = Kn // N
    if n_high == 0:
        kappa_est = kappa_from_J(max(J1 * base, 1e-12))
        vm_d = vm_convolution_density(err, kappa_est, kappa_r)
        return np.log(np.clip(vm_d, 1e-300, None))

    j_lo = J1 * base
    j_hi = J1 * (base + 1)
    kappa_low = kappa_from_J(max(j_lo, 1e-12))
    kappa_high = kappa_from_J(max(j_hi, 1e-12))
    w_hi = n_high / N
    w_lo = n_low / N
    p_hi = vm_convolution_density(err, kappa_high, kappa_r)
    p_lo = vm_convolution_density(err, kappa_low, kappa_r)
    mix = w_hi * p_hi + w_lo * p_lo
    return np.log(np.clip(mix, 1e-300, None))


def _gamma_J_grid(Jbar: float, tau: float, n_q: int = 96) -> np.ndarray:
    """Deterministic quadrature points for J ~ Gamma(mean=J̄, scale τ)."""
    Jbar = max(Jbar, 1e-8)
    tau = max(tau, 1e-8)
    shape = max(Jbar / tau, 1e-6)
    scale = tau
    qs = np.linspace(0.002, 0.998, n_q)
    return gamma_dist.ppf(qs, a=shape, scale=scale)


def log_density_vp(
    err: np.ndarray,
    N: int,
    J1_bar: float,
    alpha: float,
    tau: float,
    kappa_r: float,
    n_q: int = 96,
) -> np.ndarray:
    """VP-A (K=∞): average over J with mean J̄ = J̄₁ N^α, Gamma scale τ."""
    Jbar = J1_bar * (N ** alpha)
    Js = _gamma_J_grid(Jbar, tau, n_q=n_q)
    Js = np.clip(np.asarray(Js, float), 1e-10, None)
    err = np.asarray(err, dtype=float)
    out = np.zeros_like(err)
    for jv in Js:
        k_mem = kappa_from_J(float(jv))
        out += vm_convolution_density(err, k_mem, kappa_r)
    out /= len(Js)
    return np.log(np.clip(out, 1e-300, None))


# --- Uniform + Von Mises mixture (summary statistics) ---


def _neg_loglik_mixture(params: np.ndarray, data: np.ndarray) -> float:
    w, kappa = params
    w = np.clip(w, 1e-5, 1.0 - 1e-5)
    kappa = max(kappa, 1e-6)
    u = np.full_like(data, 1.0 / (2.0 * np.pi))
    vm = vonmises.pdf(x=data, kappa=kappa, scale=1.0)
    mix = (1.0 - w) * u + w * vm
    return -float(np.sum(np.log(np.clip(mix, 1e-300, None))))


def fit_uniform_vm_mixture(errors_rad: np.ndarray) -> Tuple[float, float, float]:
    """
    MLE for p(e) = (1-w)/(2π) + w VM(e; 0, κ).
    Returns (w, kappa, csd).
    """
    data = np.asarray(errors_rad, dtype=float)
    data = (data + np.pi) % (2.0 * np.pi) - np.pi

    best = (np.inf, None)
    for w0 in (0.4, 0.65, 0.85, 0.95):
        for k0 in (1.0, 6.0, 25.0, 90.0):
            res = minimize(
                _neg_loglik_mixture,
                x0=np.array([w0, k0]),
                args=(data,),
                method="L-BFGS-B",
                bounds=((1e-5, 1.0 - 1e-5), (1e-4, 500.0)),
            )
            if res.fun < best[0]:
                best = (res.fun, res.x)
    if best[1] is None:
        raise RuntimeError("mixture fit failed")
    w, kappa = float(best[1][0]), float(best[1][1])
    return w, kappa, csd_from_kappa(kappa)


# --- Simulate errors from fitted models (for ribbons / checks) ---


def sample_errors_il(
    rng: np.random.Generator, N: int, K: float, kappa_mem: float, kappa_r: float, n: int
) -> np.ndarray:
    out = np.empty(n)
    w = min(1.0, K / N)
    for i in range(n):
        if rng.random() < w:
            s_hat = vonmises.rvs(kappa_mem, random_state=rng)
            noise = vonmises.rvs(kappa_r, random_state=rng)
            out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
        else:
            out[i] = rng.uniform(-np.pi, np.pi)
    return out


def sample_errors_ep(
    rng: np.random.Generator, N: int, J1: float, alpha: float, kappa_r: float, n: int
) -> np.ndarray:
    J = J1 * (N ** alpha)
    kappa_mem = kappa_from_J(max(J, 1e-12))
    s_hat = vonmises.rvs(kappa_mem, size=n, random_state=rng)
    noise = vonmises.rvs(kappa_r, size=n, random_state=rng)
    return (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi


def sample_errors_sa(
    rng: np.random.Generator,
    N: int,
    K: float,
    J1: float,
    kappa_r: float,
    n: int,
) -> np.ndarray:
    kappa1 = kappa_from_J(max(J1, 1e-12))
    Kn = int(np.clip(round(K), 1, 64))
    out = np.empty(n)
    for i in range(n):
        if Kn <= N:
            w = min(1.0, Kn / N)
            if rng.random() < w:
                s_hat = vonmises.rvs(kappa1, random_state=rng)
                noise = vonmises.rvs(kappa_r, random_state=rng)
                out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
            else:
                out[i] = rng.uniform(-np.pi, np.pi)
            continue
        n_high = Kn % N
        if n_high == 0:
            base = Kn // N
            k_est = kappa_from_J(max(J1 * base, 1e-12))
            s_hat = vonmises.rvs(k_est, random_state=rng)
            noise = vonmises.rvs(kappa_r, random_state=rng)
            out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
            continue
        base = Kn // N
        j_lo = J1 * base
        j_hi = J1 * (base + 1)
        k_lo = kappa_from_J(max(j_lo, 1e-12))
        k_hi = kappa_from_J(max(j_hi, 1e-12))
        if rng.random() < n_high / N:
            s_hat = vonmises.rvs(k_hi, random_state=rng)
        else:
            s_hat = vonmises.rvs(k_lo, random_state=rng)
        noise = vonmises.rvs(kappa_r, random_state=rng)
        out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
    return out


def sample_errors_vp(
    rng: np.random.Generator,
    N: int,
    J1_bar: float,
    alpha: float,
    tau: float,
    kappa_r: float,
    n: int,
) -> np.ndarray:
    Jbar = J1_bar * (N ** alpha)
    shape = max(Jbar / max(tau, 1e-8), 1e-6)
    scale = max(tau, 1e-8)
    Js = rng.gamma(shape, scale, size=n)
    out = np.empty(n)
    for i, jv in enumerate(Js):
        k_mem = kappa_from_J(max(float(jv), 1e-12))
        s_hat = vonmises.rvs(k_mem, random_state=rng)
        noise = vonmises.rvs(kappa_r, random_state=rng)
        out[i] = (s_hat + noise + np.pi) % (2.0 * np.pi) - np.pi
    return out


# --- Collect MemGP experiment errors ---


@dataclass
class FitResult:
    name: str
    params: np.ndarray
    neg_ll: float
    x_pred_w: np.ndarray
    x_pred_csd: np.ndarray
    w_ribbon_lo: np.ndarray
    w_ribbon_hi: np.ndarray
    csd_ribbon_lo: np.ndarray
    csd_ribbon_hi: np.ndarray


def _pool_data(errors_by_n: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ns = []
    for N in sorted(errors_by_n.keys()):
        e = errors_by_n[N]
        xs.append(e)
        ns.append(np.full(len(e), N))
    return np.concatenate(xs), np.concatenate(ns)


def mle_objective(
    theta: np.ndarray,
    err: np.ndarray,
    n_arr: np.ndarray,
    logpdf: Callable[[np.ndarray, int], np.ndarray],
) -> float:
    ll = 0.0
    for N in np.unique(n_arr):
        mask = n_arr == N
        ll += float(np.sum(logpdf(err[mask], int(N))))
    return -ll


def _lbfgs_polish(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
) -> Tuple[np.ndarray, float]:
    """Refine DE solution (same box bounds)."""
    res = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 280},
    )
    return res.x, float(res.fun)


def fit_all_models(
    errors_by_n: Dict[int, np.ndarray],
    set_sizes: List[int],
    n_synth_avg: int = 10,
    seed: int = 0,
    de_maxiter: int = 60,
) -> Tuple[Dict[int, Tuple[float, float, float, float]], Dict[str, FitResult]]:
    """
    Returns:
      empirical: N -> (w, csd, w_boot_sd, csd_boot_sd)
      fits: model name -> FitResult with mean predicted w/csd and ribbons
    """
    rng = np.random.default_rng(seed)

    # Empirical mixture + bootstrap SEM
    empirical: Dict[int, Tuple[float, float, float, float]] = {}
    n_boot = 80
    for N in set_sizes:
        e = errors_by_n[N]
        w, kappa, csd = fit_uniform_vm_mixture(e)
        ws, cs = [], []
        m = len(e)
        for _ in range(n_boot):
            idx = rng.integers(0, m, size=m)
            wb, _, cb = fit_uniform_vm_mixture(e[idx])
            ws.append(wb)
            cs.append(cb)
        empirical[N] = (w, csd, float(np.std(ws, ddof=1)), float(np.std(cs, ddof=1)))

    err_all, n_all = _pool_data(errors_by_n) 
    trials_per_n = {N: len(errors_by_n[N]) for N in set_sizes}

    def pred_curves(
        sampler: Callable[[np.random.Generator, int, int], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mean w, mean CSD, and 10–90% ribbons from synthetic replicate mixture fits."""
        Ws = []
        Cs = []
        for _ in range(n_synth_avg):
            row_w = []
            row_c = []
            for N in set_sizes:
                syn = sampler(rng, N, trials_per_n[N])
                ww, _, cc = fit_uniform_vm_mixture(syn)
                row_w.append(ww)
                row_c.append(cc)
            Ws.append(row_w)
            Cs.append(row_c)
        Ws = np.asarray(Ws)
        Cs = np.asarray(Cs)
        w_mean = Ws.mean(axis=0)
        c_mean = Cs.mean(axis=0)
        w_lo = np.percentile(Ws, 10, axis=0)
        w_hi = np.percentile(Ws, 90, axis=0)
        c_lo = np.percentile(Cs, 10, axis=0)
        c_hi = np.percentile(Cs, 90, axis=0)
        return w_mean, c_mean, w_lo, w_hi, c_lo, c_hi

    fits: Dict[str, FitResult] = {}

    # --- IL ---
    def pack_il(x):
        K, lk_mem, lk_r = x
        return float(np.clip(K, 0.5, 20.0)), float(np.exp(lk_mem)), float(np.exp(lk_r))

    def obj_il(xx):
        K, km, kr = pack_il(xx)
        def logp(e, N):
            return log_density_il(e, N, K, km, kr)
        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_il = [(0.5, 12.0), (-2.5, 4.5), (-2.5, 4.5)]
    r_il = differential_evolution(
        lambda x: obj_il(x), bounds_il, seed=seed + 1, maxiter=de_maxiter, tol=0.05, polish=False
    )
    x_il, nll_il = _lbfgs_polish(obj_il, r_il.x, bounds_il)
    if obj_il(r_il.x) < nll_il:
        x_il, nll_il = r_il.x, float(obj_il(r_il.x))
    K_il, km_il, kr_il = pack_il(x_il)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_il(rg, N, K_il, km_il, kr_il, ntr),
    )
    fits["IL"] = FitResult(
        "IL", np.array([K_il, km_il, kr_il]), nll_il,
        wm, cm, wl, wh, cl, ch,
    )

    # --- SA ---
    def pack_sa(x):
        K, lJ1, lk_r = x
        return float(np.clip(K, 0.5, 20.0)), float(np.exp(lJ1)), float(np.exp(lk_r))

    def obj_sa(xx):
        K, J1, kr = pack_sa(xx)
        def logp(e, N):
            return log_density_sa(e, N, K, J1, kr)
        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_sa = [(0.5, 12.0), (-2.0, 4.5), (-2.5, 4.5)]
    r_sa = differential_evolution(
        lambda x: obj_sa(x), bounds_sa, seed=seed + 2, maxiter=de_maxiter, tol=0.05, polish=False
    )
    x_sa, nll_sa = _lbfgs_polish(obj_sa, r_sa.x, bounds_sa)
    if obj_sa(r_sa.x) < nll_sa:
        x_sa, nll_sa = r_sa.x, float(obj_sa(r_sa.x))
    K_sa, J1_sa, kr_sa = pack_sa(x_sa)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_sa(rg, N, K_sa, J1_sa, kr_sa, ntr),
    )
    fits["SA"] = FitResult(
        "SA", np.array([K_sa, J1_sa, kr_sa]), nll_sa,
        wm, cm, wl, wh, cl, ch,
    )

    # --- EP ---
    def pack_ep(x):
        lJ1, alpha, lk_r = x
        return float(np.exp(lJ1)), float(alpha), float(np.exp(lk_r))

    def obj_ep(xx):
        J1, al, kr = pack_ep(xx)
        def logp(e, N):
            return log_density_ep(e, N, J1, al, kr)
        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_ep = [(-1.5, 4.5), (-2.5, 1.5), (-2.5, 4.5)]
    r_ep = differential_evolution(
        lambda x: obj_ep(x), bounds_ep, seed=seed + 3, maxiter=de_maxiter, tol=0.05, polish=False
    )
    x_ep, nll_ep = _lbfgs_polish(obj_ep, r_ep.x, bounds_ep)
    if obj_ep(r_ep.x) < nll_ep:
        x_ep, nll_ep = r_ep.x, float(obj_ep(r_ep.x))
    J1_ep, al_ep, kr_ep = pack_ep(x_ep)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_ep(rg, N, J1_ep, al_ep, kr_ep, ntr),
    )
    fits["EP"] = FitResult(
        "EP", np.array([J1_ep, al_ep, kr_ep]), nll_ep,
        wm, cm, wl, wh, cl, ch,
    )

    # --- VP ---
    def pack_vp(x):
        lj1, alpha, ltau, lk_r = x
        return float(np.exp(lj1)), float(alpha), float(np.exp(ltau)), float(np.exp(lk_r))

    def obj_vp(xx):
        J1b, al, tau, kr = pack_vp(xx)

        def logp(e, N):
            return log_density_vp(e, N, J1b, al, tau, kr, n_q=112)

        return mle_objective(xx, err_all, n_all, lambda e, N: logp(e, N))

    bounds_vp = [(-1.5, 4.5), (-2.5, 1.5), (-3.0, 2.0), (-2.5, 4.5)]
    r_vp = differential_evolution(
        lambda x: obj_vp(x),
        bounds_vp,
        seed=seed + 4,
        maxiter=de_maxiter,
        tol=0.05,
        polish=False,
    )
    x_vp, nll_vp = _lbfgs_polish(obj_vp, r_vp.x, bounds_vp)
    if obj_vp(r_vp.x) < nll_vp:
        x_vp, nll_vp = r_vp.x, float(obj_vp(r_vp.x))
    J1_vp, al_vp, tau_vp, kr_vp = pack_vp(x_vp)

    wm, cm, wl, wh, cl, ch = pred_curves(
        lambda rg, N, ntr: sample_errors_vp(rg, N, J1_vp, al_vp, tau_vp, kr_vp, ntr),
    )
    fits["VP"] = FitResult(
        "VP", np.array([J1_vp, al_vp, tau_vp, kr_vp]), nll_vp,
        wm, cm, wl, wh, cl, ch,
    )

    return empirical, fits


# --- BIC / AIC model comparison ---

_MODEL_N_PARAMS = {"IL": 3, "SA": 3, "EP": 3, "VP": 4}


def compute_information_criteria(
    fits: Dict[str, FitResult],
    errors_by_n: Dict[int, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute AIC, BIC, and AICc for each fitted model.

    Parameters
    ----------
    fits : dict of FitResult
    errors_by_n : dict mapping set size -> array of errors (radians)

    Returns
    -------
    dict : model_name -> {"neg_ll", "k", "n", "AIC", "BIC", "AICc", "delta_AIC", "delta_BIC"}
    """
    n_total = sum(len(v) for v in errors_by_n.values())
    results = {}
    for name, fr in fits.items():
        k = _MODEL_N_PARAMS[name]
        nll = fr.neg_ll
        ll = -nll

        aic = 2 * k - 2 * ll
        bic = k * np.log(n_total) - 2 * ll
        if n_total - k - 1 > 0:
            aicc = aic + (2 * k * (k + 1)) / (n_total - k - 1)
        else:
            aicc = aic

        results[name] = {
            "neg_ll": nll,
            "k": k,
            "n": n_total,
            "AIC": aic,
            "BIC": bic,
            "AICc": aicc,
        }

    min_aic = min(r["AIC"] for r in results.values())
    min_bic = min(r["BIC"] for r in results.values())
    for name in results:
        results[name]["delta_AIC"] = results[name]["AIC"] - min_aic
        results[name]["delta_BIC"] = results[name]["BIC"] - min_bic

    return results


MODELS = ["IL", "SA", "EP", "VP"]
OUT_DIR = results_dir("setsize")


def subject_errors(trials: pd.DataFrame, subject: int) -> Dict[int, np.ndarray]:
    """Signed errors in radians, keyed by set size, for one subject."""
    sub = trials[trials["subject"] == subject]
    out = {}
    for N, g in sub.groupby("N"):
        rad = np.deg2rad(g["error"].to_numpy(float))
        out[int(N)] = (rad + np.pi) % (2.0 * np.pi) - np.pi
    return out


def fit_subject(trials: pd.DataFrame, subject: int, de_maxiter: int = 60):
    errors_by_n = subject_errors(trials, subject)
    set_sizes = sorted(errors_by_n)
    empirical, fits = fit_all_models(
        errors_by_n, set_sizes, n_synth_avg=10, seed=subject, de_maxiter=de_maxiter
    )
    ic = compute_information_criteria(fits, errors_by_n)
    ic_rows = [dict(subject=subject, model=m, k=ic[m]["k"], n=ic[m]["n"],
                    neg_loglik=ic[m]["neg_ll"], AIC=ic[m]["AIC"], BIC=ic[m]["BIC"],
                    AICc=ic[m]["AICc"], delta_AIC=ic[m]["delta_AIC"],
                    delta_BIC=ic[m]["delta_BIC"]) for m in MODELS]
    curve_rows = []
    for i, N in enumerate(set_sizes):
        w, csd, _, _ = empirical[N]
        curve_rows.append(dict(subject=subject, N=N, model="empirical", w=w, csd=csd))
        for m in MODELS:
            curve_rows.append(dict(subject=subject, N=N, model=m,
                                   w=float(fits[m].x_pred_w[i]),
                                   csd=float(fits[m].x_pred_csd[i])))
    return pd.DataFrame(ic_rows), pd.DataFrame(curve_rows)


def _shard_dir():
    path = OUT_DIR / "shards" / "model_comparison"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_subject(trials, subject, de_maxiter):
    ic, curves = fit_subject(trials, subject, de_maxiter)
    ic.to_csv(_shard_dir() / f"ic_subject_{subject:03d}.csv", index=False)
    curves.to_csv(_shard_dir() / f"curves_subject_{subject:03d}.csv", index=False)
    best = ic.loc[ic["AIC"].idxmin(), "model"]
    print(f"subject {subject}: best by AIC = {best}", flush=True)


def merge():
    for kind in ("ic", "curves"):
        files = sorted(glob.glob(str(_shard_dir() / f"{kind}_subject_*.csv")))
        if not files:
            raise SystemExit("no model-comparison shards")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        out = OUT_DIR / f"model_comparison_subjects_{kind}.csv"
        df.to_csv(out, index=False)
        print(f"wrote {out} ({df['subject'].nunique()} subjects)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject_idx", type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--de_maxiter", type=int, default=60)
    args = ap.parse_args()

    if args.merge:
        merge()
        return
    trials = pd.read_csv(OUT_DIR / "setsize_main_trials.csv")
    trials = trials[trials["probed"]]
    subjects = ([args.subject_idx] if args.subject_idx is not None
                else sorted(trials["subject"].unique()))
    for s in subjects:
        run_subject(trials, int(s), args.de_maxiter)
    if args.subject_idx is None:
        merge()


if __name__ == "__main__":
    main()


