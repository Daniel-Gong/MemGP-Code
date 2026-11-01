"""Recurrent-circuit form of MemGP maintenance (manuscript Supplementary S3).

After a normal GP encoding, the delay can be run three ways from the same
encoded state:

  GP-full  ``run_branch(frozen=False)``: the real ``maintenance_phase`` with
           every parameter free (inducing locations, kernel hyperparameters,
           variational mean and covariance).
  frozen   ``run_branch_moments`` / ``run_branch(frozen=True)``: same update
           with the geometry pinned at its encoding-end value.
  network  ``explicit_maintenance``: the explicit recurrent network
           tau m' = -W m + b(t) for the mean and the matching flow for the
           Cholesky factor of S, with W = A / (sigma^2 M) + beta K_ZZ^{-1}
           (Eq. circuit). Within each epoch the rehearsal target t is fixed,
           so the loss is quadratic in m; after each Adam step t is refreshed
           from the predictive mean at Z and b is rebuilt. Attention switches
           A, hence W, at ``cue_start_epoch``.

The connectivity W is stiff, so a 100-epoch delay does not reach the fixed
point W^{-1} b; the network reproduces the transient that the GP follows.
"""
import copy

import numpy as np
import torch
import gpytorch

from memgp.gp_model import WorkingMemoryGP
from memgp.attention import SpatialProximityAttention
from memgp.simulation import maintenance_phase, device

RC_N = 4
CONDITIONS = ["neutral", "cued"]


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s)


def maintenance_optimizer(model, config, enc_opt_state=None):
    """Adam for the delay, warm-started from the encoding optimizer.

    ``run_simulation_trial`` hands the encoding optimizer straight on to
    ``maintenance_phase``, so its moment estimates are already warm when the
    delay begins.  A fresh Adam starts with a zero second moment, so its first
    steps move nearly a full learning rate in every coordinate and the delay
    destroys far more of the trace: at N=4 that is 96 deg of recall error
    against 44 deg for the warm optimizer on the same trials.  Reproducing the
    paper's dynamics therefore means carrying this state over.

    The optimizer covers every parameter so the state-dict indices line up with
    the encoding optimizer; frozen parameters receive no gradient and Adam skips
    them.
    """
    opt = torch.optim.Adam(list(model.parameters()),
                           lr=config["training"]["maintenance_lr"])
    if enc_opt_state is not None:
        opt.load_state_dict(copy.deepcopy(enc_opt_state))
        for group in opt.param_groups:
            group["lr"] = config["training"]["maintenance_lr"]
    return opt


def warm_adam_state(model, enc_opt_state, suffix):
    """Adam moments and step count for the parameter whose name ends in suffix.

    Lets the explicit integrators start from the same place the GP's optimizer
    does.  Returns None when the encoding run left no state for it.
    """
    if enc_opt_state is None:
        return None
    names = [n for n, _ in model.named_parameters()]
    idx = next((i for i, n in enumerate(names) if n.endswith(suffix)), None)
    state = enc_opt_state.get("state", {}).get(idx)
    if state is None:
        return None
    step = state["step"]
    step = float(step.item()) if torch.is_tensor(step) else float(step)
    return (state["exp_avg"].double().clone(),
            state["exp_avg_sq"].double().clone(), int(step))


def freeze_geometry(model):
    """Freeze everything except the variational mean -> pure linear m-dynamics."""
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith("variational_mean"))


def var_mean(model):
    return model.variational_strategy._variational_distribution.variational_mean.detach().clone()


def posterior_surface(model, n=60):
    locs = torch.linspace(-180, 180, n, device=device)
    cols = torch.linspace(-180, 180, n, device=device)
    L, C = torch.meshgrid(locs, cols, indexing="ij")
    grid = torch.stack([L.reshape(-1), C.reshape(-1)], dim=-1)
    model.eval()
    with torch.no_grad():
        m = model(grid).mean.reshape(n, n)
    return m.cpu().numpy()


def clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def fresh_model(config):
    model = WorkingMemoryGP(
        inducing_grid_size=config["model"]["inducing_grid_size"],
        loc_lengthscale=config["model"]["loc_lengthscale"],
        color_lengthscale=config["model"]["color_lengthscale"],
        learn_inducing_locations=config["model"]["learn_inducing_locations"],
    ).to(device)
    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)).to(device)
    lik.noise = config["likelihood"]["noise_variance"]
    lik.noise_covar.raw_noise.requires_grad_(False)
    return model, lik


def run_branch(enc_model_state, enc_lik_state, items, config, cue_idx, frozen,
               enc_opt_state=None):
    model, lik = fresh_model(config)
    model.load_state_dict(enc_model_state)
    lik.load_state_dict(enc_lik_state)
    model.double(); lik.double()                       # (a) maintenance in float64
    if frozen:
        freeze_geometry(model)
    opt = maintenance_optimizer(model, config, enc_opt_state)
    maintenance_phase(model, lik, opt, items, config,
                      config["training"]["maintenance_epochs"],
                      cued_item_idx=cue_idx)
    model.float(); lik.float()                          # read out in float32
    return model, lik


# ===========================================================================
# EXPLICIT linear recurrent network:  tau m' = -W m + b(t)
# ===========================================================================
class LinearRNN:
    """Within-step maintenance attractor as a recurrent network:

        tau m' = -W m + b ,     unique stable fixed point  m* = W^{-1} b .

    Under self-sustain, W = A/(sigma^2 M) + beta K_ZZ^{-1} is set by attention
    (precision) and is independent of the rehearsal target t; b is linear in t.
    After each discrete step, t is refreshed from the current mean so b moves
    with the state.  In the continuous limit t = m the A t terms cancel and
    attention enters only through W (covariance / KL forgetting).
    """
    def __init__(self, W, b):
        self.W = 0.5 * (W + W.t())                     # symmetrize tiny asym.
        self.b = b
        self.eig = torch.linalg.eigvalsh(self.W)

    def grad(self, m):
        return self.W @ m - self.b                     # = ∇L_maint(m), exactly

    def fixed_point(self):
        return torch.linalg.solve(self.W, self.b)


def _init_adam_state(x, state0):
    """Zero moments, or the encoding-end moments and step count when supplied."""
    if state0 is None:
        return torch.zeros_like(x), torch.zeros_like(x), 0
    exp_avg, exp_avg_sq, step = state0
    return (exp_avg.to(x.dtype).clone(), exp_avg_sq.to(x.dtype).clone(), step)


def adam_integrate_selfsustain(m0, Wn, Wc, bn_fn, bc_fn, cue_start, T, lr,
                               t0=None, refresh_t=None,
                               betas=(0.9, 0.999), eps=1e-8, record=False,
                               state0=None):
    """Self-sustain explicit RNN: after each Adam step, refresh the rehearsal
    target t via ``refresh_t(m)`` (predictive mean at Z; required under
    whitened variational strategies where m != E[f(Z)]), rebuild
    b = bn_fn(t)/bc_fn(t), and switch W at cue_start.

    ``state0`` carries the encoding-end Adam moments so the network integrates
    the same schedule the GP does (see ``maintenance_optimizer``)."""
    m = m0.clone()
    t = (t0 if t0 is not None else m0).clone()
    exp_avg, exp_avg_sq, k0 = _init_adam_state(m, state0)
    b1, b2 = betas
    traj = []
    for k in range(T):
        W = Wc if k >= cue_start else Wn
        b = bc_fn(t) if k >= cue_start else bn_fn(t)
        g = W @ m - b
        exp_avg.mul_(b1).add_(g, alpha=1 - b1)
        exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = 1 - b1 ** (k0 + k + 1)
        bc2 = 1 - b2 ** (k0 + k + 1)
        denom = (exp_avg_sq / bc2).sqrt().add_(eps)
        m = m - lr * (exp_avg / bc1) / denom
        if refresh_t is not None:
            t = refresh_t(m)
        else:
            t = m.detach().clone()
        if record:
            traj.append(m.clone())
    return traj if record else m


def run_branch_moments(enc_model_state, enc_lik_state, items, config, cue_idx,
                       enc_opt_state=None):
    """Maintenance with the geometry pinned but both variational moments free.

    This is the setting the circuit claim is about: W is the synaptic operator,
    so holding the inducing locations, kernel and likelihood fixed is the
    fixed-synapse condition.  The mean and the width are then exactly the two
    flows that ``adam_integrate_selfsustain`` and ``adam_integrate_covariance``
    integrate.
    """
    model, lik = fresh_model(config)
    model.load_state_dict(enc_model_state)
    lik.load_state_dict(enc_lik_state)
    model.double(); lik.double()
    for n, p in model.named_parameters():
        p.requires_grad_(n.endswith(("variational_mean", "chol_variational_covar")))
    for p in lik.parameters():
        p.requires_grad_(False)
    opt = maintenance_optimizer(model, config, enc_opt_state)
    maintenance_phase(model, lik, opt, items, config,
                      config["training"]["maintenance_epochs"],
                      cued_item_idx=cue_idx)
    model.float(); lik.float()
    return model, lik


def adam_integrate_covariance(C0, Wn, Wc, cue_start, T, lr, beta,
                              betas=(0.9, 0.999), eps=1e-8, record=False,
                              state0=None):
    """Explicit width dynamics under the same operator W that drives the mean.

    In whitened coordinates the covariance-dependent part of the maintenance
    loss is

        F(S) = 0.5 tr(W S) - 0.5 beta logdet S ,     S = tril(C) tril(C)' ,

    so the circuit carries a width alongside its mean and both are set by W.
    Its stationary point is S* = beta W^{-1} (Supplementary S4.3), which a
    100-epoch delay does not reach, so the flow is integrated rather than solved:
    Adam on C at the maintenance learning rate reproduces the delay transient.

    dF/dC = tril( (W - beta S^{-1}) tril(C) ).
    """
    C = C0.clone()
    exp_avg, exp_avg_sq, k0 = _init_adam_state(C, state0)
    b1, b2 = betas
    traj = []
    for k in range(T):
        W = Wc if k >= cue_start else Wn
        L = torch.tril(C)
        S = L @ L.transpose(-1, -2)
        g = torch.tril((W - beta * torch.linalg.inv(S)) @ L)
        exp_avg.mul_(b1).add_(g, alpha=1 - b1)
        exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1 = 1 - b1 ** (k0 + k + 1)
        bc2 = 1 - b2 ** (k0 + k + 1)
        denom = (exp_avg_sq / bc2).sqrt().add_(eps)
        C = C - lr * (exp_avg / bc1) / denom
        if record:
            traj.append(C.clone())
    return traj if record else C


def _bust_posterior_cache(model):
    """Drop GPyTorch memoized forward outputs after we rewrite m or C in place."""
    for obj in (model, getattr(model, "variational_strategy", None)):
        if obj is None:
            continue
        cache = getattr(obj, "_memoize_cache", None)
        if cache is not None:
            cache.clear()
    # train/eval also drops some lazily rebuilt buffers.
    model.train()
    model.eval()


def set_covar(model, C):
    """Write a whitened Cholesky factor into the variational distribution."""
    with torch.no_grad():
        p = model.variational_strategy._variational_distribution.chol_variational_covar
        p.copy_(torch.tril(C).to(p.dtype))
    _bust_posterior_cache(model)
    return model


def whitened_covar(model):
    with torch.no_grad():
        C = model.variational_strategy._variational_distribution.chol_variational_covar
        L = torch.tril(C).double()
        return L @ L.transpose(-1, -2)


def _chol_key():
    return ("variational_strategy._variational_distribution."
            "chol_variational_covar")


def explicit_maintenance(enc_model_state, enc_lik_state, items, config, cue_idx,
                         enc_opt_state=None, record=False, track_points=None):
    """Run the delay as the explicit network (mean + width under fixed W).

    Encoding geometry, kernel and likelihood stay pinned at their encoding-end
    values.  The rate vector m and whitened Cholesky C are integrated with the
    same warm Adam moments the GP delay inherits.  When ``track_points`` is set,
    posterior variance at those (loc, col) sites is evaluated after every epoch
    (and at delay onset) by writing the live (m, C) into a float32 readout copy.

    Returns
    -------
    model, lik
        Delay-end network, ready for the paper's sampling readout.
    info : dict
        ``m_traj`` / ``C_traj`` (lists, only if ``record`` or ``track_points``),
        ``mean_trajs`` / ``var_trajs`` (lists of length-T+1 arrays when
        ``track_points`` is set; index 0 is delay onset). Mean and variance
        come from the same forward pass.
    """
    # Warm moments need a live module whose named_parameters order matches the
    # encoding optimizer; load a throwaway copy for that bookkeeping.
    book, _ = fresh_model(config)
    book.load_state_dict(enc_model_state)
    warm_m = warm_adam_state(book, enc_opt_state, "variational_mean")
    warm_c = warm_adam_state(book, enc_opt_state, "chol_variational_covar")
    del book

    Wc, _, m0, bc_fn, refresh_t, t0, _ = build_linear_rnn(
        enc_model_state, enc_lik_state, items, config, cue_idx)
    if cue_idx is not None:
        Wn, _, _, bn_fn, _, _, _ = build_linear_rnn(
            enc_model_state, enc_lik_state, items, config, None)
    else:
        Wn, bn_fn = Wc, bc_fn
    Wn = 0.5 * (Wn + Wn.t())
    Wc = 0.5 * (Wc + Wc.t())

    T = config["training"]["maintenance_epochs"]
    cs = config["training"]["cue_start_epoch"]
    lr = config["training"]["maintenance_lr"]
    beta = float(config["training"]["beta"])
    need_traj = bool(record or track_points)
    C0 = enc_model_state[_chol_key()].double().clone()

    m_out = adam_integrate_selfsustain(
        m0, Wn, Wc, bn_fn, bc_fn, cs, T, lr=lr,
        t0=t0, refresh_t=refresh_t, record=need_traj, state0=warm_m)
    C_out = adam_integrate_covariance(
        C0, Wn, Wc, cs, T, lr, beta, record=need_traj, state0=warm_c)

    if need_traj:
        m_traj, C_traj = m_out, C_out
        m_end, C_end = m_traj[-1], C_traj[-1]
    else:
        m_traj = C_traj = None
        m_end, C_end = m_out, C_out

    model, lik = fresh_model(config)
    model.load_state_dict(enc_model_state)
    lik.load_state_dict(enc_lik_state)
    set_mean(model, m_end)
    set_covar(model, C_end)

    info = {}
    if need_traj:
        info["m_traj"] = m_traj
        info["C_traj"] = C_traj
    if track_points:
        readout, _ = fresh_model(config)
        readout.load_state_dict(enc_model_state)
        mean_trajs = [[] for _ in track_points]
        var_trajs = [[] for _ in track_points]

        def _record_moments(mvec, C):
            set_mean(readout, mvec)
            set_covar(readout, C)
            for i, (loc, col) in enumerate(track_points):
                mu, var = moments_at(readout, loc, col)
                mean_trajs[i].append(mu)
                var_trajs[i].append(var)

        _record_moments(m0, C0)                       # delay onset
        for mvec, C in zip(m_traj, C_traj):
            _record_moments(mvec, C)
        info["mean_trajs"] = [np.asarray(v, float) for v in mean_trajs]
        info["var_trajs"] = [np.asarray(v, float) for v in var_trajs]
    return model, lik, info


def record_gp_maintenance(enc_model_state, enc_lik_state, items, config, cue_idx, frozen,
                          track_points=None, enc_opt_state=None):
    """Replicate previous-step self-sustain `maintenance_phase` (float64); record m.

    If ``track_points`` is a list of (loc, col), also return a list of per-point
    posterior-variance trajectories evaluated on the live model (so covariance
    updates are captured).
    """
    model, lik = fresh_model(config)
    model.load_state_dict(enc_model_state)
    lik.load_state_dict(enc_lik_state)
    model.double(); lik.double()
    if frozen:
        freeze_geometry(model)
    opt = maintenance_optimizer(model, config, enc_opt_state)

    if cue_idx is not None:
        attn = SpatialProximityAttention(
            spatial_std=config["attention"]["spatial_std"],
            attended_gain=config["attention"]["attended_gain"]).to(device).double()
        cued_loc = items[cue_idx][0]
    else:
        attn, cued_loc = None, None
    cs = config["training"]["cue_start_epoch"]
    beta = float(config["training"]["beta"])
    T = config["training"]["maintenance_epochs"]

    with torch.no_grad():
        Z0 = model.variational_strategy.inducing_points
        prev_target = lik(model(Z0)).mean.detach()

    traj = []
    var_trajs = [[] for _ in (track_points or [])]
    model.train()
    for epoch in range(T):
        opt.zero_grad()
        Z = model.variational_strategy.inducing_points
        out = model(Z)
        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
        exp_ll = lik.expected_log_prob(prev_target, out)
        M = Z.shape[0]
        if attn is not None and epoch >= cs:
            w = attn(Z[:, 0].detach(), cued_loc)
        else:
            w = torch.ones(M, dtype=torch.float64, device=device)
        loss = -(exp_ll * w).sum() / M + kl * beta
        loss.backward(); opt.step()
        with torch.no_grad():
            model.eval()
            Z = model.variational_strategy.inducing_points
            prev_target = lik(model(Z)).mean.detach()
            if track_points:
                for i, (loc, col) in enumerate(track_points):
                    pt = torch.tensor([[loc, col]], dtype=torch.float64, device=device)
                    var_trajs[i].append(float(model(pt).variance))
            model.train()
        traj.append(var_mean(model).double().clone())
    if track_points is None:
        return traj
    return traj, [np.asarray(v, float) for v in var_trajs]


def moments_at(model, loc, col):
    """Posterior mean and variance at one site, from a single forward pass."""
    pt = torch.tensor([[loc, col]], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(pt)
        return float(pred.mean), float(pred.variance)


def activation_at(model, loc, col):
    return moments_at(model, loc, col)[0]


def variance_at(model, loc, col):
    return moments_at(model, loc, col)[1]


def _maint_grad(model, lik, Z, t, a, beta, M, mvec):
    """grad of the exact GP maintenance loss w.r.t. the inducing-rate vector."""
    mp = model.variational_strategy._variational_distribution.variational_mean
    with torch.no_grad():
        mp.copy_(mvec)
    mp.requires_grad_(True)
    model.train()
    out = model(Z)
    var_dist = model.variational_strategy.variational_distribution
    prior_dist = model.variational_strategy.prior_distribution
    kl = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
    exp_ll = lik.expected_log_prob(t, out)
    loss = -(exp_ll * a).sum() / M + kl * beta
    g = torch.autograd.grad(loss, mp)[0].detach().clone()
    return g


def build_linear_rnn(enc_model_state, enc_lik_state, items, config, cue_idx):
    """Construct W (attention-modulated Hessian) and b(t) from the GP loss.

    Returns (W, b0, m0, b_fn, ctx) where b_fn(t) rebuilds the drive for a
    rolled self-sustain target and ctx supports gradient verification.
    """
    model, lik = fresh_model(config)
    model.load_state_dict(enc_model_state)
    lik.load_state_dict(enc_lik_state)
    model.double(); lik.double()

    Z = model.variational_strategy.inducing_points.detach().double()
    M = Z.shape[0]
    beta = float(config["training"]["beta"])
    with torch.no_grad():
        t0 = lik(model(Z)).mean.detach().double()
    if cue_idx is None:
        a = torch.ones(M, device=device, dtype=torch.float64)
    else:
        attn = SpatialProximityAttention(
            spatial_std=config["attention"]["spatial_std"],
            attended_gain=config["attention"]["attended_gain"]).to(device).double()
        a = attn(Z[:, 0], items[cue_idx][0]).detach().double()

    m0 = model.variational_strategy._variational_distribution.variational_mean.detach().double().clone()
    zero = torch.zeros(M, dtype=torch.float64, device=device)

    def b_fn(t):
        return -_maint_grad(model, lik, Z, t, a, beta, M, zero)

    def refresh_t(mvec):
        """Map whitened variational mean -> predictive mean at inducing Z."""
        with torch.no_grad():
            mp = model.variational_strategy._variational_distribution.variational_mean
            mp.copy_(mvec)
            model.eval()
            t_new = lik(model(Z)).mean.detach().double()
            model.train()
        return t_new

    b0 = b_fn(t0)
    W = torch.empty(M, M, dtype=torch.float64, device=device)
    e = torch.zeros(M, dtype=torch.float64, device=device)
    for i in range(M):
        e.zero_(); e[i] = 1.0
        W[:, i] = _maint_grad(model, lik, Z, t0, a, beta, M, e) + b0
    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.copy_(m0)
    return W, b0, m0, b_fn, refresh_t, t0, (model, lik, Z, t0, a, beta, M)


def verify_against_gp(rnn, ctx, m0):
    """max | grad L_GP(m) - (W m - b) | over a few random m (should be ~1e-9)."""
    model, lik, Z, t, a, beta, M = ctx
    worst, scale = 0.0, 0.0
    for _ in range(3):
        m = m0 + 0.1 * torch.randn_like(m0)
        g = _maint_grad(model, lik, Z, t, a, beta, M, m)
        gr = rnn.W @ m - rnn.b
        worst = max(worst, float((g - gr).abs().max()))
        scale = max(scale, float(g.abs().max()))
    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.copy_(m0)
    return worst, scale


def set_mean(model, m):
    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.copy_(
            m.to(model.variational_strategy._variational_distribution.variational_mean.dtype))
    _bust_posterior_cache(model)
    return model

