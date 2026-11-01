import torch
import gpytorch
from typing import List, Tuple, Dict, Optional
import yaml
import os
from memgp.generator import sample_training_data, circular_error
from memgp.gp_model import WorkingMemoryGP
from memgp.attention import SpatialProximityAttention
from memgp.paths import CONFIG_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_threads(default: Optional[int] = None) -> int:
    """Set the torch intra-op thread count from the environment.

    Honors ``$SLURM_CPUS_PER_TASK`` (set by Slurm) or ``$TORCH_NUM_THREADS`` so
    that cluster array tasks use exactly the cores they were allocated. When
    neither is set the current torch default (or ``default``) is left in place,
    so local runs are unaffected.
    """
    n = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("TORCH_NUM_THREADS")
    if n is None and default is None:
        return torch.get_num_threads()
    try:
        n_threads = max(1, int(n)) if n is not None else int(default)
    except (TypeError, ValueError):
        return torch.get_num_threads()
    torch.set_num_threads(n_threads)
    return n_threads


# Apply cluster thread allocation automatically on import (no-op locally).
configure_threads()


def load_config(path=None, filename="config_set_size.yaml"):
    """Load a YAML config from ``memgp/configs/`` (or from an explicit path)."""
    if path is None:
        path = CONFIG_DIR / filename
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def _spatial_divisive_normalize(
    model: "WorkingMemoryGP",
    query_loc: float,
    loc_pool_width_deg: float,
    baseline_frac: float = 0.1,
    n_color_samples: int = 360,
    n_loc_samples: int = 36,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Spatial divisive normalization (Carandini & Heeger, 2012).

    For each candidate color *c*, the normalization pool aggregates the GP
    posterior across nearby *spatial locations* (weighted by a Gaussian
    centered on ``query_loc``).  This lets the distractor's peak at a
    different location contribute to the pool, creating surround suppression
    that shifts the target peak away from the distractor.

    Returns (color_samples, normalized_profile).
    """
    model.eval()

    colors = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    locs = torch.linspace(-180.0, 180.0, n_loc_samples + 1, device=device)[:-1]

    # --- evaluate the full 2-D surface in one batch ---
    L, C = torch.meshgrid(locs, colors, indexing="ij")
    grid = torch.stack([L.reshape(-1), C.reshape(-1)], dim=-1)
    with torch.no_grad():
        surface = model(grid).mean.view(n_loc_samples, n_color_samples)

    surface_shifted = surface - surface.min()

    # --- spatial weights (Gaussian centered on query_loc) ---
    loc_dists = torch.abs(locs - query_loc)
    loc_dists = torch.min(loc_dists, 360.0 - loc_dists)
    spatial_w = torch.exp(-0.5 * (loc_dists / loc_pool_width_deg) ** 2)
    spatial_w = spatial_w / spatial_w.sum()

    # Spatially-weighted pool for each color:  shape (n_color_samples,)
    pool = (surface_shifted * spatial_w.unsqueeze(1)).sum(dim=0)

    # --- target response at query_loc ---
    query_idx = torch.argmin(torch.abs(locs - query_loc))
    target_response = surface_shifted[query_idx]

    sigma = baseline_frac * target_response.max().clamp(min=1e-8)
    normalized = target_response / (sigma + pool)

    return colors, normalized


def _feature_divisive_normalize(
    model: "WorkingMemoryGP",
    query_loc: float,
    item_locs: List[float],
    loc_pool_width_deg: float,
    baseline_frac: float = 0.5,
    n_color_samples: int = 360,
    probabilistic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified feature-space divisive normalization (Carandini & Heeger, 2012).

    Rather than evaluating the full 2-D loc×color surface and pooling over a
    dense grid of spatial locations, the normalization pool is built only from
    the *occupied* item locations.  For each item *i* we read the GP color
    tuning slice ``f_i(c)`` at its spatial location and weight it by a spatial
    surround gain that decays with its distance from the probed location::

        g_i  = exp(-0.5 * (Δloc_i / loc_pool_width)^2)            # target → 1
        pool(c) = Σ_i g_i · f_i^+(c)
        r(c)    = f_target^+(c) / (baseline_frac · max_c f_target^+ + pool(c))

    Two interpretable knobs:
        loc_pool_width  – spatial reach of the surround (sets distractor gain)
        baseline_frac   – semi-saturation constant (overall normalization gain)

    This collapses to the same attraction→repulsion→decay curve as the dense
    2-D pool but only evaluates ``len(item_locs)`` color slices, and it
    generalizes naturally to any number of distractors.

    Returns (color_samples, normalized_profile).
    """
    model.eval()

    colors = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    locs_t = torch.tensor([float(l) for l in item_locs], device=device)

    # Evaluate the color tuning slice at each occupied item location in one batch
    rows = [
        torch.stack([torch.full_like(colors, float(l)), colors], dim=-1)
        for l in item_locs
    ]
    grid = torch.cat(rows, dim=0)
    with torch.no_grad():
        pred = model(grid)
        # Probabilistic readout: draw a single joint sample from the GP
        # posterior over all (item × color) candidates, preserving the
        # inter-color covariance induced by the kernel.  Deterministic readout
        # uses the posterior mean (the zero-variance limit of the same draw).
        field = pred.sample() if probabilistic else pred.mean
        slices = field.view(len(item_locs), n_color_samples)

    normalized = normalize_slices(slices, locs_t, query_loc,
                                  loc_pool_width_deg, baseline_frac)
    return colors, normalized


def normalize_slices(slices, item_locs, query_loc, loc_pool_width_deg,
                     baseline_frac):
    """Divisive normalization of color slices read at the occupied locations.

    ``slices`` has shape (..., n_items, n_colors): one or more joint samples
    (or the posterior mean) of f at every item location. Returns the
    normalized response r(c) at ``query_loc`` with shape (..., n_colors)::

        f_i^+   = f_i - min f                     (min over all items, colors)
        g_i     = exp(-0.5 * (d(l_i, l_probe) / loc_pool_width)^2)
        pool(c) = sum_i g_i f_i^+(c)
        r(c)    = f_probe^+(c) / (baseline_frac * max_c f_probe^+ + pool(c))
    """
    slices = torch.as_tensor(slices)
    locs = torch.as_tensor(item_locs, dtype=slices.dtype, device=slices.device)
    shifted = slices - slices.amin(dim=(-2, -1), keepdim=True)
    loc_dists = torch.abs(locs - query_loc)
    loc_dists = torch.min(loc_dists, 360.0 - loc_dists)
    gains = torch.exp(-0.5 * (loc_dists / loc_pool_width_deg) ** 2)
    pool = (shifted * gains.unsqueeze(-1)).sum(dim=-2)
    target = shifted[..., int(torch.argmin(loc_dists)), :]
    sigma = baseline_frac * target.amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return target / (sigma + pool)


def normalized_reports(model, item_locs, normalization, n_draws=1,
                       n_color_samples=360):
    """Reported colors for every probe, from ``n_draws`` joint posterior samples.

    Each draw samples f jointly at all occupied locations (as in
    ``retrieve_color``); every item is then read out as the probe with the
    divisive normalization above. Returns an array of shape
    (n_draws, n_items) in degrees.
    """
    model.eval()
    colors = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    grid = torch.cat([
        torch.stack([torch.full_like(colors, float(l)), colors], dim=-1)
        for l in item_locs
    ], dim=0)
    with torch.no_grad():
        fields = model(grid).sample(torch.Size([n_draws]))
    slices = fields.view(n_draws, len(item_locs), n_color_samples)
    out = torch.empty(n_draws, len(item_locs))
    for j, loc in enumerate(item_locs):
        r = normalize_slices(slices, item_locs, float(loc),
                             normalization["loc_pool_width"],
                             normalization.get("baseline_frac", 0.5))
        out[:, j] = colors[torch.argmax(r, dim=-1)].cpu()
    return out.numpy()


def retrieve_color(
    model: WorkingMemoryGP,
    query_loc: float,
    true_color: float,
    n_color_samples: int = 360,
    normalization: Optional[Dict] = None,
    item_locs: Optional[List[float]] = None,
    probabilistic: bool = True,
) -> float:
    """
    Retrieve color at a given 1D location by finding the color with maximum GP prediction.

    Parameters
    ----------
    normalization : dict or None
        If provided, apply divisive normalization before argmax.
        Expected keys:
            loc_pool_width  – spatial pool / surround width (degrees)
            baseline_frac   – semi-saturation constant as fraction of peak
    item_locs : list of float or None
        Spatial locations of all items in the display (including the target).
        When provided, the simplified feature-space DN is used, building the
        normalization pool only from these occupied locations.  When ``None``,
        the legacy dense 2-D spatial pool (`_spatial_divisive_normalize`) is
        used for backward compatibility.
    probabilistic : bool
        The readout argmax is taken over a single joint sample drawn from the
        GP posterior predictive (preserving inter-color covariance). Pass
        False to use the posterior mean. Supported for the feature-space
        divisive-normalization path and the un-normalized path.
    """
    model.eval()

    if normalization is not None:
        if item_locs is not None:
            colors, profile = _feature_divisive_normalize(
                model,
                query_loc,
                item_locs=item_locs,
                loc_pool_width_deg=normalization["loc_pool_width"],
                baseline_frac=normalization.get("baseline_frac", 0.5),
                n_color_samples=n_color_samples,
                probabilistic=probabilistic,
            )
        else:
            colors, profile = _spatial_divisive_normalize(
                model,
                query_loc,
                loc_pool_width_deg=normalization["loc_pool_width"],
                baseline_frac=normalization.get("baseline_frac", 0.1),
                n_color_samples=n_color_samples,
            )
        best_idx = torch.argmax(profile)
        best_color = colors[best_idx].item()
    else:
        color_samples = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
        query_points = torch.stack([
            torch.tensor([query_loc, c], device=device) for c in color_samples
        ])
        with torch.no_grad():
            pred = model(query_points)
            field = pred.sample() if probabilistic else pred.mean
            best_idx = torch.argmax(field)
            best_color = color_samples[best_idx].item()

    return circular_error(best_color, true_color)

def retrieve_color_probabilistic(
    model: WorkingMemoryGP,
    query_loc: float,
    true_color: float,
    n_color_samples: int = 360,
) -> Tuple[float, float]:
    """
    Retrieve color via posterior predictive sampling + argmax.

    Draws a single joint sample f ~ q(f) from the full GP posterior
    MultivariateNormal over all candidate colors simultaneously, then
    returns the color at which the sampled function value is highest.

    Using pred.sample() correctly respects the inter-color covariance
    induced by the GP kernel (nearby colors are positively correlated).

    WM error components emerge naturally from GP posterior geometry:
      - Narrow posterior (concentrated around true color) → low SD
      - Wide posterior (uncertain encoding)               → high SD
      - Secondary activation peak at a non-target color   → swap error
      - Flat, uninformative posterior                     → guess error

    Returns (signed_error, unsigned_error).
    """
    model.eval()
    color_samples = torch.linspace(-180.0, 180.0, n_color_samples, device=device)
    query_points = torch.stack([
        torch.tensor([query_loc, c], device=device) for c in color_samples
    ])

    with torch.no_grad():
        # Query the GP posterior: a MultivariateNormal over all query points
        # jointly, preserving the inter-color covariance from the kernel.
        pred = model(query_points)

        # Draw one joint sample: shape (n_color_samples,)
        # Adjacent colors co-vary smoothly — this is correctly captured here.
        f_sample = pred.sample()

        # Pick the color with the highest activation in this sample
        best_idx = torch.argmax(f_sample)
        recalled_color = color_samples[best_idx].item()

    return circular_error(recalled_color, true_color)

def setup_trial(
    items: List[Tuple[float, float]],
    config: dict,
) -> Tuple[torch.Tensor, torch.Tensor, WorkingMemoryGP,
           gpytorch.likelihoods.GaussianLikelihood,
           torch.optim.Optimizer, gpytorch.mlls.VariationalELBO]:
    """
    Build the training samples, GP model, likelihood, optimizer and MLL for a
    single trial.  Factored out of ``run_simulation_trial`` so that callers
    (e.g. parameter sweeps) can drive the encoding/maintenance phases directly
    and reuse an encoded model across multiple maintenance conditions.
    """
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config['data']['n_samples_per_item'],
        loc_std=config['data']['loc_std'],
        color_std=config['data']['color_std'],
        loc_encoding_noise_std=config['data']['loc_encoding_noise_std'],
        color_encoding_noise_std=config['data']['color_encoding_noise_std']
    )

    model = WorkingMemoryGP(
        inducing_grid_size=config['model']['inducing_grid_size'],
        loc_lengthscale=config['model']['loc_lengthscale'],
        color_lengthscale=config['model']['color_lengthscale'],
        learn_inducing_locations=config['model']['learn_inducing_locations']
    ).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config['likelihood']['noise_variance']

    # By default the observation noise is frozen (the GP is a near-exact
    # interpolator of the encoded trace; uncertainty comes from the
    # variational / inducing structure).  Set likelihood.learn_noise=true to
    # instead learn it jointly with the model (data-driven, homoscedastic).
    learn_noise = bool(config['likelihood'].get('learn_noise', False))
    if learn_noise:
        params = list(model.parameters()) + list(likelihood.parameters())
    else:
        likelihood.noise_covar.raw_noise.requires_grad_(False)  # Fixed — not learned
        params = list(model.parameters())  # likelihood excluded: noise is frozen

    optimizer = torch.optim.Adam(
        params,
        lr=config['training']['encoding_lr']
    )

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))
    return samples, weights, model, likelihood, optimizer, mll


def encoding_phase(model, likelihood, optimizer, mll, samples, weights,
                   encoding_epochs, encoding_lr=None, log_cb=None):
    """
    Run the encoding phase (fit the GP to the bottom-up sensory samples).

    ``log_cb(loss, is_maint=False)`` is called after every epoch when provided.
    """
    if encoding_lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = encoding_lr
    model.train()
    likelihood.train()
    for epoch in range(encoding_epochs):
        optimizer.zero_grad()
        output = model(samples)
        loss = -mll(output, weights)
        loss.backward()
        optimizer.step()
        if log_cb is not None:
            log_cb(loss.item(), is_maint=False)


def maintenance_phase(model, likelihood, optimizer, items, config,
                      maintenance_epochs, cued_item_idx=None,
                      maintenance_lr=None, log_cb=None, track_visuals=False):
    """
    Run the maintenance phase on top of an already-encoded model.

    Self-sustaining recurrence on the *live* inducing grid.  Each epoch
    evaluates at the current inducing points and regresses toward the
    posterior mean from the *previous* epoch (previous-step self-rehearsal),
    seeded at encoding end and rolled forward after every update.  There is
    no frozen encoding-end evaluation grid.  The KL term (``beta``) relaxes
    the posterior toward the prior over the delay (growing uncertainty).  A
    spatial attention envelope anchored to the cued item (recomputed on the
    current inducing locations) up-weights the cued region's expected
    log-likelihood after ``cue_start_epoch``, raising its posterior precision.

    Because retrieval SAMPLES the posterior (see ``retrieve_color_probabilistic``
    / ``retrieve_color(..., probabilistic=True)``), the retrocue benefit is
    protection through reduced recall variability of the cued item, not a boost
    of the posterior mean above encoding precision.

    Returns ``(maint_hist_surfaces, maint_hist_ind_pts)`` populated only when
    ``track_visuals`` is True.

    ``maintenance_lr`` overrides ``config['training']['maintenance_lr']`` when
    provided (used by learning-rate sweeps).
    """
    maint_lr = maintenance_lr if maintenance_lr is not None \
        else config['training']['maintenance_lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = maint_lr

    if cued_item_idx is not None:
        cued_loc = items[cued_item_idx][0]
        attn = SpatialProximityAttention(
            spatial_std=config['attention']['spatial_std'],
            attended_gain=config['attention']['attended_gain']
        ).to(device)
    else:
        cued_loc, attn = None, None

    cue_start_epoch = config['training']['cue_start_epoch']
    beta = config['training']['beta']

    # Seed previous-step rehearsal target at encoding-end state on live Z.
    with torch.no_grad():
        Z0 = model.variational_strategy.inducing_points
        prev_target = likelihood(model(Z0)).mean.detach()

    maint_hist_surfaces = []
    maint_hist_ind_pts = []

    model.train()
    for epoch in range(maintenance_epochs):
        optimizer.zero_grad()
        # Live inducing anchors (may migrate if learnable).
        Z = model.variational_strategy.inducing_points
        output = model(Z)

        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)

        exp_ll = likelihood.expected_log_prob(prev_target, output)

        M = Z.shape[0]
        if attn is not None and epoch >= cue_start_epoch:
            attn_weights = attn(Z[:, 0].detach(), cued_loc)
        else:
            attn_weights = torch.ones(M, device=device)

        weighted_ll = (exp_ll * attn_weights).sum() / M
        loss = -weighted_ll + kl_div * beta

        loss.backward()
        optimizer.step()

        # Previous-step roll: refresh target from the post-update state at
        # current inducing locations for the next epoch.
        with torch.no_grad():
            model.eval()
            Z = model.variational_strategy.inducing_points
            prev_target = likelihood(model(Z)).mean.detach()
            model.train()

        if log_cb is not None:
            log_cb(loss.item(), is_maint=True)

        if track_visuals:
            with torch.no_grad():
                model.eval()
                locs_v = torch.linspace(-180.0, 180.0, 100, device=device)
                cols_v = torch.linspace(-180.0, 180.0, 100, device=device)
                Lv, Cv = torch.meshgrid(locs_v, cols_v, indexing='ij')
                grid_v = torch.stack([Lv.flatten(), Cv.flatten()], dim=-1)
                surf_v = likelihood(model(grid_v)).mean.view(100, 100).cpu().numpy()
                maint_hist_surfaces.append(surf_v)
                maint_hist_ind_pts.append(
                    model.variational_strategy.inducing_points.detach().cpu().numpy()
                )
            model.train()

    return maint_hist_surfaces, maint_hist_ind_pts


def run_simulation_trial(
    items: List[Tuple[float, float]],
    config: dict,
    encoding_epochs: int = None,
    maintenance_epochs: int = None,
    cued_item_idx: Optional[int] = None,
    track_visuals: bool = False,
    track_retrieval: bool = False,
    track_lengthscales: bool = False,
    track_loss: bool = False
) -> Tuple[WorkingMemoryGP, gpytorch.likelihoods.GaussianLikelihood, Dict]:
    """
    Runs the Encoding and Maintenance phases for a single trial of N items.

    The ``track_*`` flags fill ``history`` with per-epoch diagnostics.
    ``track_retrieval`` draws posterior samples every epoch and therefore
    changes the random stream seen by later code; leave it off when
    reproducing the manuscript results.
    """
    samples, weights, model, likelihood, optimizer, mll = setup_trial(items, config)

    history = {
        'encoding_loss': [],
        'maintenance_loss': [],
        'loc_lengthscale': [],
        'color_lengthscale': [],
        'unsigned_errors': {i: [] for i in range(len(items))},
        'signed_errors': {i: [] for i in range(len(items))},
        'maintenance_surfaces': [],
        'maintenance_inducing_points': [],
    }

    def log_parameters(epoch_loss, is_maint=False):
        if track_loss:
            key = 'maintenance_loss' if is_maint else 'encoding_loss'
            history[key].append(epoch_loss)

        if track_lengthscales:
            history['loc_lengthscale'].append(model.covar_module.base_kernel.kernels[0].lengthscale.item())
            history['color_lengthscale'].append(model.covar_module.base_kernel.kernels[1].lengthscale.item())

        if track_retrieval:
            norm = config.get("normalization", None)
            item_locs = [float(loc) for loc, _ in items]
            for i, (l, c) in enumerate(items):
                signed_err = retrieve_color(
                    model, l, c,
                    normalization=norm, item_locs=item_locs, probabilistic=True,
                )
                history['unsigned_errors'][i].append(abs(signed_err))
                history['signed_errors'][i].append(signed_err)

    # ==================== ENCODING ====================
    if encoding_epochs is None:
        encoding_epochs = config['training']['encoding_epochs']
    encoding_phase(model, likelihood, optimizer, mll, samples, weights,
                   encoding_epochs, log_cb=log_parameters)

    # ==================== MAINTENANCE (OPTIONAL) ====================
    if maintenance_epochs is None:
        maintenance_epochs = config['training']['maintenance_epochs']
    if maintenance_epochs > 0:
        surfaces, ind_pts = maintenance_phase(
            model, likelihood, optimizer, items, config,
            maintenance_epochs, cued_item_idx=cued_item_idx,
            log_cb=log_parameters, track_visuals=track_visuals,
        )
        history['maintenance_surfaces'] = surfaces
        history['maintenance_inducing_points'] = ind_pts

    return model, likelihood, history
