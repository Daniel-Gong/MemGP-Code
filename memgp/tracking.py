"""Encode-and-maintain trial that records the representation at every epoch.

Used for the retrocue maintenance dynamics (Fig. 3k-l): inducing-point
positions and values, and the posterior mean and variance at each item, after
every encoding and maintenance step. The update rule is the same as
``memgp.simulation.maintenance_phase``.
"""
import copy

import numpy as np
import torch
import gpytorch

from memgp.gp_model import WorkingMemoryGP
from memgp.generator import sample_training_data
from memgp.attention import SpatialProximityAttention
from memgp.simulation import device


def circular_distance(a, b):
    """Minimum circular distance on [-180, 180)."""
    d = np.abs(a - b)
    return np.minimum(d, 360.0 - d)


def evaluate_items_at_positions(model, likelihood, items):
    """
    Evaluate the GP posterior mean at each item's (location, color) coordinate.
    Returns a vector of length N_items — the "activation" for each stored item.
    """
    model.eval()
    coords = torch.tensor(items, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = likelihood(model(coords))
        return pred.mean.cpu().numpy()


def run_trial_with_tracking(items, config, cued_item_idx=0):
    """
    Run a full trial (encode + maintain with retrocue) and return
    inducing point positions AND values at every epoch.
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
        learn_inducing_locations=config['model']['learn_inducing_locations'],
    ).to(device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config['likelihood']['noise_variance']
    likelihood.noise_covar.raw_noise.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['encoding_lr'])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(samples))

    # Track inducing points AND values
    ind_pts_history = []   # list of (M, 2) arrays — positions
    ind_vals_history = []  # list of (M,) arrays — variational means
    item_activations = []  # list of (N_items,) — posterior mean at item coords
    item_variances = []    # list of (N_items,) — posterior variance at item coords
    phase_labels = []      # 'encoding', 'pre_cue', 'post_cue'

    item_coords = torch.tensor(items, dtype=torch.float32, device=device)

    def snapshot(phase):
        pts = model.variational_strategy.inducing_points.detach().cpu().numpy().copy()
        vals = model.variational_strategy.variational_distribution.mean.detach().cpu().numpy().copy()
        model.eval()
        with torch.no_grad():
            pred = likelihood(model(item_coords))
            acts = pred.mean.cpu().numpy().copy()
            var = pred.variance.cpu().numpy().copy()
        ind_pts_history.append(pts)
        ind_vals_history.append(vals)
        item_activations.append(acts)
        item_variances.append(var)
        phase_labels.append(phase)
        model.train()

    # ─── ENCODING ───
    model.train()
    likelihood.train()
    encoding_epochs = config['training']['encoding_epochs']
    for epoch in range(encoding_epochs):
        optimizer.zero_grad()
        output = model(samples)
        loss = -mll(output, weights)
        loss.backward()
        optimizer.step()
        snapshot('encoding')

    # ─── MAINTENANCE ───
    maintenance_epochs = config['training']['maintenance_epochs']
    maint_lr = config['training']['maintenance_lr']
    for param_group in optimizer.param_groups:
        param_group['lr'] = maint_lr

    # Previous-step self-sustain (matches simulation.maintenance_phase).
    with torch.no_grad():
        Z0 = model.variational_strategy.inducing_points
        prev_target = likelihood(model(Z0)).mean.detach()

    cued_loc = items[cued_item_idx][0]
    attn = SpatialProximityAttention(
        spatial_std=config['attention']['spatial_std'],
        attended_gain=config['attention']['attended_gain']
    ).to(device)

    cue_start_epoch = config['training']['cue_start_epoch']
    beta = config['training']['beta']

    for epoch in range(maintenance_epochs):
        optimizer.zero_grad()
        Z = model.variational_strategy.inducing_points
        output = model(Z)

        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)

        exp_ll = likelihood.expected_log_prob(prev_target, output)

        M = Z.shape[0]
        if epoch >= cue_start_epoch:
            attn_weights = attn(Z[:, 0].detach(), cued_loc)
            phase = 'post_cue'
        else:
            attn_weights = torch.ones(M, device=device)
            phase = 'pre_cue'

        weighted_ll = (exp_ll * attn_weights).sum() / M
        loss = -weighted_ll + kl_div * beta
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            Z = model.variational_strategy.inducing_points
            prev_target = likelihood(model(Z)).mean.detach()
            model.train()
        snapshot(phase)

    return (ind_pts_history, ind_vals_history, item_activations, item_variances,
            phase_labels, items, cued_item_idx, cue_start_epoch)


def _build(config, n_data):
    model = WorkingMemoryGP(
        inducing_grid_size=config['model']['inducing_grid_size'],
        loc_lengthscale=config['model']['loc_lengthscale'],
        color_lengthscale=config['model']['color_lengthscale'],
        learn_inducing_locations=config['model']['learn_inducing_locations'],
    ).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
    ).to(device)
    likelihood.noise = config['likelihood']['noise_variance']
    likelihood.noise_covar.raw_noise.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['encoding_lr'])
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_data)
    return model, likelihood, optimizer, mll


def _maintain_probed(config, model_state, lik_state, opt_state, items, probed, cued_loc):
    """One delay from a saved encoding. ``cued_loc`` None is neutral maintenance.

    Returns the probed item's posterior mean and variance after every epoch.
    """
    model, likelihood, optimizer, _ = _build(config, 1)
    model.load_state_dict(model_state)
    likelihood.load_state_dict(lik_state)
    optimizer.load_state_dict(opt_state)
    for group in optimizer.param_groups:
        group['lr'] = config['training']['maintenance_lr']

    attn = None
    if cued_loc is not None:
        attn = SpatialProximityAttention(
            spatial_std=config['attention']['spatial_std'],
            attended_gain=config['attention']['attended_gain']
        ).to(device)
    cue_start = config['training']['cue_start_epoch']
    beta = config['training']['beta']
    T = config['training']['maintenance_epochs']
    probed_xy = torch.tensor([items[probed]], dtype=torch.float32, device=device)

    with torch.no_grad():
        Z0 = model.variational_strategy.inducing_points
        prev_target = likelihood(model(Z0)).mean.detach()

    acts, variances = [], []
    model.train()
    likelihood.train()
    for epoch in range(T):
        optimizer.zero_grad()
        Z = model.variational_strategy.inducing_points
        output = model(Z)
        var_dist = model.variational_strategy.variational_distribution
        prior_dist = model.variational_strategy.prior_distribution
        kl_div = torch.distributions.kl.kl_divergence(var_dist, prior_dist)
        exp_ll = likelihood.expected_log_prob(prev_target, output)
        M = Z.shape[0]
        if attn is not None and epoch >= cue_start:
            weights = attn(Z[:, 0].detach(), cued_loc)
        else:
            weights = torch.ones(M, device=device)
        loss = -(exp_ll * weights).sum() / M + kl_div * beta
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            Z = model.variational_strategy.inducing_points
            prev_target = likelihood(model(Z)).mean.detach()
            pred = likelihood(model(probed_xy))
            acts.append(float(pred.mean))
            variances.append(float(pred.variance))
            model.train()
    return np.asarray(acts), np.asarray(variances)


def run_paired_tracking(items, config, probed):
    """Encode once, then maintain the probed item twice: cued and neutral."""
    samples, weights, _ = sample_training_data(
        items,
        n_samples_per_item=config['data']['n_samples_per_item'],
        loc_std=config['data']['loc_std'],
        color_std=config['data']['color_std'],
        loc_encoding_noise_std=config['data']['loc_encoding_noise_std'],
        color_encoding_noise_std=config['data']['color_encoding_noise_std']
    )
    model, likelihood, optimizer, mll = _build(config, len(samples))
    model.train()
    likelihood.train()
    for _ in range(config['training']['encoding_epochs']):
        optimizer.zero_grad()
        loss = -mll(model(samples), weights)
        loss.backward()
        optimizer.step()

    saved = (copy.deepcopy(model.state_dict()),
             copy.deepcopy(likelihood.state_dict()),
             copy.deepcopy(optimizer.state_dict()))
    cued = _maintain_probed(config, *saved, items, probed, items[probed][0])
    neutral = _maintain_probed(config, *saved, items, probed, None)
    return cued, neutral

