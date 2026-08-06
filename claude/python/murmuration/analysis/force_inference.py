"""Maximum-likelihood force-channel-weight and reaction-delay (tau) estimation from a
trajectory, plus model comparison (Cai, Zhang, Chen & Wang 2021, "A General Statistical
Mechanics Framework for the Collective Motion of Animals", `sci/2112.15560v1.pdf`) —
design/03_observables_bindings.md §3. Pairs with `murmur_maxent_social`
(design/02_plugins.md §5), which adapts the same paper's model into a real `FlockingMode` —
this module is the analysis-layer half: it operates on *any* trajectory (Rust-core snapshots,
pymurmur CSVs, or real tracking data) regardless of which `FlockingMode` produced it, exactly
as the design doc specifies, so it never reads `murmur_maxent_social`'s own internal state.

**The paper's model** (its eq. 3-7): a boid's velocity at time `t`, given the flock's state at
time `t-tau`, is multivariate-normal with mean `F_i^{t-tau} / lambda_kin` and covariance
`I / lambda_kin`, where `F_i` (eq. 5) sums weighted force channels (alignment, repulsion,
attraction, boundary, desire) evaluated at `t-tau`. `lambda_kin` plays the role of "average
inertia" (eq. 7's own remark) — equivalently, `1/lambda_kin` is the model's noise variance.

**Channels implemented here**: alignment, repulsion, attraction — the three channels
computable from raw position/velocity data alone. Boundary and desire are deliberately *not*
auto-inferred: boundary needs a domain geometry the trajectory alone doesn't carry, and desire
is an external "will" input by the paper's own definition, not derivable from `s_{t-tau}` at
all — a caller with that side information can add its own extra design-matrix columns, but
this module doesn't guess at either.

**Uniform (mean) weighting for every channel**, disclosed: the paper leaves the `g` "strength"
functions as configurable per model (its eq. 5's own text: "we do not require the
formats/formulas/values of the ... g ... terms to remain unchanged"), naming only alignment's
`g_ali = 1/N_ali` explicitly (its eq. 9, the paper's own Vicsek-reduction special case) for its
worked examples. Extending that same "mean, not sum" convention to repulsion and attraction
too keeps a single, consistent, density-independent definition across all three — and matches
`murmur_maxent_social`'s own attraction channel exactly (that plugin's own repulsion channel
uses an inverse-distance weight instead, for direct physical-plausibility reasons that don't
apply to a maximum-likelihood *fit*, where a simpler, count-invariant channel definition is a
cleaner regressor).

**Fitting, given a fixed tau and channel set, is ordinary least squares, not a search.**
`F_i^{t-tau}/lambda_kin` is *linear* in the `lambda_c/lambda_kin` weight ratios once the
channel terms themselves are computed from data, so the paper's own maximum-likelihood
estimator (its own worked example, §I.C) reduces to closed-form OLS: fit
`v_t = sum_c w_c * channel_c(s_{t-tau}) + noise` via `numpy.linalg.lstsq`, recover
`w_c = lambda_c/lambda_kin` directly, and recover `lambda_kin` itself from the residual
variance (`lambda_kin = 1/mean(residual^2)`, the Gaussian MLE). **Only the ratios `w_c` are
identifiable this way, not `lambda_c` and `lambda_kin` individually** — the same
non-identifiability the paper's own Discussion section gestures at ("it is possible that
lambda_kin may change with respect to time, unlike the physical mass"); this module reports
`w_c` (labelled accordingly) rather than pretending to separate them without further
assumptions.

**tau selection**: sweep candidate delays, refit at each, and take the one with the highest
pooled log-likelihood across the whole trajectory — the paper's own methodology (§I.C: "we
select the tau ... that maximize[s] the log likelihood ... among the candidates").

**Model comparison uses AIC, not raw log-likelihood**, disclosed: comparing two *nested*
channel sets by raw log-likelihood always weakly favours the larger set (more free parameters
can only improve an OLS fit, never worsen it) — not a fair test of which channel set actually
matters. AIC (`2k - 2*log_likelihood`, `k` = number of fitted weights including `lambda_kin`)
penalises that complexity, the standard fix, and is what `compare_models` actually ranks by;
raw log-likelihood is still reported alongside it for readers who want the paper's own literal
quantity.
"""

import numpy as np


def _pairwise_offsets(pos: np.ndarray) -> np.ndarray:
    """pos: (N, 3). Returns (N, N, 3) with `out[i, j] = pos[j] - pos[i]` (offset from i to j)."""
    pos = np.asarray(pos, dtype=np.float64)
    return pos[np.newaxis, :, :] - pos[:, np.newaxis, :]


def alignment_term(vel: np.ndarray, pos: np.ndarray = None, radius: float = None) -> np.ndarray:
    """vel: (N, 3). Mean neighbour velocity per boid (eq. 9's own `g_ali = 1/N_ali`
    convention) -- every *other* boid if `pos`/`radius` are omitted (global mean, the Vicsek
    special case), or only those within `radius` (metric) if both are given. A boid with no
    qualifying neighbours gets `(0, 0, 0)` (no alignment pull, not undefined)."""
    vel = np.asarray(vel, dtype=np.float64)
    n = vel.shape[0]
    if pos is None or radius is None:
        mask = ~np.eye(n, dtype=bool)
    else:
        dist = np.linalg.norm(_pairwise_offsets(pos), axis=-1)
        mask = (dist <= radius) & ~np.eye(n, dtype=bool)
    counts = mask.sum(axis=1, keepdims=True)
    summed = mask.astype(np.float64) @ vel
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, summed / np.maximum(counts, 1), 0.0)
    return out


def _radial_term(pos: np.ndarray, radius: float, sign: float) -> np.ndarray:
    """Shared implementation for repulsion (`sign=-1`, away from neighbours) and attraction
    (`sign=+1`, toward neighbours): mean unit direction to every neighbour within `radius`."""
    pos = np.asarray(pos, dtype=np.float64)
    offsets = _pairwise_offsets(pos)  # offsets[i, j] = pos[j] - pos[i]
    dist = np.linalg.norm(offsets, axis=-1)
    n = pos.shape[0]
    within = (dist <= radius) & (dist > 1e-9) & ~np.eye(n, dtype=bool)
    unit = np.divide(
        offsets, dist[..., np.newaxis], out=np.zeros_like(offsets), where=dist[..., np.newaxis] > 1e-9
    )
    counts = within.sum(axis=1, keepdims=True)
    summed = np.einsum("ij,ijk->ik", within.astype(np.float64), unit)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, summed / np.maximum(counts, 1), 0.0)
    return sign * out


def repulsion_term(pos: np.ndarray, radius: float) -> np.ndarray:
    """(N, 3) mean unit direction *away* from every neighbour within `radius`."""
    return _radial_term(pos, radius, sign=-1.0)


def attraction_term(pos: np.ndarray, radius: float) -> np.ndarray:
    """(N, 3) mean unit direction *toward* every neighbour within `radius`."""
    return _radial_term(pos, radius, sign=1.0)


_CHANNEL_NAMES = ("alignment", "repulsion", "attraction")


def channel_terms(pos: np.ndarray, vel: np.ndarray, channels, radii: dict) -> dict:
    """Builds the requested channel arrays for one time slice. `radii`: dict mapping channel
    name -> radius (ignored for `"alignment"`, which is always the global mean -- matching
    `murmur_maxent_social`'s own "alignment reads every composed neighbour, not a re-filtered
    subset" choice, ported to this module's single-radius-per-channel convention)."""
    out = {}
    for c in channels:
        if c == "alignment":
            out[c] = alignment_term(vel)
        elif c == "repulsion":
            out[c] = repulsion_term(pos, radii["repulsion"])
        elif c == "attraction":
            out[c] = attraction_term(pos, radii["attraction"])
        else:
            raise ValueError(f"unknown channel {c!r}, expected one of {_CHANNEL_NAMES}")
    return out


def _design_matrix_and_target(pos_traj, vel_traj, dt, tau, channels, radii):
    """Stacks (channel design matrix X, target y) pairs across every valid (t-tau, t) frame
    pair in the trajectory, flattened over boids and the 3 spatial dimensions -- the "dN
    observations" eq. 3 itself treats the population as. `tau` is in whole trajectory frames
    (an integer step-lag), matching this project's own one-frame-per-`run_batch(1,...)`
    trajectory-recording convention elsewhere (e.g. `information_transfer.py`)."""
    pos_traj = np.asarray(pos_traj, dtype=np.float64)
    vel_traj = np.asarray(vel_traj, dtype=np.float64)
    t_steps, n, _ = pos_traj.shape
    if tau < 1 or tau >= t_steps:
        return None, None
    rows_x = []
    rows_y = []
    for t in range(tau, t_steps):
        terms = channel_terms(pos_traj[t - tau], vel_traj[t - tau], channels, radii)
        x_t = np.stack([terms[c] for c in channels], axis=-1)  # (N, 3, C)
        rows_x.append(x_t.reshape(n * 3, len(channels)))
        rows_y.append(vel_traj[t].reshape(n * 3))
    x = np.concatenate(rows_x, axis=0)
    y = np.concatenate(rows_y, axis=0)
    return x, y


def fit_weights(pos_traj, vel_traj, dt, tau, channels, radii=None) -> dict:
    """Ordinary-least-squares fit of `v_t = sum_c w_c * channel_c(s_{t-tau}) + noise`, pooled
    across every valid frame pair at delay `tau`. Returns `weights` (dict: channel -> w_c,
    `w_c = lambda_c/lambda_kin`, only the *ratio* is identifiable -- see module doc),
    `lambda_kin` (`1/mean(residual^2)`, the Gaussian MLE), `log_likelihood` (eq. 3's own
    profile log-likelihood at that MLE), `n_obs`, and `n_params` (`len(channels) + 1`, for
    AIC). `None` fields if `tau` leaves no valid frame pairs."""
    radii = radii or {}
    x, y = _design_matrix_and_target(pos_traj, vel_traj, dt, tau, channels, radii)
    if x is None or x.shape[0] == 0:
        return {
            "tau": tau,
            "weights": None,
            "lambda_kin": None,
            "log_likelihood": None,
            "n_obs": 0,
            "n_params": len(channels) + 1,
        }
    w, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    # macOS's Accelerate BLAS backend (numpy's default there) emits spurious "divide by
    # zero"/"overflow"/"invalid value" RuntimeWarnings on some matmul shapes even when every
    # input and output is genuinely finite and correct (same known quirk already documented
    # and worked around in structure_factor.py — confirmed via numpy.show_config() there,
    # not re-derived here) — not a real numerical problem.
    with np.errstate(all="ignore"):
        residuals = y - x @ w
    mean_sq_residual = float(np.mean(residuals**2))
    mean_sq_residual = max(mean_sq_residual, 1e-300)  # guard a literal 1/0 for a noiseless fit
    lambda_kin = 1.0 / mean_sq_residual
    d_n = y.shape[0]
    log_likelihood = -0.5 * d_n * (np.log(2.0 * np.pi * mean_sq_residual) + 1.0)
    return {
        "tau": tau,
        "weights": dict(zip(channels, w.tolist())),
        "lambda_kin": lambda_kin,
        "log_likelihood": float(log_likelihood),
        "n_obs": d_n,
        "n_params": len(channels) + 1,
    }


def select_delay_time(pos_traj, vel_traj, dt, tau_candidates, channels, radii=None) -> dict:
    """Fits `fit_weights` at every candidate `tau`, returns the one with the highest pooled
    log-likelihood (the paper's own delay-selection methodology, §I.C) alongside the full
    per-tau table for inspection."""
    fits = [fit_weights(pos_traj, vel_traj, dt, tau, channels, radii) for tau in tau_candidates]
    valid = [f for f in fits if f["log_likelihood"] is not None]
    if not valid:
        return {"best": None, "all": fits}
    best = max(valid, key=lambda f: f["log_likelihood"])
    return {"best": best, "all": fits}


def compare_models(pos_traj, vel_traj, dt, tau_candidates, model_channel_sets, radii=None) -> dict:
    """For each named model (a channel-subset spec, e.g. `{"vicsek": ["alignment"], "full":
    ["alignment", "repulsion", "attraction"]}`), runs `select_delay_time` and reports its best
    fit, its AIC (`2*n_params - 2*log_likelihood`), and `n_params` -- ranked by AIC (lower is
    better), not raw log-likelihood (see module doc: nested channel sets otherwise always
    favour the larger one). `best_model` is the AIC winner's name."""
    results = {}
    for name, channels in model_channel_sets.items():
        selection = select_delay_time(pos_traj, vel_traj, dt, tau_candidates, channels, radii)
        best = selection["best"]
        aic = None
        if best is not None:
            aic = 2.0 * best["n_params"] - 2.0 * best["log_likelihood"]
        results[name] = {"fit": best, "aic": aic, "all_tau": selection["all"]}
    ranked = [(name, r["aic"]) for name, r in results.items() if r["aic"] is not None]
    best_model = min(ranked, key=lambda p: p[1])[0] if ranked else None
    return {"models": results, "best_model": best_model}


def force_inference_report(
    pos_traj,
    vel_traj,
    dt,
    tau_candidates=(1, 2, 3, 4),
    repulsion_radius: float = 2.0,
    attraction_radius: float = 10.0,
) -> dict:
    """Top-level orchestrator: compares a Vicsek-only model (alignment) against the full
    data-derivable channel set (alignment + repulsion + attraction) and returns the AIC-ranked
    result -- the paper's own headline-style comparison, generalised beyond its one specific
    non-Vicsek alternative."""
    radii = {"repulsion": repulsion_radius, "attraction": attraction_radius}
    model_channel_sets = {
        "vicsek": ["alignment"],
        "full": ["alignment", "repulsion", "attraction"],
    }
    return compare_models(pos_traj, vel_traj, dt, tau_candidates, model_channel_sets, radii)
