"""Throwaway O(N^2) numpy port of the Pearce (2014) projection model (roadmap.md §3 item 1).

Not production code — never imported by `murmuration`. Purpose: fast (no Rust/PyO3 rebuild)
iteration to find a jointly-tuned parameter set that passes all 5 hard gates, before porting
the winning set back to `claude/python/murmuration/validate/results.py`'s
`DEFAULT_SIM_KWARGS`. Once trusted, doubles as the tight-tolerance f64 golden oracle for
cross-checking the Rust port's `occlude()` output per-bird (not done in this pass — see
`sci/param_table.md`'s "Numpy prototype" section for what's still open).

Faithfully mirrors, field-for-field:
  - claude/crates/murmur_core/src/occlusion.rs        (occlude())
  - claude/crates/plugins/murmur_pearce/src/lib.rs     (PearceProjection::desired, steric loop)
  - claude/crates/plugins/murmur_instant_response/     (InstantResponse::respond — clamp)
  - claude/crates/murmur_core/src/speed_model.rs       (BandSpeed::enforce)
  - claude/crates/plugins/murmur_radius_gather/         (distance-filtered candidate gathering)
  - claude/crates/murmur_core/src/rng.rs               (sample_unit_sphere's exact formula)
  - OpenSpace domain (delta = b - a, apply is a no-op) — the domain the acceptance harness uses.

Integration convention: `vel += acc * dt; pos += vel * dt` at `dt = 1.0` (design/01_core.md §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MIN_LEN = 1e-9


@dataclass
class Params:
    # CoreParams
    cruise_speed: float = 1.0
    max_force: float = 1.0
    speed_min_factor: float = 0.3
    vision_radius: float = 10.0
    dt: float = 1.0
    # OcclusionParams
    body_radius: float = 0.5
    anisotropy: float = 1.0
    anisotropic_enabled: bool = False
    blind_cone_half_angle: float = 0.524
    blind_cone_enabled: bool = True
    max_candidates: int = 20
    steric_enabled: bool = False
    steric: float = 0.6
    steric_radius_factor: float = 4.0
    # PearceParams
    phi_p: float = 0.03
    phi_a: float = 0.80
    sigma: int = 4

    @property
    def phi_n(self) -> float:
        return max(0.0, 1.0 - self.phi_p - self.phi_a)


def sample_unit_sphere(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Exact same formula as murmur_core::rng::sample_unit_sphere — area-uniform on S^2."""
    z = 1.0 - 2.0 * rng.random(n)
    t = 2.0 * np.pi * rng.random(n)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=-1)


def _normalized(v: np.ndarray) -> np.ndarray:
    """Row-wise normalize; zero-length rows -> zero (never NaN), matching Vec3::normalized()."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(n > MIN_LEN, n, 1.0)
    out = v / safe
    return np.where(n > MIN_LEN, out, 0.0)


def occlude_one(
    i: int,
    pos: np.ndarray,
    vel: np.ndarray,
    heading: np.ndarray,
    p: Params,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Mirrors occlusion.rs::occlude() for boid i. Returns (delta_hat, theta, align)."""
    offset = pos - pos[i]  # (N,3), b - a convention (OpenSpace::delta)
    dist = np.linalg.norm(offset, axis=-1)
    dist[i] = np.inf  # exclude self
    in_range = dist <= p.vision_radius
    in_range[i] = False
    idx = np.nonzero(in_range)[0]
    if idx.size == 0:
        return np.zeros(3), 0.0, np.zeros(3)

    d = np.maximum(dist[idx], MIN_LEN)
    direction = offset[idx] / d[:, None]  # unit bearing, observer -> neighbour
    nvel = vel[idx]

    if p.blind_cone_enabled:
        keep = direction @ (-heading) < np.cos(p.blind_cone_half_angle)
        idx, d, direction, nvel = idx[keep], d[keep], direction[keep], nvel[keep]
        if idx.size == 0:
            return np.zeros(3), 0.0, np.zeros(3)

    if p.anisotropic_enabled:
        vlen_sq = np.sum(nvel * nvel, axis=-1)
        has_vel = vlen_sq > MIN_LEN * MIN_LEN
        vhat = np.where(has_vel[:, None], _normalized(nvel), 0.0)
        a = p.body_radius * p.anisotropy
        cos_psi = np.abs(np.sum(direction * vhat, axis=-1))
        sin_psi = np.sqrt(np.maximum(1.0 - cos_psi * cos_psi, 0.0))
        b_eff = np.where(
            has_vel,
            np.sqrt((a * sin_psi) ** 2 + (p.body_radius * cos_psi) ** 2),
            p.body_radius,
        )
    else:
        b_eff = np.full(idx.size, p.body_radius)

    alpha = np.arcsin(np.minimum(b_eff / d, 1.0))
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)

    order = np.argsort(d, kind="stable")[: p.max_candidates]
    direction, nvel, sin_a, cos_a = direction[order], nvel[order], sin_a[order], cos_a[order]

    # Closest-first visibility cull: occluded if inside a nearer *visible* cap.
    visible_mask = np.zeros(direction.shape[0], dtype=bool)
    vis_dirs: list[np.ndarray] = []
    vis_cos: list[float] = []
    for k in range(direction.shape[0]):
        occluded = False
        for vd, vc in zip(vis_dirs, vis_cos):
            if direction[k] @ vd >= vc:
                occluded = True
                break
        if not occluded:
            visible_mask[k] = True
            vis_dirs.append(direction[k])
            vis_cos.append(cos_a[k])

    if not np.any(visible_mask):
        return np.zeros(3), 0.0, np.zeros(3)

    v_dir, v_vel, v_sin, v_cos = (
        direction[visible_mask],
        nvel[visible_mask],
        sin_a[visible_mask],
        cos_a[visible_mask],
    )
    delta = np.sum(v_dir * v_sin[:, None], axis=0)
    sin_sum = np.sum(v_sin)
    delta_hat = delta / sin_sum if sin_sum > MIN_LEN else np.zeros(3)
    transp = np.prod(1.0 - (1.0 - v_cos) * 0.5)
    theta = 1.0 - transp
    n_align = min(p.sigma, v_vel.shape[0])
    align = np.mean(v_vel[:n_align], axis=0) if n_align > 0 else np.zeros(3)
    return delta_hat, float(theta), align


def step(
    pos: np.ndarray,
    vel: np.ndarray,
    p: Params,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One full step: read phase (per-boid desired_v/extra_force/theta) + write phase
    (InstantResponse clamp, integrate, BandSpeed clamp, OpenSpace apply is a no-op).
    Returns (new_pos, new_vel, theta_per_boid)."""
    n = pos.shape[0]
    heading_all = _normalized(vel)
    noise_all = sample_unit_sphere(rng, n)

    desired_v = np.empty_like(vel)
    extra_force = np.zeros_like(vel)
    theta = np.empty(n)

    r_s = p.body_radius * p.steric_radius_factor

    for i in range(n):
        delta_hat, theta_i, align = occlude_one(i, pos, vel, heading_all[i], p)
        theta[i] = theta_i
        align_hat = _normalized(align) if np.dot(align, align) > MIN_LEN * MIN_LEN else heading_all[i]
        desired = delta_hat * p.phi_p + align_hat * p.phi_a + noise_all[i] * p.phi_n
        desired_sq = float(np.dot(desired, desired))
        desired_v[i] = (
            _normalized(desired[None, :])[0] * p.cruise_speed
            if desired_sq > MIN_LEN * MIN_LEN
            else vel[i]
        )

        if p.steric_enabled:
            offset = pos - pos[i]
            dist = np.linalg.norm(offset, axis=-1)
            dist[i] = np.inf
            close = dist < r_s
            if np.any(close):
                d = np.maximum(dist[close], MIN_LEN)
                direction = offset[close] / d[:, None]
                f = -np.sum(direction / (d * d)[:, None], axis=0)
                f = f * p.steric
                flen = np.linalg.norm(f)
                if flen > p.max_force:
                    f = f * (p.max_force / flen)
                extra_force[i] = f

    # InstantResponse: acc = clamp(desired_v - vel, max_force), plus extra_force (bypasses clamp).
    steer = desired_v - vel
    slen = np.linalg.norm(steer, axis=-1, keepdims=True)
    scale = np.minimum(1.0, p.max_force / np.maximum(slen, MIN_LEN))
    acc = steer * scale + extra_force

    new_vel = vel + acc * p.dt

    # BandSpeed: species-agnostic here (prototype has no predator concept), prey band only.
    vmax = p.cruise_speed
    vmin = p.cruise_speed * p.speed_min_factor
    speed = np.linalg.norm(new_vel, axis=-1, keepdims=True)
    over = speed[:, 0] > vmax
    under = speed[:, 0] < vmin
    stalled = speed[:, 0] <= MIN_LEN
    new_vel = np.where(
        over[:, None], new_vel * (vmax / np.maximum(speed, MIN_LEN)), new_vel
    )
    speed = np.linalg.norm(new_vel, axis=-1, keepdims=True)
    boosted = under & ~stalled
    new_vel = np.where(
        boosted[:, None], new_vel * (vmin / np.maximum(speed, MIN_LEN)), new_vel
    )
    if np.any(stalled):
        new_vel[stalled] = sample_unit_sphere(rng, int(stalled.sum())) * vmin

    # Integration convention is `vel += acc*dt; pos += vel*dt` — pos must use the *post*-speed-
    # clamp velocity, matching write_phase's actual order (integrate vel -> enforce speed ->
    # integrate pos), not the pre-clamp `new_vel` computed above for r_max readability.
    new_pos = pos + new_vel * p.dt

    return new_pos, new_vel, theta


def init_sphere_volume(n: int, radius: float, cruise_speed: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Matches murmur_initializers::SphereVolume closely enough for this prototype: uniform
    placement within a sphere of the given radius, random unit-sphere velocities at
    cruise_speed."""
    # Uniform-in-volume via rejection-free method: r ~ radius * u^(1/3).
    u = rng.random(n)
    r = radius * np.cbrt(u)
    dirs = sample_unit_sphere(rng, n)
    pos = dirs * r[:, None]
    vel = sample_unit_sphere(rng, n) * cruise_speed
    return pos, vel


def run(n: int, steps: int, p: Params, seed: int = 0, init_radius: float | None = None):
    rng = np.random.default_rng(seed)
    if init_radius is None:
        target_density = 0.02
        init_radius = (n / ((4.0 / 3.0) * np.pi * target_density)) ** (1.0 / 3.0)
    pos, vel = init_sphere_volume(n, init_radius, p.cruise_speed, rng)
    theta = np.zeros(n)
    for _ in range(steps):
        pos, vel, theta = step(pos, vel, p, rng)
    return pos, vel, theta
