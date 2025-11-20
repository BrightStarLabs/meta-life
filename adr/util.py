from typing import Tuple
import math
import numpy as np

from config import Cfg
# --------------------------- Utility functions ---------------------------
def wrap_xy(xy: np.ndarray, W: float, H: float) -> None:
    xy[:, 0] %= W
    xy[:, 1] %= H


def torus_delta(A: np.ndarray, B: np.ndarray, W: float, H: float) -> np.ndarray:
    d = B[None, :, :] - A[:, None, :]
    d[..., 0] = (d[..., 0] + W/2.0) % W - W/2.0
    d[..., 1] = (d[..., 1] + H/2.0) % H - H/2.0
    return d


def pairwise_dist_torus(A: np.ndarray, B: np.ndarray, W: float, H: float) -> np.ndarray:
    d = torus_delta(A, B, W, H)
    return np.sqrt(np.sum(d * d, axis=-1))


def top_percentile_mean(values: np.ndarray, pct: float = 0.05) -> float:
    n = values.size
    if n == 0 or not np.isfinite(values).any():
        return float('nan')
    k = max(1, int(math.ceil(pct * n)))
    part = np.partition(values, n - k)
    top_k = part[n - k:]
    return float(np.mean(top_k))


# --------------------------- CEM Brain (a1,a2,m) ---------------------------
def _lap(x: np.ndarray) -> np.ndarray:
    # Laplacian on ring: l - 2c + r
    l = np.roll(x, 1, axis=1)
    r = np.roll(x, -1, axis=1)
    return l - 2.0 * x + r


def _grad(x: np.ndarray) -> np.ndarray:
    # Simple central difference on ring: r - l
    l = np.roll(x, 1, axis=1)
    r = np.roll(x, -1, axis=1)
    return r - l


def cem_derivatives(a1: np.ndarray, a2: np.ndarray, m: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized derivatives for all agents:
      da1/dt = d*Lap(a1) + v*Grad(a1) + ω*a2 + χ*m*a1 - β*(a1^2 + a2^2)*a1
      da2/dt = d*Lap(a2) + v*Grad(a2) - ω*a1 + χ*m*a2 - β*(a1^2 + a2^2)*a2
      dm/dt  = -λ_m*m + ρ*(a1*a2)
    Shapes:
      a1,a2,m: (A, N)
      k:       (A, 7) with (d,v,ω,χ,β,ρ,λ_m)
    """
    d   = k[:, 0][:, None]
    v   = k[:, 1][:, None]
    omg = k[:, 2][:, None]
    chi = k[:, 3][:, None]
    bet = k[:, 4][:, None]
    rho = k[:, 5][:, None]
    lam = k[:, 6][:, None]

    r2 = a1*a1 + a2*a2

    lap1 = _lap(a1)
    lap2 = _lap(a2)
    grd1 = _grad(a1)
    grd2 = _grad(a2)

    fa1 = d*lap1 + v*grd1 + omg*a2 + chi*(m*a1) - bet*(r2*a1) - 0.01*a1**3
    fa2 = d*lap2 + v*grd2 - omg*a1 + chi*(m*a2) - bet*(r2*a2) - 0.01*a1**3 + 0.1*lam*m
    fm  = 0.003*rho*(a1*a2)
    return fa1, fa2, fm


def cem_integrate_rk2(a1: np.ndarray,
                      a2: np.ndarray,
                      m: np.ndarray,
                      k: np.ndarray,
                      dt: float,
                      T: int,
                      abs_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A, N = a1.shape
    bad = np.zeros(A, dtype=bool)
    for _ in range(T):
        fa1_1, fa2_1, fm_1 = cem_derivatives(a1, a2, m, k)
        a1_p = a1 + dt * fa1_1
        a2_p = a2 + dt * fa2_1
        m_p  = m  + dt * fm_1

        fa1_2, fa2_2, fm_2 = cem_derivatives(a1_p, a2_p, m_p, k)
        a1_new = a1 + 0.5 * dt * (fa1_1 + fa1_2)
        a2_new = a2 + 0.5 * dt * (fa2_1 + fa2_2)
        m_new  = m  + 0.5 * dt * (fm_1  + fm_2)

        nonfinite = ~(np.isfinite(a1_new).all(axis=1) & np.isfinite(a2_new).all(axis=1) & np.isfinite(m_new).all(axis=1))
        overflow = (np.max(np.abs(a1_new), axis=1) > abs_max) | (np.max(np.abs(a2_new), axis=1) > abs_max) | (np.max(np.abs(m_new), axis=1) > abs_max)
        step_bad = nonfinite | overflow
        bad |= step_bad

        if bad.any():
            keep = ~bad
            if keep.any():
                a1[keep] = a1_new[keep]
                a2[keep] = a2_new[keep]
                m[keep]  = m_new[keep]
        else:
            a1 = a1_new
            a2 = a2_new
            m  = m_new
    return a1, a2, m, bad


# --------------------------- GA Helpers (crossover + fat-tail mutation) ---------------------------
def crossover_mutate(rng: np.random.Generator,
                     A_param: np.ndarray,
                     B_param: np.ndarray,
                     crossover_prob: float,
                     mut_prob: float,
                     sigma_base: float,
                     t_df: float) -> np.ndarray:
    """
    Elementwise crossover between A_param and B_param with Bernoulli mask,
    then elementwise Student-t mutation scaled by |param| magnitude.
    Shapes of A_param and B_param must match.
    """
    mask = (rng.random(A_param.shape) < crossover_prob)
    child = np.where(mask, A_param, B_param).astype(np.float32)

    if mut_prob > 0.0 and sigma_base > 0.0:
        m_mask = (rng.random(child.shape) < mut_prob)
        if m_mask.any():
            t_noise = rng.standard_t(df=t_df, size=int(m_mask.sum())).astype(np.float32)
            eps = 1e-3
            mag = np.abs(child[m_mask]) + eps
            child[m_mask] += (sigma_base * mag) * t_noise
    return child


# --------------------------- Agent helpers ---------------------------
def random_spawn(N: int, w: float, h: float, baseline_birth: float, rng: np.random.Generator):
    xy = np.empty((N, 2), np.float32)
    xy[:, 0] = rng.uniform(0, w, N)
    xy[:, 1] = rng.uniform(0, h, N)
    heading = rng.uniform(0.0, 2.0 * math.pi, N).astype(np.float32)
    speed = np.zeros(N, np.float32)
    life = np.full(N, baseline_birth, np.float32)
    return xy, heading, speed, life


def spawn_near(parent_xy: np.ndarray, w: float, h: float, rng: np.random.Generator, sigma: float = 80.0):
    xy = parent_xy + rng.normal(0.0, sigma, size=(1, 2)).astype(np.float32)
    xy[:, 0] %= w
    xy[:, 1] %= h
    heading = rng.uniform(0.0, 2.0 * math.pi, 1).astype(np.float32)
    speed = np.zeros(1, np.float32)
    return xy, heading, speed


# --------------------------- Runner ---------------------------
def build_cfg_from_hparams(hparams: dict) -> Cfg:
    return Cfg(**hparams)

