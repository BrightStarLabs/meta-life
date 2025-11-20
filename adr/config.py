from dataclasses import dataclass
from typing import Optional
import numpy as np
import math


# --------------------------- Config dataclass ---------------------------
@dataclass
class Cfg:
    # World / timing
    width: int
    height: int
    dt: float
    speed_px_s: float

    # Sensors
    n_rays: int
    fov_deg: float
    ray_length: float

    # Agents
    n_target_agents: int
    baseline_birth: float
    agent_radius: float
    max_life: float

    # Food
    food_radius: float
    food_reward: float
    food_spawn_prob: float
    food_max: int

    # Poison
    poison_radius: float
    poison_reward: float
    poison_spawn_prob: float
    poison_max: int

    # Costs
    consumption_base: float
    consumption_speed_coeff: float

    # Looping penalty (EMA)
    ema_coeff: float
    looping_penalty: float

    # CEM Brain
    cem_N: int
    cem_T: int
    cem_dt: float
    cem_abs_max: float
    cem_lambda_write: float
    cem_P: int

    # Sparse encode/decode
    mod_every: int
    enc_clip: float

    # Genome init
    e_init_sigma: float
    r_init_sigma: float
    k_init_sigma: float
    m0_init_sigma: float

    # Readout gains
    cem_steer_gain: float
    cem_speed_gain: float

    # Refill trigger & composition
    refill_frac: float
    elite_fraction: float
    immigrants_per_refill: int

    # GA ops
    crossover_prob: float
    mutation_prob: float
    mutation_sigma_base: float
    mutation_t_df: float

    # Bank
    bank_max: int
    bank_decay_prob: float
    bank_add_every: int

    # Derived
    input_dim: int = 0  # I
    output_dim: int = 2 # O
    sel_idx: Optional[np.ndarray] = None
    N_sel: int = 0
    refill_threshold: int = 0

    def __post_init__(self):
        # Input: food rays + poison rays + reward scalar
        self.input_dim = 2 * self.n_rays + 1
        assert self.output_dim == 2

        # sparse selection
        assert self.mod_every >= 1
        self.sel_idx = np.arange(0, self.cem_N, self.mod_every, dtype=np.int32)
        self.N_sel = int(self.sel_idx.size)
        assert self.N_sel >= 1, "mod_every too large; no selected cells."

        # checks
        assert 0.0 <= self.cem_lambda_write <= 1.0
        assert self.cem_P >= 7
        assert 0.0 <= self.ema_coeff < 1.0
        assert 0.0 < self.refill_frac <= 1.0
        assert 0.0 <= self.elite_fraction <= 1.0
        assert self.immigrants_per_refill >= 0
        assert self.bank_add_every >= 1
        assert 0.0 <= self.bank_decay_prob <= 1.0

        # refill threshold
        self.refill_threshold = int(math.floor(self.refill_frac * self.n_target_agents))

