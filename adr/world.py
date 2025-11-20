import math
import numpy as np

try:
    import pygame
    _PG = True
except ModuleNotFoundError:
    pygame = None
    _PG = False

from config import Cfg
from util import (
    torus_delta,
    wrap_xy,
    pairwise_dist_torus,
    cem_integrate_rk2,
    top_percentile_mean,
    random_spawn,
    spawn_near,
    crossover_mutate
)

# --------------------------- World ---------------------------
class World:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng()
        self.time = 0.0
        self.step_count = 0
        self.ga_log = []
        self._compute_ray_angles()

        N = cfg.cem_N
        I = cfg.input_dim
        O = cfg.output_dim
        N_sel = cfg.N_sel

        # Agents (initial surge = 10x target for exploration)
        A = cfg.n_target_agents * 10
        self.a_xy, self.a_dir, self.a_speed, self.a_life = random_spawn(A, cfg.width, cfg.height, cfg.baseline_birth, self.rng)

        # Runtime states (a1, a2 start at 0; m starts at m0)
        self.cem_a1 = np.zeros((A, N), dtype=np.float32)
        self.cem_a2 = np.zeros((A, N), dtype=np.float32)
        # Genome: coefficients k, memory prior m0, encoder E (N_sel×I), readout R (O×N_sel)
        self.cem_k  = self.rng.normal(0.0, cfg.k_init_sigma,  size=(A, cfg.cem_P)).astype(np.float32)
        self.gen_m0 = self.rng.normal(0.0, cfg.m0_init_sigma, size=(A, N)).astype(np.float32)
        self.cem_m  = self.gen_m0.copy().astype(np.float32)  # runtime memory starts from m0

        self.gen_E  = self.rng.normal(0.0, cfg.e_init_sigma, size=(A, N_sel, I)).astype(np.float32)
        self.gen_R  = self.rng.normal(0.0, cfg.r_init_sigma, size=(A, O, N_sel)).astype(np.float32)

        self.a_fit = np.zeros(A, np.float32)

        # Reward (one-step delayed signal)
        self._reward_signal = np.zeros(A, dtype=np.float32)

        # EMA looping penalty
        self.ema_steer = np.zeros(A, dtype=np.float32)

        # Elite bank
        self.elite_bank = []   # list of dicts {"k","m0","E","R","fit","life","step"}
        self._bank_cycle_count = 0

        # World resources
        self.f_xy = self._spawn_food(cfg.food_max // 2)
        self.p_xy = self._spawn_poison(cfg.poison_max // 2)

        # Sensor cache
        self._last_sensor = np.ones((A, 2 * cfg.n_rays), np.float32)

        print(f"Initialized: A={A}, N={N}, N_sel={N_sel}, I={I}, O={O}")
        print(f"EMA alpha={cfg.ema_coeff} | loop_penalty={cfg.looping_penalty} | enc_clip={cfg.enc_clip}")
        print(f"Sparse indices: {cfg.sel_idx.tolist()}")

    def _compute_ray_angles(self):
        k = self.cfg.n_rays
        if k == 1:
            self._ray_rel_angles = np.array([0.0], np.float32)
        else:
            span = math.radians(self.cfg.fov_deg)
            self._ray_rel_angles = np.linspace(-span / 2.0, span / 2.0, k, dtype=np.float32)

    def _spawn_food(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), np.float32)
        xy = np.empty((n, 2), np.float32)
        xy[:, 0] = self.rng.uniform(0, self.cfg.width, n)
        xy[:, 1] = self.rng.uniform(0, self.cfg.height, n)
        return xy

    def _spawn_poison(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.empty((0, 2), np.float32)
        xy = np.empty((n, 2), np.float32)
        xy[:, 0] = self.rng.uniform(0, self.cfg.width, n)
        xy[:, 1] = self.rng.uniform(0, self.cfg.height, n)
        return xy

    def _sense(self) -> np.ndarray:
        cfg = self.cfg
        A = len(self.a_xy)
        k = cfg.n_rays
        X = np.ones((A, 2 * k), np.float32)  # [food rays, poison rays]
        if A == 0:
            return X

        W, H, rng_len = cfg.width, cfg.height, cfg.ray_length
        targets_sets = [(self.f_xy, 0), (self.p_xy, k)]

        for targets, out_off in targets_sets:
            T = len(targets)
            if T == 0:
                continue

            delta = torus_delta(self.a_xy, targets, W, H)
            dist = np.sqrt(np.sum(delta * delta, axis=-1))
            angles_abs = np.arctan2(delta[..., 1], delta[..., 0])

            if k == 1:
                half_beam = math.radians(cfg.fov_deg) / 2.0
                step = None
            else:
                step = math.radians(cfg.fov_deg) / (k - 1)

            for j, rel in enumerate(self._ray_rel_angles):
                dir_j = (self.a_dir + rel) % (2.0 * math.pi)
                dtheta = np.abs(np.angle(np.exp(1j * (angles_abs - dir_j[:, None]))))
                in_beam = (dtheta < half_beam) if k == 1 else (dtheta < (step / 2.0))
                mask = in_beam & (dist <= rng_len)
                if not mask.any():
                    continue

                nearest = np.where(mask, dist, np.inf)
                dmin = nearest.min(axis=1)
                norm = dmin / rng_len
                inf_mask = np.isinf(norm)
                if inf_mask.any():
                    norm[inf_mask] = 1.0
                X[:, out_off + j] = norm.astype(np.float32)

        return X

    def _build_input_vector(self, X: np.ndarray) -> np.ndarray:
        """ x = [f', p', reward], shape (A, I). """
        cfg = self.cfg
        k = cfg.n_rays
        X_food = X[:, :k]
        X_poison = X[:, k:2*k]

        fprime = 1.0 - 2.0 * X_food
        pprime = 1.0 - 2.0 * X_poison
        reward = self._reward_signal[:, None].astype(np.float32)

        x = np.concatenate([fprime, pprime, reward], axis=1)
        return x

    def step(self):
        cfg = self.cfg
        A = len(self.a_xy)

        X = self._sense()
        self._last_sensor = X

        if A == 0:
            self.time += cfg.dt
            self.step_count += 1
            if len(self.a_xy) <= cfg.refill_threshold:
                self._refill_population()
            return

        # ---------- Encode via learnable E (sparse → a1[:, sel_idx]) ----------
        x = self._build_input_vector(X)  # (A, I)
        u_sel = np.einsum('ani,ai->an', self.gen_E, x, optimize=True)  # (A, N_sel)
        if cfg.enc_clip is not None and cfg.enc_clip > 0:
            np.clip(u_sel, -cfg.enc_clip, cfg.enc_clip, out=u_sel)

        lam = cfg.cem_lambda_write
        if lam == 1.0:
            self.cem_a1[:, cfg.sel_idx] = u_sel
        else:
            self.cem_a1[:, cfg.sel_idx] = (1.0 - lam) * self.cem_a1[:, cfg.sel_idx] + lam * u_sel

        # ---------- Integrate brain (a1,a2,m) ----------
        self.cem_a1, self.cem_a2, self.cem_m, bad = cem_integrate_rk2(
            self.cem_a1, self.cem_a2, self.cem_m, self.cem_k,
            cfg.cem_dt, cfg.cem_T, cfg.cem_abs_max
        )
        if bad.any():
            self.a_life[bad] = -1.0  # instant death

        # ---------- Readout from a1[:, sel_idx] → (steer, speed) ----------
        a1_sel = self.cem_a1[:, cfg.sel_idx]  # (A, N_sel)
        y = np.einsum('aon,an->ao', self.gen_R, a1_sel, optimize=True)  # (A, 2)
        steer_raw = y[:, 0] * cfg.cem_steer_gain
        speed_raw = y[:, 1] * cfg.cem_speed_gain

        steer = np.clip(steer_raw, -1.0, 1.0).astype(np.float32)
        speed_cmd = np.clip(0.5 + 0.5 * speed_raw, 0.0, 1.0).astype(np.float32)

        # ---------- Kinematics ----------
        self.a_dir = (self.a_dir + steer * (math.pi / 4.0)) % (2.0 * math.pi)
        self.a_speed = speed_cmd
        dx = self.a_speed * np.cos(self.a_dir) * cfg.speed_px_s * cfg.dt
        dy = self.a_speed * np.sin(self.a_dir) * cfg.speed_px_s * cfg.dt
        self.a_xy[:, 0] += dx
        self.a_xy[:, 1] += dy
        wrap_xy(self.a_xy, cfg.width, cfg.height)

        # ---------- EMA looping penalty ----------
        alpha = cfg.ema_coeff
        self.ema_steer = alpha * self.ema_steer + (1.0 - alpha) * steer
        loop_penalty_rate = cfg.looping_penalty * np.abs(self.ema_steer)

        # ---------- Life consumption ----------
        speed_cost = cfg.consumption_speed_coeff * self.a_speed
        baseline = cfg.consumption_base
        total_cost = baseline + speed_cost + loop_penalty_rate

        self.a_life -= cfg.dt * total_cost
        self.a_fit  -= cfg.dt * (speed_cost + loop_penalty_rate)

        # ---------- Food collisions ----------
        ate_any = np.zeros(A, dtype=bool)
        if len(self.f_xy):
            dist_af = pairwise_dist_torus(self.a_xy, self.f_xy, cfg.width, cfg.height)
            hit_f = dist_af < (cfg.agent_radius + cfg.food_radius)
            a_idx_f, f_idx = np.where(hit_f)
            if len(a_idx_f):
                eaten_counts = np.bincount(a_idx_f, minlength=A).astype(np.float32)
                self.a_life[:len(eaten_counts)] += eaten_counts * cfg.food_reward
                self.a_fit[:len(eaten_counts)]  += eaten_counts * cfg.food_reward
                ate_any[:len(eaten_counts)] = eaten_counts > 0

                keep = np.ones(len(self.f_xy), dtype=bool)
                keep[f_idx] = False
                self.f_xy = self.f_xy[keep]

        # ---------- Poison collisions ----------
        poisoned_any = np.zeros(A, dtype=bool)
        if len(self.p_xy):
            dist_ap = pairwise_dist_torus(self.a_xy, self.p_xy, cfg.width, cfg.height)
            hit_p = dist_ap < (cfg.agent_radius + cfg.poison_radius)
            a_idx_p, p_idx = np.where(hit_p)
            if len(a_idx_p):
                poisoned_counts = np.bincount(a_idx_p, minlength=A).astype(np.float32)
                self.a_life[:len(poisoned_counts)] += poisoned_counts * cfg.poison_reward
                self.a_fit[:len(poisoned_counts)]  += poisoned_counts * cfg.poison_reward
                poisoned_any[:len(poisoned_counts)] = poisoned_counts > 0

                keep_p = np.ones(len(self.p_xy), dtype=bool)
                keep_p[p_idx] = False
                self.p_xy = self.p_xy[keep_p]

        # ---------- NEXT step's reward signal (one-step delayed) ----------
        reward = ate_any.astype(np.int8) - poisoned_any.astype(np.int8)
        reward = np.clip(reward, -1, 1).astype(np.float32)
        self._reward_signal = reward

        # ---------- Cap life, spawn resources, cull dead ----------
        self.a_life = np.minimum(self.a_life, cfg.max_life).astype(np.float32)

        if (len(self.f_xy) < cfg.food_max) and (self.rng.random() < cfg.food_spawn_prob):
            self.f_xy = np.vstack([self.f_xy, self._spawn_food(1)]) if len(self.f_xy) else self._spawn_food(1)

        if (len(self.p_xy) < cfg.poison_max) and (self.rng.random() < cfg.poison_spawn_prob):
            self.p_xy = np.vstack([self.p_xy, self._spawn_poison(1)]) if len(self.p_xy) else self._spawn_poison(1)

        alive_mask = self.a_life > 0.0
        if not alive_mask.all():
            self._prune_dead(alive_mask)

        # ---------- Advance time ----------
        self.time += cfg.dt
        self.step_count += 1

        # ---------- GA refill trigger ----------
        if len(self.a_xy) <= cfg.refill_threshold:
            self._refill_population()

    def _prune_dead(self, alive_mask: np.ndarray):
        self.a_xy = self.a_xy[alive_mask]
        self.a_dir = self.a_dir[alive_mask]
        self.a_speed = self.a_speed[alive_mask]
        self.a_life = self.a_life[alive_mask]
        self.a_fit  = self.a_fit[alive_mask]

        self.cem_a1 = self.cem_a1[alive_mask]
        self.cem_a2 = self.cem_a2[alive_mask]
        self.cem_m  = self.cem_m[alive_mask]
        self.cem_k  = self.cem_k[alive_mask]

        self.gen_m0 = self.gen_m0[alive_mask]
        self.gen_E  = self.gen_E[alive_mask]
        self.gen_R  = self.gen_R[alive_mask]

        self._last_sensor = self._last_sensor[alive_mask] if len(self._last_sensor) else self._last_sensor
        self._reward_signal = self._reward_signal[alive_mask] if len(self._reward_signal) else self._reward_signal
        self.ema_steer = self.ema_steer[alive_mask]

    def _refill_population(self):
        cfg = self.cfg
        A = len(self.a_xy)
        target = cfg.n_target_agents
        needed = max(0, target - A)

        # ---- Elite pool by FITNESS (a_fit) ----
        if A > 0 and cfg.elite_fraction > 0.0:
            order = np.argsort(-self.a_fit)  # descending
            elite_size = max(1, int(math.ceil(cfg.elite_fraction * A)))
            elite_idx = order[:elite_size]
        else:
            elite_size = 0
            elite_idx = np.array([], dtype=int)

        # ---- Snapshot best-by-fitness into bank ----
        self._bank_cycle_count += 1
        if A > 0 and (self._bank_cycle_count % cfg.bank_add_every == 0):
            best_i = int(np.argmax(self.a_fit))
            entry = {
                "k":  self.cem_k[best_i].copy(),
                "m0": self.gen_m0[best_i].copy(),
                "E":  self.gen_E[best_i].copy(),
                "R":  self.gen_R[best_i].copy(),
                "fit": float(self.a_fit[best_i]),
                "life": float(self.a_life[best_i]),
                "step": int(self.step_count)
            }
            self.elite_bank.append(entry)
            if len(self.elite_bank) > cfg.bank_max:
                overflow = len(self.elite_bank) - cfg.bank_max
                if overflow > 0:
                    self.elite_bank = self.elite_bank[overflow:]

        # ---- Bank decay ----
        if len(self.elite_bank) and cfg.bank_decay_prob > 0.0:
            keep_list = []
            for e in self.elite_bank:
                if self.rng.random() >= cfg.bank_decay_prob:
                    keep_list.append(e)
            self.elite_bank = keep_list

        # Logging (before)
        mean_life_before = float(np.mean(self.a_life)) if A > 0 else float('nan')
        mean_fit_before  = float(np.mean(self.a_fit))  if A > 0 else float('nan')
        top5_before      = top_percentile_mean(self.a_fit, 0.05) if A > 0 else float('nan')

        if needed == 0:
            bank_size = len(self.elite_bank)
            log = {
                't': float(self.time), 'step': int(self.step_count),
                'alive_before': int(A), 'needed': 0,
                'immigrants': 0, 'children': 0, 'alive_after': int(A),
                'mean_life_before': mean_life_before,
                'mean_fit_before':  mean_fit_before,
                'top5_fit_before':  top5_before,
                'mean_life_after':  mean_life_before,
                'mean_fit_after':   mean_fit_before,
                'top5_fit_after':   top5_before,
                'elite_size': int(elite_size),
                'bank_size': int(bank_size)
            }
            self.ga_log.append(log)
            print(f"[Refill] t={self.time:.2f}s step={self.step_count} | alive={A}/{target} (no fill) | "
                  f"mean_fit={mean_fit_before:.2f} | top5%={top5_before:.2f} | bank={bank_size}")
            return

        # ---- FIXED immigrants + ONLY children for remainder ----
        immigrants = min(needed, len(self.elite_bank), cfg.immigrants_per_refill)
        children = needed - immigrants
        # If elite_size == 0 then children=0 (can't breed); we still add immigrants if available.

        N, I, O, N_sel = cfg.cem_N, cfg.input_dim, cfg.output_dim, cfg.N_sel

        new_xy = []
        new_dir = []
        new_speed = []
        new_life = []

        new_a1 = []
        new_a2 = []
        new_m  = []
        new_k  = []

        new_gen_m0 = []
        new_gen_E  = []
        new_gen_R  = []

        new_fit = []
        new_reward_signal = []
        new_ema_steer = []

        # Immigrants from bank (exact copies)
        if immigrants > 0:
            idxs = self.rng.choice(len(self.elite_bank), size=immigrants, replace=False)
            xy_i, dir_i, speed_i, life_i = random_spawn(immigrants, cfg.width, cfg.height, cfg.baseline_birth, self.rng)

            k_i  = np.empty((immigrants, cfg.cem_P), dtype=np.float32)
            m0_i = np.empty((immigrants, N), dtype=np.float32)
            E_i  = np.empty((immigrants, N_sel, I), dtype=np.float32)
            R_i  = np.empty((immigrants, O, N_sel), dtype=np.float32)

            for j, bank_idx in enumerate(idxs):
                be = self.elite_bank[bank_idx]
                k_i[j]  = be["k"]
                m0_i[j] = be["m0"]
                E_i[j]  = be["E"]
                R_i[j]  = be["R"]

            a1_i = np.zeros((immigrants, N), dtype=np.float32)
            a2_i = np.zeros((immigrants, N), dtype=np.float32)
            m_i  = m0_i.copy()

            new_xy.append(xy_i); new_dir.append(dir_i); new_speed.append(speed_i); new_life.append(life_i)
            new_a1.append(a1_i); new_a2.append(a2_i); new_m.append(m_i); new_k.append(k_i)
            new_gen_m0.append(m0_i); new_gen_E.append(E_i); new_gen_R.append(R_i)
            new_fit.append(np.zeros(immigrants, np.float32))
            new_reward_signal.append(np.zeros(immigrants, np.float32))
            new_ema_steer.append(np.zeros(immigrants, np.float32))

        # Children from fitness-elite via crossover + Student-t mutation
        if children > 0 and elite_size > 0:
            for _ in range(children):
                pa = int(elite_idx[self.rng.integers(0, elite_size)])
                pb = int(elite_idx[self.rng.integers(0, elite_size)])

                xy_c, dir_c, speed_c = spawn_near(self.a_xy[pa:pa+1], cfg.width, cfg.height, self.rng)
                life_c = np.array([cfg.baseline_birth], dtype=np.float32)

                k_c = crossover_mutate(self.rng, self.cem_k[pa], self.cem_k[pb],
                                       cfg.crossover_prob, cfg.mutation_prob,
                                       cfg.mutation_sigma_base, cfg.mutation_t_df)

                m0_c = crossover_mutate(self.rng, self.gen_m0[pa], self.gen_m0[pb],
                                        cfg.crossover_prob, cfg.mutation_prob,
                                        cfg.mutation_sigma_base, cfg.mutation_t_df)

                E_c = crossover_mutate(self.rng, self.gen_E[pa], self.gen_E[pb],
                                       cfg.crossover_prob, cfg.mutation_prob,
                                       cfg.mutation_sigma_base, cfg.mutation_t_df)

                R_c = crossover_mutate(self.rng, self.gen_R[pa], self.gen_R[pb],
                                       cfg.crossover_prob, cfg.mutation_prob,
                                       cfg.mutation_sigma_base, cfg.mutation_t_df)

                a1_c = np.zeros((1, N), dtype=np.float32)
                a2_c = np.zeros((1, N), dtype=np.float32)
                m_c  = m0_c[None, :].copy()

                new_xy.append(xy_c); new_dir.append(dir_c); new_speed.append(speed_c); new_life.append(life_c)
                new_a1.append(a1_c); new_a2.append(a2_c); new_m.append(m_c); new_k.append(k_c[None, ...])
                new_gen_m0.append(m0_c[None, ...]); new_gen_E.append(E_c[None, ...]); new_gen_R.append(R_c[None, ...])
                new_fit.append(np.zeros(1, np.float32))
                new_reward_signal.append(np.zeros(1, np.float32))
                new_ema_steer.append(np.zeros(1, np.float32))

        # Concatenate additions
        if len(new_xy):
            self.a_xy = np.vstack([self.a_xy] + new_xy) if len(self.a_xy) else np.vstack(new_xy)
            self.a_dir = np.concatenate([self.a_dir] + new_dir) if len(self.a_dir) else np.concatenate(new_dir)
            self.a_speed = np.concatenate([self.a_speed] + new_speed) if len(self.a_speed) else np.concatenate(new_speed)
            self.a_life = np.concatenate([self.a_life] + new_life) if len(self.a_life) else np.concatenate(new_life)

            self.cem_a1 = np.vstack([self.cem_a1] + new_a1) if len(self.cem_a1) else np.vstack(new_a1)
            self.cem_a2 = np.vstack([self.cem_a2] + new_a2) if len(self.cem_a2) else np.vstack(new_a2)
            self.cem_m  = np.vstack([self.cem_m]  + new_m ) if len(self.cem_m)  else np.vstack(new_m)
            self.cem_k  = np.vstack([self.cem_k]  + new_k ) if len(self.cem_k)  else np.vstack(new_k)

            self.gen_m0 = np.vstack([self.gen_m0] + new_gen_m0) if len(self.gen_m0) else np.vstack(new_gen_m0)
            self.gen_E  = np.vstack([self.gen_E]  + new_gen_E ) if len(self.gen_E)  else np.vstack(new_gen_E)
            self.gen_R  = np.vstack([self.gen_R]  + new_gen_R ) if len(self.gen_R)  else np.vstack(new_gen_R)

            self.a_fit = np.concatenate([self.a_fit] + new_fit) if len(self.a_fit) else np.concatenate(new_fit)
            self._reward_signal = np.concatenate([self._reward_signal] + new_reward_signal) if len(self._reward_signal) else np.concatenate(new_reward_signal)
            self.ema_steer = np.concatenate([self.ema_steer] + new_ema_steer) if len(self.ema_steer) else np.concatenate(new_ema_steer)

            if len(self._last_sensor):
                extra = np.ones((len(self.a_xy) - len(self._last_sensor), 2*self.cfg.n_rays), np.float32)
                self._last_sensor = np.vstack([self._last_sensor, extra])
            else:
                self._last_sensor = np.ones((len(self.a_xy), 2*self.cfg.n_rays), np.float32)

        A_after = len(self.a_xy)
        mean_life_after = float(np.mean(self.a_life)) if A_after > 0 else float('nan')
        mean_fit_after  = float(np.mean(self.a_fit))  if A_after > 0 else float('nan')
        top5_after      = top_percentile_mean(self.a_fit, 0.05) if A_after > 0 else float('nan')

        log = {
            't': float(self.time),
            'step': int(self.step_count),
            'alive_before': int(A),
            'needed': int(needed),
            'immigrants': int(immigrants),
            'children': int(children),
            'alive_after': int(A_after),
            'mean_life_before': mean_life_before,
            'mean_fit_before':  mean_fit_before,
            'top5_fit_before':  top5_before,
            'mean_life_after':  mean_life_after,
            'mean_fit_after':   mean_fit_after,
            'top5_fit_after':   top5_after,
            'elite_size': int(elite_size),
            'bank_size': int(len(self.elite_bank))
        }
        self.ga_log.append(log)
        print(f"[Refill] t={self.time:.2f}s step={self.step_count} | alive {A}→{A_after} (need {needed}, imm {immigrants}, child {children}) | "
              f"mean_fit={mean_fit_after:.2f} | top5%={top5_after:.2f} | bank={len(self.elite_bank)}")

    def draw(self, screen):
        if not (_PG and (pygame is not None)):
            return
        cfg = self.cfg
        screen.fill((0, 0, 0))

        k = cfg.n_rays
        for i, (pos, heading) in enumerate(zip(self.a_xy, self.a_dir)):
            for j, rel_angle in enumerate(self._ray_rel_angles):
                ray_angle = heading + rel_angle
                ray_dir = np.array([math.cos(ray_angle), math.sin(ray_angle)])
                ray_end = pos + ray_dir * cfg.ray_length
                ray_end[0] = np.clip(ray_end[0], 0, cfg.width)
                ray_end[1] = np.clip(ray_end[1], 0, cfg.height)

                food_d = self._last_sensor[i, j] if i < len(self._last_sensor) else 1.0
                poison_d = self._last_sensor[i, j + k] if i < len(self._last_sensor) else 1.0

                if food_d < 0.9:
                    color = (0, 255, 0); thickness = 2
                elif poison_d < 0.9:
                    color = (255, 100, 100); thickness = 2
                else:
                    color = (60, 60, 60); thickness = 1

                pygame.draw.line(screen, color, pos.astype(int), ray_end.astype(int), thickness)

        # Food (green)
        for pos in self.f_xy:
            pygame.draw.circle(screen, (0, 180, 0), pos.astype(int), int(cfg.food_radius))

        # Poison (red)
        for pos in self.p_xy:
            pygame.draw.circle(screen, (220, 50, 50), pos.astype(int), int(cfg.poison_radius))

        # Agents
        for pos, ang in zip(self.a_xy, self.a_dir):
            pygame.draw.circle(screen, (50, 120, 255), pos.astype(int), int(cfg.agent_radius))
            end = pos + np.array([math.cos(ang), math.sin(ang)]) * cfg.agent_radius
            pygame.draw.line(screen, (255, 255, 255), pos.astype(int), end.astype(int), 2)
