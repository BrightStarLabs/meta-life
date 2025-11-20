import numpy as np

try:
    import pygame
    _PG = True
except ModuleNotFoundError:
    pygame = None
    _PG = False

from config import Cfg
from world import World
from util import top_percentile_mean, build_cfg_from_hparams

def run(cfg: Cfg, steps: int = 3000, vis: bool = False, fps: int = 60) -> World:
    world = World(cfg)
    if vis:
        if not _PG or (pygame is None):
            raise RuntimeError("Pygame not available but vis=True was requested.")

        pygame.init()
        screen = pygame.display.set_mode((cfg.width, cfg.height))
        pygame.display.set_caption("Living Game – 2-Channel Cortex + Memory (Sparse Enc/Dec + Fixed Immigrants)")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            world.step()
            world.draw(screen)

            mean_life = float(np.mean(world.a_life)) if len(world.a_life) else float('nan')
            mean_fit  = float(np.mean(world.a_fit))  if len(world.a_fit)  else float('nan')
            top5_fit  = top_percentile_mean(world.a_fit, 0.05) if len(world.a_fit) else float('nan')
            ema_bias_mean = float(np.mean(np.abs(world.ema_steer))) if len(world.ema_steer) else float('nan')

            hud = (
                f"t={world.time:6.2f}s | alive={len(world.a_xy):3d}/{cfg.n_target_agents} "
                f"| food={len(world.f_xy):3d} | poison={len(world.p_xy):3d} "
                f"| mean life={mean_life:.2f} | mean fit={mean_fit:.2f} | top5% fit={top5_fit:.2f} "
                f"| |EMA(steer)|={ema_bias_mean:.2f} | bank={len(world.elite_bank)}"
            )
            surf = font.render(hud, True, (255, 255, 255))
            screen.blit(surf, (10, 10))

            pygame.display.flip()
            clock.tick(fps)
        pygame.quit()
    else:
        report_every = max(1, int(0.5 / cfg.dt))  # ~ every 0.5s
        for i in range(steps):
            world.step()
            if (i % report_every) == 0:
                mean_life = float(np.mean(world.a_life)) if len(world.a_life) else float('nan')
                mean_fit  = float(np.mean(world.a_fit))  if len(world.a_fit)  else float('nan')
                top5_fit  = top_percentile_mean(world.a_fit, 0.05) if len(world.a_fit) else float('nan')
                ema_bias_mean = float(np.mean(np.abs(world.ema_steer))) if len(world.ema_steer) else float('nan')
                print(
                    f"t={world.time:6.2f}s | step={world.step_count:6d} | alive={len(world.a_xy):3d}/{cfg.n_target_agents} "
                    f"| food={len(world.f_xy):3d} | poison={len(world.p_xy):3d} "
                    f"| mean life={mean_life:.2f} | mean fit={mean_fit:.2f} | top5% fit={top5_fit:.2f} "
                    f"| |EMA(steer)|={ema_bias_mean:.2f} | bank={len(world.elite_bank)}"
                )

    return world

if __name__ == "__main__":

    # === Hyperparameters / Config ===
    HYPERPARAMS = dict(
        # World / timing
        width=1500,
        height=900,
        dt=1/30.0,
        speed_px_s=55.0,

        # Sensors (normalized distance 0..1, 1.0 = no hit)
        n_rays=3,                 # I = 2*n_rays + 1 (reward)
        fov_deg=120.0,
        ray_length=70.0,

        # Agents
        n_target_agents=100,
        baseline_birth=8.0,
        agent_radius=5.0,
        max_life=25.0,

        # Food (green)
        food_radius=3.5,
        food_reward=+5.0,
        food_spawn_prob=0.7,
        food_max=80,

        # Poison (red)
        poison_radius=3.5,
        poison_reward=-8.0,
        poison_spawn_prob=0.7,
        poison_max=100,

        # Costs (life drain per second)
        consumption_base=0.2,
        consumption_speed_coeff=0.1,

        # Looping penalty (EMA)
        ema_coeff=0.98,           # ema = alpha*ema + (1-alpha)*steer
        looping_penalty=5.0,      # per-second drain scaled by |ema_steer|

        # === CEM Brain grid ===
        cem_N=24,                 # grid length
        cem_T=30,                 # microsteps per world tick
        cem_dt=0.1,               # microstep size
        cem_abs_max=10.0,
        cem_lambda_write=0.1,     # leaky write into a1 from encoder

        # ODE coefficients per agent (P=7; see header)
        cem_P=7,

        # Sparse encode/decode
        mod_every=3,              # use every N%mod_every cells
        enc_clip=2.0,             # clip(E@x) before mixing into a1

        # Genome init (encoder/readout, memory prior, coeffs)
        e_init_sigma=0.8,
        r_init_sigma=0.8,
        k_init_sigma=0.8,
        m0_init_sigma=1,

        # Readout gains
        cem_steer_gain=1.0,
        cem_speed_gain=2.0,

        # Refill trigger & composition
        refill_frac=0.4,         # trigger when alive <= refill_frac * n_target_agents
        elite_fraction=0.6,       # parents sampled from top fraction by FITNESS (a_fit) for children
        immigrants_per_refill=3,  # FIXED number of immigrants each refill (if available in bank)

        # GA ops
        crossover_prob=0.08,

        # Fat-tail mutation (Student-t)
        mutation_prob=0.2,
        mutation_sigma_base=0.05,
        mutation_t_df=2.0,

        # Elite bank
        bank_max=512,
        bank_decay_prob=0.0,
        bank_add_every=2
    )

    # --- Build config & run ---
    cfg = build_cfg_from_hparams(HYPERPARAMS)
    vis = True
    world = run(cfg, steps=1500, vis=vis, fps=60)

    print(f"\nRefill log entries: {len(world.ga_log)} (show last 5)")
    for entry in world.ga_log[-5:]:
        print(entry)