from heston.pricer_class import HestonFDPricer

def simulation_worker(
    worker_id: int,
    n_rows: int,
    OPTtype: str,
    chunk_size: int,
    out_dir: str,
    seed_base: int = 0,
):
    import numpy as np
    import pandas as pd
    import QuantLib as ql
    import os

    os.makedirs(out_dir, exist_ok=True)

    # QuantLib globals (per-process)
    calendar = ql.UnitedStates(ql.UnitedStates.Settlement)
    todays_date = calendar.adjust(ql.Date.todaysDate())
    ql.Settings.instance().evaluationDate = todays_date
    day_count = ql.Actual365Fixed()

    pricer = HestonFDPricer(
        todays_date=todays_date,
        calendar=calendar,
        day_count=day_count,
        engine_grid=(900, 1500, 700),
        damping_steps=10,
        scheme="hundsdorfer",
        spot_init=205.0,
        r_init=0.03,
        q_init=0.01,
    )

    buffer = []
    chunk_id = 0

    # One RNG per worker; advance deterministically per chunk
    rng = np.random.default_rng(seed_base + worker_id * 1_000_000)

    for i in range(n_rows):
        S = 205.0
        K = rng.uniform(5, 405)
        q = rng.uniform(0.00, 0.06)
        r = rng.uniform(0.01, 0.06)
        T = rng.uniform(0.003, 3.0)

        v0    = rng.uniform(0.02, 0.50)
        theta = rng.uniform(0.01, 1.50)
        kappa = rng.uniform(0.01, 2.00)
        sigma = rng.uniform(0.01, 1.00)
        rho   = rng.uniform(-1.00, 0.00)

        # Price; protect the whole job from a single bad draw
        try:
            pricer.set_market(r=r, q=q)
            price = pricer.price_american(
                K, T, v0, theta, kappa, sigma, rho, option_type=OPTtype
            )
        except Exception:
            price = np.nan

        buffer.append({
            "S": S, "K": K, "r": r, "q": q, "T": T,
            "v0": v0, "heston_theta": theta, "heston_kappa": kappa,
            "heston_sigma": sigma, "heston_rho": rho,
            "price_american": price,
        })

        # Flush buffer
        if (i + 1) % chunk_size == 0 or (i + 1) == n_rows:
            df = pd.DataFrame(buffer)
            fname = os.path.join(out_dir, f"{OPTtype}_worker{worker_id:03d}_chunk{chunk_id:05d}.parquet")
            df.to_parquet(fname, index=False)

            buffer.clear()
            chunk_id += 1

    # IMPORTANT: return something tiny (or None) to avoid IPC blowups
    return worker_id
