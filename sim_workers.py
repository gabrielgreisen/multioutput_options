import os
import time
import numpy as np
import pandas as pd
import QuantLib as ql
from multiprocessing import get_context

from heston.heston_pricing import price_heston_american_option
from heston.pricer_class import HestonFDPricer

def simulation_worker(
    worker_id,
    n_rows,
    OPTtype,
    chunk_size,
    out_dir,
    seed_offset=0,
):
    import scipy.stats as stat
    import pandas as pd
    import time
    import QuantLib as ql
    import numpy as np
    import os

    rng = np.random.default_rng(seed_offset + worker_id)

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

    for i in range(n_rows):
        start_time = time.perf_counter()
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

        pricer.set_market(r=r, q=q)
        price = pricer.price_american(K, T, v0, theta, kappa, sigma, rho, option_type=OPTtype)
        buffer.append({
            "S": S, "K": K, "r": r, "q": q, "T": T,
            "v0": v0, "heston_theta": theta, "heston_kappa": kappa,
            "heston_sigma": sigma, "heston_rho": rho,
            "price_american": price,
        })
        end_time = time.perf_counter()
        print(f"time per run{i}: {end_time-start_time}")

        if (i + 1) % chunk_size == 0 or (i + 1) == n_rows:
            df = pd.DataFrame(buffer)
            fname = f"{out_dir}/worker{worker_id}_chunk{chunk_id:04d}.parquet"
            df.to_parquet(fname, index=False)
            buffer.clear()
            chunk_id += 1
