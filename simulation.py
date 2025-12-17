# simulation.py

from multiprocessing import get_context
from sim_workers import simulation_worker
import argparse
import os


def run_parallel_simulation(
    N_total: int,
    OPTtype: str = "call",
    chunk_size: int = 5000,
    out_dir: str = "simulation_output",
    seed_base: int = 100_000,
    n_workers: int = 0,       # 0 = auto (SLURM-aware)
    max_workers: int = 24,     # ICC-safe default; bump to 12/16 after stable
):
    os.makedirs(out_dir, exist_ok=True)

    # Choose worker count
    if n_workers and n_workers > 0:
        workers = n_workers
    else:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            workers = int(slurm_cpus)
        else:
            workers = max(1, (os.cpu_count() or 1) - 3)

    workers = max(1, min(workers, max_workers))

    # Distribute rows so we don't drop N_total % workers
    base = N_total // workers
    rem = N_total % workers
    rows = [base + (1 if i < rem else 0) for i in range(workers)]

    # Build args list
    args_list = [
        (wid, rows[wid], OPTtype, chunk_size, out_dir, seed_base)
        for wid in range(workers)
        if rows[wid] > 0
    ]

    print(
        f"[simulation.py] N_total={N_total} OPTtype={OPTtype} "
        f"workers={workers} chunk_size={chunk_size} out_dir={out_dir} seed_base={seed_base}"
    )

    # Multiprocessing: spawn is safest for QuantLib
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers, maxtasksperchild=25) as pool:
        pool.starmap(simulation_worker, args_list, chunksize=1)


def main():
    p = argparse.ArgumentParser(description="Run Heston American option simulations in parallel (ICC-friendly).")
    p.add_argument("--N_total", type=int, required=True, help="Total number of simulated rows to generate.")
    p.add_argument("--OPTtype", type=str, default="call", choices=["call", "put"], help="Option type.")
    p.add_argument("--chunk_size", type=int, default=5000, help="Rows per parquet chunk written by each worker.")
    p.add_argument("--out_dir", type=str, default="simulation_output", help="Output directory for parquet files.")
    p.add_argument("--seed_base", type=int, default=100_000, help="Base seed for RNG (worker seeds derived from this).")
    p.add_argument("--n_workers", type=int, default=0, help="Override worker count (0 = auto).")
    p.add_argument("--max_workers", type=int, default=24, help="Cap workers even if SLURM allocates more CPUs.")

    args = p.parse_args()

    run_parallel_simulation(
        N_total=args.N_total,
        OPTtype=args.OPTtype,
        chunk_size=args.chunk_size,
        out_dir=args.out_dir,
        seed_base=args.seed_base,
        n_workers=args.n_workers,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
