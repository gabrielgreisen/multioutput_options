from sim_workers import simulation_worker
from multiprocessing import get_context
import argparse
import os

def run_parallel_simulation(
    N_total,
    OPTtype="call",
    chunk_size=5000,
    out_dir="simulation_output",
):
    os.makedirs(out_dir, exist_ok=True)

    n_workers = max(1, os.cpu_count() - 3)
    rows_per_worker = N_total // n_workers

    ctx = get_context("spawn")  # safest for QuantLib

    with ctx.Pool(processes=n_workers) as pool:
        pool.starmap(
            simulation_worker,
            [
                (wid, rows_per_worker, OPTtype, chunk_size, out_dir, 100_000)
                for wid in range(n_workers)
            ],
        )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N_total", type=int, required=True)
    p.add_argument("--OPTtype", type=str, default="call", choices=["call", "put"])
    p.add_argument("--chunk_size", type=int, default=5000)
    p.add_argument("--out_dir", type=str, default="simulation_output")
    p.add_argument("--seed_base", type=int, default=100_000)
    args = p.parse_args()

    run_parallel_simulation(
        N_total=args.N_total,
        OPTtype=args.OPTtype,
        chunk_size=args.chunk_size,
        out_dir=args.out_dir,
        seed_base=args.seed_base,
    )

if __name__ == "__main__":
    main()