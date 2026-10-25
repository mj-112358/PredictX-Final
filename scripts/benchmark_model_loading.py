"""Controlled warm-cache model loading, separate from order-dependent process startup."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd

from predictx.data import ROOT
from predictx.inference import DISEASES, load_model, registry

OUT = ROOT / "results/reproduced/model_loading"
OUT.mkdir(parents=True, exist_ok=True)


def worker(d, conn):
    load_model(d)
    conn.send("warm")
    while conn.recv():
        load_model.cache_clear()
        t = time.perf_counter_ns()
        load_model(d)
        conn.send(time.perf_counter_ns() - t)
    conn.close()


def main():
    # Warm imports/file caches explicitly before every timed comparison.
    for d in DISEASES:
        load_model(d)
    ctx = mp.get_context("spawn")
    workers = []
    for d in DISEASES:
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(d, b))
        p.start()
        workers.append((p, a))
    for p, a in workers:
        assert a.recv() == "warm"
    rows = []
    rng = np.random.default_rng(2026)
    try:
        for iteration in range(10):
            for mode in rng.permutation(
                [
                    "integrated_reload",
                    "separate_sequential_reload",
                    "separate_parallel_reload",
                ]
            ):
                t = time.perf_counter_ns()
                service_ns = []
                if mode == "integrated_reload":
                    load_model.cache_clear()
                    for d in DISEASES:
                        a = time.perf_counter_ns()
                        load_model(d)
                        service_ns.append(time.perf_counter_ns() - a)
                elif mode == "separate_sequential_reload":
                    for p, a in workers:
                        a.send(True)
                        service_ns.append(a.recv())
                else:
                    for p, a in workers:
                        a.send(True)
                    service_ns = [a.recv() for p, a in workers]
                rows.append(
                    {
                        "mode": mode,
                        "iteration": iteration,
                        "five_model_wall_ms": (time.perf_counter_ns() - t) / 1e6,
                        "summed_service_ms": sum(service_ns) / 1e6,
                    }
                )
    finally:
        for p, a in workers:
            a.send(False)
        for p, a in workers:
            p.join(30)
            a.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "model_loading_samples.csv", index=False)
    summary = df.groupby("mode").five_model_wall_ms.agg(["median", "mean", "max"]).reset_index()
    summary.to_csv(OUT / "model_loading_benchmark.csv", index=False)
    (OUT / "model_loading_scope.json").write_text(
        json.dumps(
            {
                "model_registry": registry(),
                "repetitions": 10,
                "seed": 2026,
                "scope": "hash verification plus deserialization with imports and file caches warm; actual five worker processes; local IPC included for separate modes",
                "mode_order": "randomized within each repetition",
                "cold_disk": "not measured",
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
