"""Randomized CPU benchmarks with matched records and seven execution modes."""

import os

for k in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util
import json
import multiprocessing as mp
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from predictx.data import ROOT
from predictx.inference import (
    DISEASES,
    assess,
    evaluate,
    finish_assessment,
    load_model,
    prepare_assessment,
    registry,
)
from predictx.processing import process_panel
from predictx.schema import LabPanel

spec = importlib.util.spec_from_file_location(
    "benchmark_fixture", ROOT / "scripts/benchmark_integration.py"
)
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)
R = ROOT / "results/reproduced/benchmarks"
R.mkdir(parents=True, exist_ok=True)


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        rss = (
            int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip())
            * 1024
        )
    except Exception:
        rss = None
    pss = None
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/self/smaps_rollup"):
                if line.startswith("Pss:"):
                    pss = int(line.split()[1]) * 1024
        except OSError:
            pass
    return {
        "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
        "current_rss_bytes": rss,
        "current_pss_bytes": pss,
    }


def minimal(payload, disease):
    features = set(registry()[disease]["features"]) | set(registry()[disease]["essential"])
    deps = {
        "nlr": ["neutrophils", "lymphocytes"],
        "plr": ["platelets", "lymphocytes"],
        "mlr": ["monocytes", "lymphocytes"],
        "ast_alt": ["ast", "alt"],
        "albumin_globulin": ["albumin", "total_protein"],
        "non_hdl": ["total_cholesterol", "hdl"],
        "wbc_cells": ["wbc"],
    }
    for k, v in deps.items():
        if k in features:
            features.update(v)
    # QC depends on all differential counts if any count is consumed. Retain that evidence.
    if features & {
        "neutrophils",
        "lymphocytes",
        "monocytes",
        "eosinophils",
        "basophils",
    }:
        features.update(
            [
                "wbc",
                "neutrophils",
                "lymphocytes",
                "monocytes",
                "eosinophils",
                "basophils",
            ]
        )
    return {
        **payload,
        "measurements": {k: v for k, v in payload["measurements"].items() if k in features},
    }


def worker(diseases, conn):
    with threadpool_limits(limits=1):
        start = time.perf_counter_ns()
        [load_model(d) for d in diseases]
        if "liver" in diseases:
            import torch

            torch.set_num_threads(1)
        conn.send({"load_ns": time.perf_counter_ns() - start, "memory": memory()})
        while True:
            payloads = conn.recv()
            if payloads is None:
                break
            scores = []
            counts = []
            start = time.perf_counter_ns()
            for payload in payloads:
                output = []
                for disease in diseases:
                    panel = LabPanel.model_validate(minimal(payload, disease))
                    r = process_panel(panel)
                    out = assess(panel, r, disease)
                    output.append(out["score"])
                    counts.append(r.counters)
                scores.append(output)
            conn.send(
                {
                    "scores": scores,
                    "service_ns": time.perf_counter_ns() - start,
                    "counters": counts,
                    "memory": None,
                }
            )
        conn.close()


def spawn_groups(groups):
    ctx = mp.get_context("spawn")
    workers = []
    start = time.perf_counter_ns()
    for group in groups:
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(group, b))
        p.start()
        workers.append((p, a))
    info = [conn.recv() for proc, conn in workers]
    return workers, info, time.perf_counter_ns() - start


def stop(workers):
    for p, c in workers:
        c.send(None)
    for p, c in workers:
        p.join(30)
        c.close()


def run(mode, payloads, workers):
    counts = []
    if mode.startswith("five_workers"):
        responses = []
        if mode.endswith("parallel"):
            for p, c in workers:
                c.send(payloads)
            responses = [c.recv() for p, c in workers]
        else:
            for p, c in workers:
                c.send(payloads)
                responses.append(c.recv())
        scores = np.column_stack([np.array(r["scores"]).ravel() for r in responses])
        counts = [c for r in responses for c in r["counters"]]
    elif mode == "optimized_vectorized_shared_ingest":
        panels = [LabPanel.model_validate(v) for v in payloads]
        records = [process_panel(p) for p in panels]
        scores = np.empty((len(records), 5))
        for k, disease in enumerate(DISEASES):
            cfg = registry()[disease]
            b = load_model(disease)
            frame = pd.DataFrame(
                [{c: r.values.get(c, np.nan) for c in b.preprocessor.columns} for r in records]
            )
            vectors = b.preprocessor.transform(frame)
            scores[:, k] = b.predict_preprocessed(vectors)[:, 0]
            for j, (panel, record) in enumerate(zip(panels, records)):
                result, _, _ = prepare_assessment(panel, record, disease, b)
                assert result["eligible"]
                output = finish_assessment(result, cfg, b, float(scores[j, k]), vectors[j : j + 1])
                assert output["score"] is not None
        counts = [r.counters for r in records]
        counts[0]["transform_calls"] = 5
        counts[0]["projection_builds"] = 5
    else:
        scores = []
        for payload in payloads:
            if mode == "integrated":
                out = evaluate(LabPanel.model_validate(payload))
                scores.append([o["score"] for o in out["outputs"]])
                counts.append(out["processing"])
            elif mode == "integrated_no_transform_cache":
                p = LabPanel.model_validate(payload)
                r = process_panel(p)
                out = []
                for disease in DISEASES:
                    r.transformed.clear()
                    out.append(assess(p, r, disease)["score"])
                scores.append(out)
                counts.append(r.counters)
            else:
                out = []
                for disease in DISEASES:
                    data = (
                        minimal(payload, disease)
                        if mode == "independent_feature_minimal"
                        else payload
                    )
                    p = LabPanel.model_validate(data)
                    r = process_panel(p)
                    out.append(assess(p, r, disease)["score"])
                    counts.append(r.counters)
                scores.append(out)
        scores = np.asarray(scores)
    return scores, {k: sum(c[k] for c in counts) for k in counts[0]}


def model_size(d):
    cfg = registry()[d]
    b = load_model(d)
    m = b.model
    if hasattr(m, "coef_"):
        count = int(m.coef_.size + m.intercept_.size)
        definition = "classifier coefficients and intercept, excluding preprocessing/calibrator"
    elif hasattr(m, "parameters"):
        count = sum(p.numel() for p in m.parameters())
        definition = "all neural learned scalar parameters, excluding preprocessing/calibrator"
    elif hasattr(m, "_predictors"):
        count = int(
            sum(tree.nodes["is_leaf"].sum() for stage in m._predictors for tree in stage)
            + np.size(m._baseline_prediction)
        )
        definition = "boosting leaf prediction values and baseline"
    else:
        count = None
        definition = "not available"
    return {
        "disease": d,
        "artifact_bytes": (ROOT / cfg["artifact"]).stat().st_size,
        "learned_scalar_count": count,
        "parameter_definition": definition,
        "sha256": cfg["sha256"],
    }


def main():
    fixture = bm.fixture()
    (R / "SYNTHETIC_benchmark_input.json").write_text(json.dumps(fixture, indent=2))
    [load_model(d) for d in DISEASES]
    import torch

    torch.set_num_threads(1)
    workers, initial, launch = spawn_groups([[d] for d in DISEASES])
    rows = []
    order = []
    rng = np.random.default_rng(2026)
    modes = [
        "integrated",
        "integrated_no_transform_cache",
        "independent_full_processing",
        "independent_feature_minimal",
        "optimized_vectorized_shared_ingest",
        "five_workers_sequential",
        "five_workers_parallel",
    ]
    try:
        for size in [1, 8, 32]:
            payloads = [
                {**fixture, "patient_id": f"SYNTHETIC_BENCHMARK_{j}", "age": 45 + j % 3}
                for j in range(size)
            ]
            reference, _ = run("integrated", payloads, workers)
            assert np.isfinite(reference).all()
            for mode in modes:
                for _ in range(5):
                    run(mode, payloads, workers)
            for round_ in range(8):
                for mode in rng.permutation(modes):
                    order.append({"batch_size": size, "round": round_, "mode": mode})
                    for repeat in range(3):
                        start = time.perf_counter_ns()
                        scores, counters = run(mode, payloads, workers)
                        ns = time.perf_counter_ns() - start
                        np.testing.assert_allclose(scores, reference, rtol=1e-6, atol=1e-7)
                        rows.append(
                            {
                                "batch_size": size,
                                "round": round_,
                                "repeat": repeat,
                                "mode": mode,
                                "latency_ms": ns / 1e6,
                                "per_record_ms": ns / 1e6 / size,
                                **counters,
                            }
                        )
            print("Completed randomized batch-size benchmark", size, flush=True)
    finally:
        stop(workers)
    df = pd.DataFrame(rows)
    df.to_csv(R / "benchmark_samples.csv", index=False)
    pd.DataFrame(order).to_csv(R / "benchmark_order.csv", index=False)
    summary = (
        df.groupby(["batch_size", "mode"])
        .agg(
            measurements=("latency_ms", "size"),
            p50_ms=("latency_ms", "median"),
            p95_ms=("latency_ms", lambda z: z.quantile(0.95)),
            mean_per_record_ms=("per_record_ms", "mean"),
            canonical_measurements=("canonical_measurements", "first"),
            transforms=("transform_calls", "first"),
            cache_hits=("transform_cache_hits", "first"),
        )
        .reset_index()
    )
    summary.to_csv(R / "benchmark_summary.csv", index=False)
    # Measure startup in fresh interpreters with the operating-system file cache retained.
    starts = []
    for round_ in range(3):
        for mode in rng.permutation(["one_process_five_models", "five_process_one_model_each"]):
            groups = (
                [list(DISEASES)] if mode == "one_process_five_models" else [[d] for d in DISEASES]
            )
            w, info, ns = spawn_groups(groups)
            starts.append(
                {
                    "round": round_,
                    "mode": mode,
                    "process_cold_start_ms": ns / 1e6,
                    "summed_peak_rss_bytes": sum(i["memory"]["peak_rss_bytes"] for i in info),
                    "summed_current_pss_bytes": None
                    if any(i["memory"]["current_pss_bytes"] is None for i in info)
                    else sum(i["memory"]["current_pss_bytes"] for i in info),
                }
            )
            stop(w)
        print("Completed process-cold startup repetition", round_, flush=True)
    pd.DataFrame(starts).to_csv(R / "process_cold_start_samples.csv", index=False)
    pd.DataFrame([model_size(d) for d in DISEASES]).to_csv(
        R / "model_size_parameters.csv", index=False
    )
    meta = {
        "evidence": "SYNTHETIC_INPUT_EXECUTED_SOFTWARE_NOT_CLINICAL",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version,
        "threads": [
            {**info, "filepath": Path(info["filepath"]).name} for info in threadpool_info()
        ],
        "model_threads": 1,
        "torch_threads": torch.get_num_threads(),
        "gpu_used": False,
        "affinity": "OS scheduling on a shared macOS host",
        "warmup_per_mode_per_batch": 5,
        "randomized_rounds": 8,
        "repeats_per_round": 3,
        "batch_sizes": [1, 8, 32],
        "output_equivalence": "every mode/iteration within rtol 1e-6, atol 1e-7 of all five true model scores",
        "integrated_memory_after_benchmark": memory(),
        "five_workers_memory_after_load": [i["memory"] for i in initial],
        "pss_status": "Linux smaps_rollup when supported; unavailable on this macOS host",
        "process_cold_start_definition": "fresh interpreter, imports and model process; filesystem cache retained; randomized process startup order",
        "warm_start_definition": "persistent loaded models, 5 warmup batches per mode and size",
        "independent_optimized_baseline": "complete endpoint/support/score/threshold/explanation output; shared ingestion and derivation; per-model vectorized transformations and predictions",
        "integrated_batch_definition": "isolated scalar inference per patient",
        "ipc": "local multiprocessing Pipe with serialization",
        "memory_limit": "summed RSS double-counts shared pages; peaks may occur at different times; current PSS unavailable on macOS",
        "checkpoint_hashes": {d: registry()[d]["sha256"] for d in DISEASES},
    }
    if sys.platform == "darwin":
        meta["hardware_after_run"] = {
            k: subprocess.check_output(["/usr/sbin/sysctl", "-n", k], text=True).strip()
            for k in [
                "hw.model",
                "machdep.cpu.brand_string",
                "hw.physicalcpu",
                "hw.memsize",
            ]
        }
        meta["hardware_note"] = (
            "Hardware identifiers queried on the measurement host; Python process architecture recorded separately."
        )
    (R / "benchmark_environment.json").write_text(json.dumps(meta, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
