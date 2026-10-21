"""Exact Gaussian composition, conditional on a public fixed-slot schedule.

No sampling amplification. Neighboring datasets replace one public slot by a
zero contribution, retaining its positions, token-independent schedule and N.
"""
from collections import Counter
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr


def fixed_schedule(n, steps, batch, seed):
    assert n > 0 and steps > 0 and batch > 0
    rng = np.random.default_rng(seed)
    stream = []
    while len(stream) < steps * batch:
        stream.extend(rng.permutation(n).tolist())
    return np.asarray(stream[:steps * batch], dtype=np.int64).reshape(steps, batch)


def participation(schedule, n):
    counts = np.bincount(schedule.ravel(), minlength=n)
    squared = np.zeros(n, dtype=np.int64)
    for batch in schedule:
        for i, multiplicity in Counter(batch.tolist()).items():
            squared[i] += multiplicity ** 2
    return {"counts": counts.tolist(), "squared_multiplicities": squared.tolist(),
            "max_participations": int(counts.max()), "max_squared_multiplicity_sum": int(squared.max()),
            "steps": len(schedule), "logical_batch": schedule.shape[1], "sampling_amplification": False}


def gdp_delta(epsilon, mu):
    assert epsilon >= 0 and mu > 0
    a = float(log_ndtr(-epsilon / mu + mu / 2))
    b = epsilon + float(log_ndtr(-epsilon / mu - mu / 2))
    return math.exp(a) * -math.expm1(min(0.0, b - a))


def epsilon_for_mu(mu, delta):
    if gdp_delta(0, mu) <= delta:
        return 0.0
    upper = 1.0
    while gdp_delta(upper, mu) > delta:
        upper *= 2
    return float(brentq(lambda e: gdp_delta(e, mu) - delta, 0, upper, xtol=1e-12))


def calibrate(audit, epsilon=6.0, delta=1e-5, clip=1.0):
    mu = float(brentq(lambda m: gdp_delta(epsilon, m) - delta, 1e-4, 100, xtol=1e-12))
    k2 = audit["max_squared_multiplicity_sum"]
    sigma = math.sqrt(k2) / mu
    actual_mu = math.sqrt(k2) / sigma
    return {**audit, "adjacency": "add/remove-zero-out on fixed public slots; not replacement",
            "accountant": "GDP exact Gaussian adaptive composition; no Poisson accountant",
            "target_epsilon": epsilon, "delta": delta, "mu": actual_mu,
            "epsilon": epsilon_for_mu(actual_mu, delta), "clip": clip,
            "noise_multiplier": sigma, "noise_std_on_sum": clip * sigma,
            "noise_std_on_mean": clip * sigma / audit["logical_batch"],
            "scope": "one training run; public validation; public deduplication and N; independent unreleased noise seed",
            "excluded_releases": ["nonprivate runs", "raw training losses", "data audit", "hyperparameter search composition"]}


def compose_runs(audits, delta=1e-5):
    """Conservative same-person composition across all released DP fits."""
    mu = math.sqrt(sum(a["mu"] ** 2 for a in audits))
    return {"runs": len(audits), "mu": mu, "delta": delta, "epsilon": epsilon_for_mu(mu, delta),
            "bound": "all runs may involve the same individual; no cross-dataset disjointness assumed"}
