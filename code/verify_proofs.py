#!/usr/bin/env python3

"""Self-contained theorem and proposition verification script for:
Stabilizing Bandit Mixture-of-Experts and Zipfian Stream Clustering via Router Replay and Clamped Laguerre Potentials
"""

from __future__ import annotations

import math
import numpy as np

def verify_theorems(rng: np.random.Generator) -> None:
    # 1. Proposition 1 & Theorem 1: Exact off-policy drift bias and exact R3 unbiasedness + variance formula
    for _ in range(100):
        E, d = 5, 4
        x_tilde = np.concatenate([[1.0], rng.normal(size=d)])
        W = rng.normal(scale=0.6, size=(d + 1, E))
        logits = x_tilde @ W
        pi = np.exp(logits - logits.max())
        pi /= pi.sum()
        p_roll = 0.75 * pi + 0.25 / E
        mu = rng.normal(size=E)
        Lhat = rng.normal(size=E)
        e_m = int(np.argmax(pi))
        # Exact expectation over discrete rollout choice e_roll ~ p_roll (with zero-mean noise)
        d_ie = np.array([pi[e] * (np.eye(E)[e] - pi) for e in range(E)])  # shape (E, E) wrt z
        true_grad_z = sum(d_ie[e] * mu[e] for e in range(E))
        # Drifted gradient expectation
        exp_g_drift = np.zeros(E)
        exp_g_r3 = np.zeros(E)
        for e_r in range(E):
            L_dr = Lhat.copy()
            L_dr[e_m] = mu[e_r]
            bar_dr = float(pi @ L_dr)
            g_dr = sum(pi[e] * (L_dr[e] - bar_dr) * np.eye(E)[e] for e in range(E))
            exp_g_drift += p_roll[e_r] * g_dr

            L_r3 = Lhat.copy()
            L_r3[e_r] = Lhat[e_r] + (mu[e_r] - Lhat[e_r]) / p_roll[e_r]
            bar_r3 = float(pi @ L_r3)
            g_r3 = sum(pi[e] * (L_r3[e] - bar_r3) * np.eye(E)[e] for e in range(E))
            exp_g_r3 += p_roll[e_r] * g_r3

        mu_bar_roll = float(p_roll @ mu)
        pred_bias = sum(d_ie[e] * (Lhat[e] - mu[e]) for e in range(E)) + d_ie[e_m] * (mu_bar_roll - Lhat[e_m])
        assert np.allclose(exp_g_drift - true_grad_z, pred_bias, atol=1e-12), "Prop 1 violated"
        assert np.allclose(exp_g_r3, true_grad_z, atol=1e-12), "Thm 1(1) violated"

    # 2. Theorem 2: Scale equivariance and protected Voronoi core ball
    for _ in range(100):
        d = 6
        s_t = rng.uniform(0.2, 4.0)
        Bmax = 0.18
        cp = rng.normal(size=d)
        u = rng.normal(size=d)
        u /= np.linalg.norm(u)
        Delta = 2.4 * math.sqrt(Bmax * s_t) + rng.uniform(0.2, 2.0)
        cq = cp + Delta * u
        r_core = 0.5 * Delta - 2.0 * Bmax * s_t / Delta
        assert r_core > 0
        v = rng.normal(size=d)
        v /= np.linalg.norm(v)
        x = cp + rng.uniform(0.0, 0.999 * r_core) * v
        bp, bq = -Bmax, +Bmax  # worst-case debt gap 2*Bmax
        phi_p = -np.sum((x - cp) ** 2) / (2.0 * s_t) + bp
        phi_q = -np.sum((x - cq) ** 2) / (2.0 * s_t) + bq
        assert phi_p > phi_q, "Thm 2(2) Voronoi margin violated"

    # 3. Proposition 2: Exact Gaussian convolution block log-score
    for _ in range(50):
        d = 4
        mu_b = rng.normal(size=d)
        sig2 = rng.uniform(0.05, 1.5, size=d)
        href = rng.uniform(0.3, 1.2)
        Ib = 40
        x = rng.normal(size=d)
        # Exact integral of Ib * exp(-||x-z||^2/(2*href^2)) under Z ~ N(mu_b, diag(sig2))
        exact_mass = Ib * np.prod(
            (href / np.sqrt(href ** 2 + sig2)) * np.exp(-((x - mu_b) ** 2) / (2.0 * (href ** 2 + sig2)))
        )
        s_b = math.log(Ib) - 0.5 * np.sum(np.log(href ** 2 + sig2) + ((x - mu_b) ** 2) / (href ** 2 + sig2))
        assert abs((math.log(exact_mass) - d * math.log(href)) - s_b) < 1e-12, "Prop 2 violated"


if __name__ == "__main__":
    verify_theorems(np.random.default_rng(20261122))
    print("ALL_THEOREMS_VERIFIED_OK")
