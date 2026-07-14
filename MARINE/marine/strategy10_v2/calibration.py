"""
Sec 4.3 -- Probe-Derived Threshold Calibration. This is the ONLY thing v2 changes
relative to v1.

    mu_hat    = (1/K) sum_p Delta(p)                                  (Eq. 8)
    sigma_hat = sqrt( (1/(K-1)) sum_p (Delta(p) - mu_hat)^2 )         (Eq. 8, Bessel)
    tau       = mu_hat + kappa * sigma_hat                            (Eq. 9)

    O_pos = { o : Delta(o) >= tau }                                   (Eq. 11)
    H     = O \\ O_pos                                                 (Eq. 12)

Both moments are recomputed from scratch per image, from that image's own probe
population, and discarded -- tau is per-image and self-calibrating.

Read Sec 4.3.2 before attaching meaning to a kappa. The arithmetic above is
assumption-free, but kappa carries NO false-verification-rate guarantee by
default: v1's conformal rank test got Pr[p <= eps] <= eps from exceedance
counting alone; a mean/sd threshold does not inherit that. This module therefore
also exposes the two OPTIONAL readings the spec permits, so the report can print
them clearly marked as assumption-laden rather than smuggling them in:

  * gaussian_tail(kappa)  -- 1 - Phi(kappa). Only valid if probe Delta is
    approximately normal, which is an empirical question. We ship a normality
    check (probe_normality) so the report can say whether the assumption holds
    on this data rather than assuming it.
  * cantelli_bound(kappa) -- 1/(1+kappa^2). Distribution-free but very loose
    (kappa=2 -> only <=20%).
"""

import math
from typing import Dict, List, Sequence, Tuple


def probe_moments(probe_deltas: Sequence[float]) -> Tuple[float, float]:
    """Eq. (8). Sample mean and (Bessel-corrected) sample sd."""
    K = len(probe_deltas)
    if K == 0:
        return float("nan"), float("nan")
    mu = sum(probe_deltas) / K
    if K < 2:
        return mu, 0.0
    var = sum((d - mu) ** 2 for d in probe_deltas) / (K - 1)
    return mu, math.sqrt(max(var, 0.0))


def threshold(mu: float, sigma: float, kappa: float) -> float:
    """Eq. (9)."""
    return mu + kappa * sigma


def partition(candidate_deltas: Dict[str, float], tau: float) -> Tuple[List[str], List[str]]:
    """Eq. (11)/(12).  verified = Delta >= tau ; flagged = the rest."""
    verified, flagged = [], []
    for obj, d in candidate_deltas.items():
        (verified if d >= tau else flagged).append(obj)
    return verified, flagged


# --------------------------------------------------------------------------- #
# Sec 4.3.2 -- the two OPTIONAL probability readings. Not invoked by default.
# --------------------------------------------------------------------------- #

def gaussian_tail(kappa: float) -> float:
    """1 - Phi(kappa). REQUIRES approximate normality of probe Delta."""
    return 0.5 * math.erfc(kappa / math.sqrt(2.0))


def cantelli_bound(kappa: float) -> float:
    """One-sided Chebyshev-Cantelli: Pr[Delta - mu >= kappa*sigma] <= 1/(1+kappa^2).
    Distribution-free, but only meaningful for kappa > 0."""
    if kappa <= 0:
        return 1.0
    return 1.0 / (1.0 + kappa ** 2)


def probe_normality(all_probe_deltas: Sequence[float]) -> Dict[str, float]:
    """Cheap empirical check of the normality approximation Sec 4.3.2 flags as
    'an empirical question' that 'should be checked ... before being relied
    upon'. We report skew and excess kurtosis of the pooled, per-image
    standardised probe Deltas -- a normal reference gives ~0 for both.
    """
    n = len(all_probe_deltas)
    if n < 3:
        return {"n": n, "skew": float("nan"), "excess_kurtosis": float("nan")}

    mu = sum(all_probe_deltas) / n
    var = sum((d - mu) ** 2 for d in all_probe_deltas) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return {"n": n, "skew": 0.0, "excess_kurtosis": 0.0}

    m3 = sum(((d - mu) / sd) ** 3 for d in all_probe_deltas) / n
    m4 = sum(((d - mu) / sd) ** 4 for d in all_probe_deltas) / n
    return {"n": n, "skew": m3, "excess_kurtosis": m4 - 3.0}


def apply_sigma_shrinkage(records, scores, lam: float) -> None:
    """Shrink each image's sigma_hat toward the pooled sigma across images.

    sigma_hat is estimated from only K (=20) probes, so its relative standard error is
    ~1/sqrt(2(K-1)) ~ 16%. tau = mu_hat + kappa*sigma_hat therefore inherits that noise:
    tau is itself a random variable, and an image that happened to draw a small
    sigma_hat gets a spuriously tight threshold. Shrinking toward the pooled estimate
    (empirical Bayes / James-Stein) trades a little bias for a large variance
    reduction, which is exactly the trade you want when the estimator is this noisy.

        sigma' = (1 - lam) * sigma_hat + lam * sigma_pooled

    lam = 0 recovers Eq. (8) exactly, so the default is spec-faithful and this is
    strictly opt-in.
    """
    if lam <= 0:
        return
    for sc in scores:
        sig = [r["null"][sc]["sigma"] for r in records
               if not r.get("skipped") and sc in (r.get("null") or {})
               and r["null"][sc]["sigma"] == r["null"][sc]["sigma"]]   # drop NaN
        if len(sig) < 2:
            continue
        pooled = sum(sig) / len(sig)
        for r in records:
            if r.get("skipped") or sc not in (r.get("null") or {}):
                continue
            n = r["null"][sc]
            n["sigma_raw"] = n["sigma"]
            n["sigma"] = (1.0 - lam) * n["sigma"] + lam * pooled
