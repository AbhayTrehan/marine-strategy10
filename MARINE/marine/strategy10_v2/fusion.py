"""
The Causal Existence framework.

WHAT THE FIRST FULL RUN ACTUALLY SHOWED
---------------------------------------
    s_det          0.859   <- a raw detector score, no LVLM at all
    delta          0.792   <- the spec's deletion test
    delta_ctrl     0.737
    delta_lo       0.727
    delta_lo_ins   0.698
    delta_lo_ctrl  0.684
    delta_ins      0.658   <- the "improvement" that was supposed to win

Insertion came LAST. Its separation of the means was actually LARGER than delta's
(1.478 vs 1.390) -- the means moved the right way -- but its variance exploded, so
AUROC collapsed. Two reasons, and both are instructive:

  1. l_keep is read off a ~90%-grey image: severely out of distribution, and its value
     is dominated by HOW BIG R(w) is (a 6% region leaves a 94% grey canvas; a 77%
     region leaves the image nearly intact). Deletion and insertion therefore carry
     the masked-AREA confound with OPPOSITE signs, and subtracting them AMPLIFIES it.

  2. delta_ins and delta_ctrl both drop l_full -- and l_full carries real signal, not
     just the selection bias I accused it of. delta_ctrl's separation fell from 1.39
     to 1.13 when l_full was removed.

THE MEASUREMENT THAT WAS MISSING
--------------------------------
Every score so far is a linear combination of three numbers: l_full, l_del, l_keep.
That is why the algebra kept collapsing back onto the same quantity. The one never
measured is the LANGUAGE PRIOR: the model's belief in w with the image ENTIRELY
masked. Which is exactly what Sec 3.4 says a hallucination is -- "driven by language
priors or scene-level plausibility rather than the pixels".

With a blank-image pass, existence log-odds DECOMPOSE:

    PRIOR(w) = LO_blank                 pure language prior, no image at all
    SUF(w)   = LO_keep  - LO_blank      evidence from the REGION, above the prior
    CTX(w)   = LO_del   - LO_blank      evidence from the SCENE,  above the prior
    NEC(w)   = LO_full  - LO_del        necessity            (this is delta_lo)

CTX is the new, never-measured signal, and it points the OTHER WAY:

    hallucinated -> deleting R(w) changes nothing, so LO_del stays high while
                    LO_blank is only the prior  ->  CTX LARGE
    real         -> deleting R(w) removes the object, LO_del collapses
                                                ->  CTX SMALL or NEGATIVE

An anti-correlated feature is exactly what a fusion needs. And SUF = LO_keep -
LO_blank is the RIGHT sufficiency contrast: it compares two heavily-masked images to
each other, instead of pitting a 90%-grey image against a 10%-grey one, which is what
made delta_ins so noisy.

THE FUSION
----------
Standardise every feature against THAT IMAGE'S OWN probe null -- which is nothing more
than Eq. (8) applied to a vector instead of a scalar -- and combine with fixed signs:

    z_f(w) = (f(w) - mu_f) / sigma_f          mu_f, sigma_f from the K probes
    CES(w) = z(NEC) + z(SUF) - z(CTX)

High necessity, high sufficiency, LOW context-dependence == grounded. No labels, no
fitting, no training: the probes supply mu and sigma exactly as they already do for
tau. The combination is a hypothesis about SIGNS, not a learned weight vector.
"""

from typing import Dict, List, Optional

from . import calibration

# Raw per-word features. Everything downstream is built from these.
#   *_lo  = existence log-odds head  (causal EXISTENCE)
#   bare  = length-normalised likelihood head (the spec's l)
FEATURES = [
    # likelihood head
    "nec", "suf", "ctx", "ctrl", "prior",
    # existence head
    "nec_lo", "suf_lo", "ctx_lo", "ctrl_lo", "prior_lo",
]

# Fusions. Sign convention throughout: HIGHER == MORE GROUNDED == more likely REAL.
FUSIONS: Dict[str, Dict[str, float]] = {
    # Pure causal existence -- no detector, no likelihood head. This is the one that
    # answers "does the causal machinery work on its own?"
    "ces": {"nec_lo": 1.0, "suf_lo": 1.0, "ctx_lo": -1.0},

    # The same decomposition on the likelihood head, for comparison.
    "ces_lik": {"nec": 1.0, "suf": 1.0, "ctx": -1.0},

    # Both heads.
    "ces_both": {"nec_lo": 1.0, "suf_lo": 1.0, "ctx_lo": -1.0,
                 "nec": 1.0, "suf": 1.0, "ctx": -1.0},

    # Causal existence + the detector. If THIS does not beat s_det alone, the LVLM's
    # occlusion response is adding nothing the detector did not already know.
    "ces_det": {"nec_lo": 1.0, "suf_lo": 1.0, "ctx_lo": -1.0, "s_det": 1.0},

    # Everything.
    "fuse_all": {"nec_lo": 1.0, "suf_lo": 1.0, "ctx_lo": -1.0, "ctrl_lo": 1.0,
                 "nec": 1.0, "suf": 1.0, "ctx": -1.0, "ctrl": 1.0, "s_det": 1.0},
}


def compute_features(row: Dict) -> Dict[str, float]:
    """The intervention decomposition, from the raw LO / l measurements on one word."""
    f: Dict[str, float] = {}

    def put(name, a, b):
        if a is not None and b is not None:
            f[name] = a - b

    # likelihood head
    put("nec", row.get("ell"), row.get("ell_masked"))          # = delta (Eq. 7)
    put("suf", row.get("ell_keep"), row.get("ell_blank"))      # region above the prior
    put("ctx", row.get("ell_masked"), row.get("ell_blank"))    # scene above the prior
    put("ctrl", row.get("ell_ctrl"), row.get("ell_masked"))    # area-controlled
    if row.get("ell_blank") is not None:
        f["prior"] = row["ell_blank"]

    # existence head -- CAUSAL EXISTENCE
    put("nec_lo", row.get("lo"), row.get("lo_masked"))
    put("suf_lo", row.get("lo_keep"), row.get("lo_blank"))
    put("ctx_lo", row.get("lo_masked"), row.get("lo_blank"))
    put("ctrl_lo", row.get("lo_ctrl"), row.get("lo_masked"))
    if row.get("lo_blank") is not None:
        f["prior_lo"] = row["lo_blank"]

    return f


def fit_null(probe_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    """mu and sigma for every feature, from THIS IMAGE'S probes. Eq. (8), vectorised."""
    null: Dict[str, Dict[str, float]] = {}
    keys = set()
    for p in probe_rows:
        keys.update(p.get("features", {}))
    keys.add("s_det")

    for k in sorted(keys):
        vals = [(p["s_det"] if k == "s_det" else p.get("features", {}).get(k))
                for p in probe_rows]
        vals = [v for v in vals if v is not None and v == v]
        if len(vals) < 2:
            continue
        mu, sd = calibration.probe_moments(vals)
        null[k] = {"mu": mu, "sigma": sd}
    return null


def zscore(row: Dict, null: Dict, key: str) -> Optional[float]:
    """Standardise one feature against the probe null. sigma == 0 -> the feature is
    constant across probes and carries no usable scale, so it is dropped rather than
    dividing by zero."""
    n = null.get(key)
    if n is None or not n["sigma"] or n["sigma"] != n["sigma"]:
        return None
    v = row["s_det"] if key == "s_det" else row.get("features", {}).get(key)
    if v is None or v != v:
        return None
    return (v - n["mu"]) / n["sigma"]


def apply_fusions(rows: List[Dict], null: Dict) -> None:
    """Write every fusion into row["scores"], in place.

    A fusion is only emitted if EVERY one of its components was available for that
    word, so a fusion is never silently computed from a subset of its terms.
    """
    for row in rows:
        for name, weights in FUSIONS.items():
            zs = {k: zscore(row, null, k) for k in weights}
            if any(v is None for v in zs.values()):
                continue
            row["scores"][name] = sum(w * zs[k] for k, w in weights.items())
