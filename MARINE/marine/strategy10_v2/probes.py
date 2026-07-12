"""
Sec 4.1 -- Probe Sampling.

    "A probe set P = {p_1,...,p_K} of words believed absent from I is sampled
     from a fixed object vocabulary V: excluding O and its synonyms / hypernyms /
     hyponyms, excluding words with s_det(w) > tau_low, and biasing sampling
     toward words that co-occur frequently with objects in O (the language-prior
     distractors that make the null informative rather than trivial)."

Three design decisions, all forced by the repo:

1. V = the 80 canonical MSCOCO categories. Working in *canonical* (node-word)
   space means the "excluding O and its synonyms/hypernyms/hyponyms" clause is
   satisfied automatically: CHAIR's inverse_synonym_dict has already collapsed
   "man"/"lady"/"biker" -> "person", so excluding node_words(O) excludes every
   synonym of every mentioned object by construction.

2. The co-occurrence bias uses data/org_qa/pope/coco/coco_co_occur.json, which
   maps each COCO category to the categories that most frequently co-occur with
   it, ranked. This is precisely the "language-prior distractor" signal the spec
   asks for -- a probe like "fork" next to a mentioned "dining table" is a hard
   null, whereas "giraffe" is a trivial one.

3. Probes are chosen WITHOUT looking at ground truth. Using GT to guarantee
   absence would make the null model unattainable at inference time. The honest
   filter is the detector one (s_det <= tau_low). We do, however, *measure* how
   often a sampled probe turns out to be GT-present and report it -- probe
   contamination inflates mu_hat and is the first thing to check if the
   calibration misbehaves.
"""

import random
from typing import Dict, List, Sequence, Set, Tuple


def sample_probes(
    vocabulary: Sequence[str],
    mentioned: Set[str],
    det_scores: Dict[str, float],
    cooccur: Dict[str, List[str]],
    K: int,
    tau_low: float,
    cooccur_bias: float,
    rng: random.Random,
) -> Tuple[List[str], Dict]:
    """Returns (probes, info)."""
    # --- exclude O (and, via canonicalisation, all its synonyms) -------------
    pool = [v for v in vocabulary if v not in mentioned]

    # --- exclude words the detector thinks are present -----------------------
    absent = [v for v in pool if det_scores.get(v, 0.0) <= tau_low]

    relaxed = False
    if len(absent) < K:
        # Safety valve: rather than silently shrinking K (which would change the
        # variance of sigma_hat across images), fall back to the K lowest-s_det
        # words in the pool and flag it.
        relaxed = True
        absent = sorted(pool, key=lambda v: det_scores.get(v, 0.0))[: max(K, len(absent))]

    # --- bias toward language-prior distractors -----------------------------
    weights = []
    for v in absent:
        w = 1.0
        for o in mentioned:
            neighbours = cooccur.get(o, [])
            if v in neighbours:
                rank = neighbours.index(v)
                w += cooccur_bias / (1.0 + rank)
        weights.append(w)

    probes: List[str] = []
    candidates = list(absent)
    wts = list(weights)
    k = min(K, len(candidates))
    for _ in range(k):
        total = sum(wts)
        if total <= 0:
            pick = rng.randrange(len(candidates))
        else:
            r = rng.random() * total
            acc = 0.0
            pick = len(candidates) - 1
            for i, w in enumerate(wts):
                acc += w
                if r <= acc:
                    pick = i
                    break
        probes.append(candidates.pop(pick))
        wts.pop(pick)

    info = {
        "pool_size": len(pool),
        "absent_pool_size": len(absent),
        "relaxed_tau_low": relaxed,
        "n_probes": len(probes),
    }
    return probes, info
