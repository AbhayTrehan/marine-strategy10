"""
Sec 2: O = ExtractCanonicalObjects(y), with a surface-span pointer mu(o_i).

We do NOT reimplement this. The repo's own CHAIR evaluator
(eval/eval_chair.py) already contains, verbatim from Rohrbach et al. 2018:

  * CHAIR.caption_to_words(caption) -> (words, node_words, idxs, raw_words)
        - `words`      : surface forms found in the caption ("man", "dining table")
        - `node_words` : canonical MSCOCO category for each ("person", "dining table")
        - `idxs`       : token index of each mention == the surface span mu(o_i)
    This IS ExtractCanonicalObjects.

  * CHAIR.imid_to_objects[image_id] -> set of canonical COCO objects actually in
    the image (union of segmentation annotations and the 5 human captions).
    This is the ground truth the CHAIR benchmark itself scores against, so our
    "REAL vs HALLUCINATED" labels are by construction identical to CHAIR's.

  * CHAIR.mscoco_objects / inverse_synonym_dict -> the object vocabulary V and
    the synonym collapse used for probe sampling (Sec 4.1).

Loading it by file path (rather than `from eval.eval_chair import CHAIR`) keeps
us independent of cwd and of the fact that `eval/` has no __init__.py.
"""

import importlib.util
import os
import pickle
from typing import Dict, List, Set, Tuple


def _ensure_nltk():
    """caption_to_words -> nltk.word_tokenize -> needs the punkt tokenizer."""
    try:
        import nltk

        for pkg in ("punkt", "punkt_tab"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    except Exception:
        pass


def load_chair_module(marine_root: str):
    path = os.path.join(marine_root, "eval", "eval_chair.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find the repo's CHAIR evaluator at {path}")
    spec = importlib.util.spec_from_file_location("marine_eval_chair_s10v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_chair(marine_root: str, coco_annotations: str, cache_path: str):
    """Build (or load from cache) the CHAIR evaluator.

    Uses the SAME pickle cache path the repo's eval_chair.py main() uses, so if
    you have already run CHAIR evaluation this is instant.
    """
    _ensure_nltk()
    chair_mod = load_chair_module(marine_root)

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            evaluator = pickle.load(f)
        print(f"[extraction] loaded CHAIR evaluator from cache: {cache_path}")
    else:
        print("[extraction] building CHAIR evaluator from COCO annotations (one-off, ~1-2 min)...")
        evaluator = chair_mod.CHAIR(coco_annotations)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(evaluator, f)
            print(f"[extraction] cached CHAIR evaluator to: {cache_path}")

    return evaluator


def coco_categories(evaluator) -> List[str]:
    """The 80 canonical MSCOCO category names == probe vocabulary V (Sec 4.1)."""
    return sorted(set(evaluator.inverse_synonym_dict.values()))


def image_file_to_id(image_file: str) -> int:
    """COCO_val2014_000000144305.jpg -> 144305"""
    stem = os.path.basename(image_file).split(".")[0]
    return int(stem.split("_")[-1])


def extract_objects(evaluator, caption: str) -> List[Dict]:
    """O = ExtractCanonicalObjects(y).

    Returns one entry per DISTINCT canonical object (first mention wins, so the
    surface span mu(o_i) points at the first occurrence -- which is what the
    rewriter in Sec 5 would need).
    """
    words, node_words, idxs, _raw = evaluator.caption_to_words(caption)

    seen: Set[str] = set()
    objects: List[Dict] = []
    for surface, node, idx in zip(words, node_words, idxs):
        if node in seen:
            continue
        seen.add(node)
        objects.append({"object": node, "surface": surface, "span_idx": int(idx)})
    return objects


def ground_truth_objects(evaluator, image_id: int) -> Set[str]:
    """imid_to_objects is a defaultdict; use .get so we never insert a key."""
    gt = evaluator.imid_to_objects.get(image_id, set())
    return set(gt)


def label_object(node_word: str, gt: Set[str]) -> str:
    """CHAIR's own definition: a mention is hallucinated iff its canonical object
    is not in the image's ground-truth object set."""
    return "REAL" if node_word in gt else "HALLUCINATED"


def load_cooccurrence(path: str) -> Dict[str, List[str]]:
    """coco_co_occur.json: canonical category -> categories that most frequently
    co-occur with it, ranked most-frequent-first. Exactly the 'language-prior
    distractors' Sec 4.1 wants probes biased toward."""
    import json

    if not os.path.exists(path):
        print(f"[extraction] WARNING: co-occurrence file not found at {path}; "
              f"probe sampling will fall back to uniform.")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_questions(path: str, n: int) -> List[Tuple[str, int]]:
    """Take the first n images from the repo's CHAIR question list."""
    import json

    with open(path, "r") as f:
        data = json.load(f)
    out = []
    for item in data[:n]:
        img = item["image"]
        out.append((img, image_file_to_id(img)))
    return out
