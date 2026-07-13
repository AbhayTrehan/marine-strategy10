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

Imported as `eval.eval_chair` (a PEP 420 namespace package, since eval/ has no
__init__.py) rather than via `from eval.eval_chair import CHAIR` directly, so
this module works regardless of the caller's cwd. See load_chair_module() for
why it matters that this go through a real import rather than a hand-loaded
module object.
"""

import importlib
import os
import pickle
import sys
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
    """Import the repo's own eval/eval_chair.py as `eval.eval_chair`.

    eval/ has no __init__.py, so this relies on PEP 420 namespace packages
    (Python >= 3.3): any directory on sys.path becomes an importable package
    even without an __init__.py. Putting `marine_root` on sys.path makes
    `import eval.eval_chair` work exactly like importing any other module.

    This matters for pickling: pickle.dump on a CHAIR instance records its
    class as "eval.eval_chair.CHAIR" and, when *loading* that pickle later
    (possibly in a different process), the pickler re-imports that dotted path
    via the REAL import system to verify the class exists there. An earlier
    version of this function loaded eval_chair.py via
    importlib.util.spec_from_file_location() with a synthetic module name and
    manually inserted it into sys.modules. That is enough to satisfy the
    pure-Python `pickle` module's __import__ check within the SAME process,
    but the C-accelerated `_pickle` module (which is what `import pickle`
    actually gives you by default) does its own import validation during
    save_global, and a module that only exists because someone hand-inserted
    it into sys.modules -- rather than one the real finders (sys.path
    scanning) can locate -- is not guaranteed to satisfy that check. Going
    through a real `import eval.eval_chair` sidesteps the issue entirely:
    the module is genuinely discoverable via sys.path in any process, so both
    the pure-Python and the C pickler can import it reliably at dump time
    AND at load time.
    """
    path = os.path.join(marine_root, "eval", "eval_chair.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find the repo's CHAIR evaluator at {path}")

    if marine_root not in sys.path:
        sys.path.insert(0, marine_root)  # front of sys.path: shadow any same-named package

    return importlib.import_module("eval.eval_chair")


def build_chair_for_images(marine_root: str, coco_annotations: str,
                           images: List[Tuple[str, int]], cache_path: str = None):
    """Build a CHAIR-compatible evaluator, with ground truth for ONLY the given
    images -- not the repo's full ~40K-image, ~617K-caption COCO corpus.

    `images`: the (image_file, image_id) list load_questions() returns.

    WHY THIS EXISTS: eval/eval_chair.py's CHAIR.__init__() calls
    get_annotations(), which loads BOTH instances_train2014.json and
    instances_val2014.json (~900K segmentation annotations total) and BOTH
    captions_train2014.json and captions_val2014.json (~617K captions total),
    and -- this is the actual cost -- runs NLTK tokenisation + TextBlob
    singularisation on EVERY ONE of those ~617K captions. JSON parsing itself
    is fast; that per-caption NLP call is what takes the many minutes you were
    seeing, and it happens in FULL on every run regardless of how many images
    this sanity check actually scores, because build_chair() builds ground
    truth for the entire dataset up front.

    We only ever need ground truth for the N images this run actually looks
    at. Every image in data/org_qa/chair/coco_chair.json is a COCO_val2014_*
    image, so this function:
        1. never opens *_train2014.json at all,
        2. filters val2014's annotations/captions down to `images` BEFORE
           calling caption_to_words(), so NLP runs on ~5*N captions instead
           of ~617,000.

    It reuses the REAL CHAIR.caption_to_words() and the same
    inverse_synonym_dict / double_word_dict the full build uses -- both are
    set up in __init__ before get_annotations() is ever called -- so the
    ground-truth labels this produces are IDENTICAL to what the slow,
    full-corpus build would produce for these same images. This is strictly a
    speed optimisation, not a different definition of ground truth.

    Raises ValueError if any requested image is not a *_val2014_* image (the
    fast path can't serve it); use build_chair_full_corpus() for that case.
    """
    import json
    from collections import defaultdict

    non_val = [f for f, _ in images if "val2014" not in f]
    if non_val:
        raise ValueError(
            f"build_chair_for_images() only has annotations loaded for "
            f"*_val2014_* images, but {len(non_val)} requested image(s) are "
            f"not val2014 (e.g. {non_val[0]!r}). Use build_chair_full_corpus() "
            f"instead if you need train2014 images too (much slower)."
        )

    target_ids = [imid for _, imid in images]
    cache_key = tuple(sorted(set(target_ids)))

    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
            if (payload.get("image_ids") == cache_key
                    and payload.get("coco_annotations") == coco_annotations):
                print(f"[extraction] loaded targeted ground truth from cache: "
                      f"{cache_path}  ({len(cache_key)} images)")
                ev = payload["evaluator"]
                # A cached evaluator carries the OLD double_word_dict. Re-install
                # the compound rules or a warm cache silently resurrects the
                # phantom "player"->PERSON / "baby"->PERSON mentions.
                n = install_compound_rules(ev)
                print(f"[extraction] re-installed {n} compound-noun rules onto the cached evaluator")
                return ev
            print(f"[extraction] cache at {cache_path} is for a different image "
                  f"set / annotations path -- rebuilding (fast; see below).")
        except Exception as exc:
            print(f"[extraction] cache at {cache_path} could not be loaded "
                  f"({type(exc).__name__}: {exc}) -- rebuilding.")

    _ensure_nltk()
    chair_mod = load_chair_module(marine_root)
    chair_cls = chair_mod.CHAIR

    print(f"[extraction] building TARGETED ground truth for {len(cache_key)} "
          f"images (val2014 only, skips the ~617K-caption full-corpus build; "
          f"should take a few seconds)...")

    # __init__ does two things: (a) cheap in-memory synonym/double-word setup,
    # (b) self.get_annotations() -- the expensive full-corpus pass. Stub out
    # (b) for the duration of construction so we get (a) for free and nothing
    # else.
    orig_get_annotations = chair_cls.get_annotations
    chair_cls.get_annotations = lambda self: None
    try:
        evaluator = chair_cls(coco_annotations)
    finally:
        chair_cls.get_annotations = orig_get_annotations

    n_rules = install_compound_rules(evaluator)
    print(f"[extraction] installed {n_rules} compound-noun rules "
          f"(suppresses 'CD/DVD player'->PERSON, 'baby carriage'->PERSON, ...)")

    target_set = set(target_ids)

    inst_path = os.path.join(coco_annotations, "instances_val2014.json")
    if not os.path.exists(inst_path):
        raise FileNotFoundError(
            f"Missing {inst_path} -- please download MSCOCO instance "
            f"annotations for the val set (see README)."
        )
    with open(inst_path) as f:
        inst = json.load(f)
    id_to_name = {cat["id"]: cat["name"] for cat in inst["categories"]}
    for ann in inst["annotations"]:
        imid = ann["image_id"]
        if imid not in target_set:
            continue
        node_word = evaluator.inverse_synonym_dict[id_to_name[ann["category_id"]]]
        evaluator.imid_to_objects[imid].append(node_word)

    cap_path = os.path.join(coco_annotations, "captions_val2014.json")
    if not os.path.exists(cap_path):
        raise FileNotFoundError(
            f"Missing {cap_path} -- please download MSCOCO caption "
            f"annotations for the val set (see README)."
        )
    with open(cap_path) as f:
        caps = json.load(f)
    for ann in caps["annotations"]:
        imid = ann["image_id"]
        if imid not in target_set:
            continue
        _, node_words, _, _ = evaluator.caption_to_words(ann["caption"])
        evaluator.imid_to_objects[imid].extend(node_words)

    # same dedup step as the repo's own get_annotations()
    for imid in list(evaluator.imid_to_objects):
        evaluator.imid_to_objects[imid] = set(evaluator.imid_to_objects[imid])

    if cache_path:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"evaluator": evaluator, "image_ids": cache_key,
                        "coco_annotations": coco_annotations}, f)
        print(f"[extraction] cached targeted ground truth to: {cache_path}")

    return evaluator


def build_chair_full_corpus(marine_root: str, coco_annotations: str, cache_path: str):
    """The repo's OWN full-corpus CHAIR build (both train+val, all ~617K
    captions). NOT used by default -- build_chair_for_images() above is what
    the sanity-check script actually calls, and is what you want unless you
    have a specific reason to need ground truth for images outside your
    current run (e.g. building one cache to reuse across many differently
    sized sanity-check runs). This takes many minutes; see that function's
    docstring for why.

    CAVEAT: pickle records a class's location as "<module>.<ClassName>". If the
    cache on disk was written by running eval/eval_chair.py DIRECTLY (`python
    eval/eval_chair.py ...`), CHAIR was pickled as "__main__.CHAIR" -- and
    __main__ here is THIS script, which has no CHAIR attribute, so unpickling
    fails. We treat that as a stale/incompatible cache rather than a fatal
    error: warn, ignore it, and rebuild.
    """
    _ensure_nltk()
    chair_mod = load_chair_module(marine_root)

    evaluator = None
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                evaluator = pickle.load(f)
            print(f"[extraction] loaded CHAIR evaluator from cache: {cache_path}")
        except Exception as exc:
            print(f"[extraction] cache at {cache_path} could not be loaded "
                  f"({type(exc).__name__}: {exc}).")
            print("[extraction] this usually means the cache was pickled by a "
                  "different entry-point script (e.g. eval/eval_chair.py run "
                  "directly) -- ignoring it and rebuilding.")
            evaluator = None

    if evaluator is None:
        print("[extraction] building CHAIR evaluator from the FULL COCO corpus "
              "(train+val, ~617K captions -- this can take many minutes)...")
        evaluator = chair_mod.CHAIR(coco_annotations)
        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(evaluator, f)
            print(f"[extraction] cached CHAIR evaluator to: {cache_path}")

    return evaluator



# --------------------------------------------------------------------------- #
# Compound-noun rules -- fixing CHAIR's false object mentions
# --------------------------------------------------------------------------- #
# CHAIR resolves objects by WORD-level synonym lookup, and its person-synonym list
# literally contains "player" and "baby" (they are there for "a player kicks the
# ball" / "a baby sleeps"). So a caption saying
#
#     "a laptop computer with a CD/DVD player"   -> player -> PERSON
#     "a doll sitting in a baby carriage"        -> baby   -> PERSON
#
# invents a person that was never mentioned. Those phantom mentions then go through
# the whole occlusion pipeline and are scored against ground truth, so they poison
# the metrics at the source: no amount of fixing the masking can rescue an object
# that should never have been extracted.
#
# CHAIR already has the machinery for exactly this -- `double_word_dict`, which is
# how it stops "hot dog" firing DOG and "baby bird" firing PERSON. It simply lacks
# the entries. So we EXTEND that dict rather than reimplement caption_to_words:
# a bigram mapped to NON_OBJECT collapses to a token that is not in mscoco_objects,
# and CHAIR's own filter then drops it. Nothing in eval_chair.py is modified.
#
# Crucially these rules are narrow. "baseball player" / "tennis player" still fire
# PERSON, because they are people; only DEVICE players are suppressed. Likewise
# "baby" is only suppressed in front of a thing a baby cannot be.

NON_OBJECT = "__s10v2_nonobject__"

# "<device> player" is an appliance, not a person.
_MEDIA_PLAYER_PREFIXES = [
    "cd", "dvd", "cd/dvd", "dvd/cd", "vcr", "dvd/vcr", "blu-ray", "bluray",
    "record", "media", "music", "mp3", "mp4", "video", "disc", "cassette", "tape",
]

# "baby <thing>" where the thing is not something a baby can be.
_BABY_NON_PERSON = [
    "carriage", "stroller", "buggy", "pram", "bottle", "monitor", "crib", "cot",
    "seat", "gate", "blanket", "wipe", "food", "formula", "powder", "shampoo",
    "doll", "toy", "shower", "clothes", "shoe", "sock",
]

# CHAIR maps "computer" -> LAPTOP and "monitor" -> TV, so these compounds double-fire.
_COMPUTER_COMPOUNDS = {
    "computer mouse": "mouse",
    "computer keyboard": "keyboard",
    "computer monitor": "tv",
    "computer screen": "tv",
    "computer tower": NON_OBJECT,
    "computer desk": "dining table",
}

_MISC_COMPOUNDS = {
    "water pitcher": NON_OBJECT,   # "pitcher" is a person-synonym (baseball)
    "orange juice": NON_OBJECT,    # "orange" the colour/flavour, not the fruit
    "teddy": NON_OBJECT,           # bare "teddy" without "bear" is not a COCO object
}


def install_compound_rules(evaluator) -> int:
    """Extend the evaluator's double_word_dict in place. Returns rules added.

    MUST be called on every evaluator, including one restored from the pickle
    cache -- a cached evaluator carries the OLD dict, so skipping this on the cache
    path would silently reintroduce the phantom mentions on any run that hits a
    warm cache.
    """
    d = evaluator.double_word_dict
    before = len(d)

    for p in _MEDIA_PLAYER_PREFIXES:
        d[f"{p} player"] = NON_OBJECT
    for x in _BABY_NON_PERSON:
        d[f"baby {x}"] = NON_OBJECT
    d.update(_COMPUTER_COMPOUNDS)
    d.update(_MISC_COMPOUNDS)

    return len(d) - before


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


# --------------------------------------------------------------------------- #
# Extended vocabulary: RAM++ tags
# --------------------------------------------------------------------------- #
#
# THE HARD CONSTRAINT, STATED PLAINLY
# -----------------------------------
# Ground truth is CHAIR.imid_to_objects, which contains ONLY the 80 MSCOCO
# categories. So an object can be scored REAL vs HALLUCINATED **only if it is one of
# those 80**. That is not a limitation of this code; it is a limitation of what COCO
# annotates.
#
# Two consequences, both of which the pipeline now handles explicitly rather than
# silently:
#
#   1. "desk" -> "dining table" is NOT a parser bug. COCO genuinely annotates desks
#      under `dining table`, so for GT purposes the mapping is CORRECT. It looks
#      wrong; it is the taxonomy, not the code. The report now prints the
#      surface -> canonical mapping for every mention so these collapses are VISIBLE
#      instead of silent.
#
#   2. Mentions with no COCO class at all -- "headphones", "doll", "carriage" -- used
#      to be DROPPED, i.e. never tested, even though they are plausibly the most
#      hallucination-prone things the model says. They are now EXTRACTED and scored
#      (you can see Delta for "headphones" in the report), but they carry
#      gt_label = "UNKNOWN" and are EXCLUDED from catch-rate / false-flag / AUROC,
#      because there is no ground truth to score them against. Including them in the
#      metrics would be inventing labels.
#
# Expanding the CANDIDATE vocabulary therefore expands what you can SEE, not what you
# can MEASURE. Expanding the PROBE vocabulary is different -- probes need no ground
# truth at all -- so --probe_vocab ram is a straightforwardly good idea.

import json as _json
import os as _os

_RAM_STOPWORDS = {
    # RAM++ tags are not all nouns; drop the obvious verbs/abstractions so they do
    # not become probes or candidates.
    "adjust", "attach", "approach", "balance", "take", "stand", "sit", "walk", "run",
    "hold", "look", "sew", "watch", "play", "ride", "eat", "drink", "smile", "wear",
    "cut", "cook", "throw", "catch", "jump", "fly", "swim", "read", "write", "talk",
    "art", "area", "accident", "back", "front", "side", "top", "bottom", "view",
    "photo", "picture", "image", "scene", "background", "foreground", "color",
    "light", "shadow", "reflection", "pattern", "texture", "material", "shape",
    "group", "pair", "row", "line", "pile", "stack", "collection", "set", "part",
}


def load_ram_vocabulary(path: str) -> List[str]:
    """The RAM++ tag vocabulary shipped with this repo
    (data/marine_qa/guidance/coco_ram_th0.68.json): ~1,180 distinct tags."""
    if not _os.path.exists(path):
        print(f"[extraction] RAM++ tag file not found at {path}; falling back to COCO-80.")
        return []
    with open(path) as f:
        data = _json.load(f)

    tags = set()
    for item in data:
        for t in item.get("objects", []):
            t = str(t).strip().lower()
            if t and len(t) < 30 and t not in _RAM_STOPWORDS:
                tags.add(t)
    return sorted(tags)


def extract_objects_extended(evaluator, caption: str, extra_vocab=None) -> List[Dict]:
    """O = ExtractCanonicalObjects(y), plus mentions that have no COCO class.

    COCO mentions come from CHAIR exactly as before (so they stay bit-identical to
    what the CHAIR benchmark scores). On top of that, any tag from `extra_vocab`
    appearing in the response is surfaced as a NON-COCO candidate:
    scorable by the pipeline, visible in the report, but gt_label = UNKNOWN and
    excluded from every metric.
    """
    objects = extract_objects(evaluator, caption)
    if not extra_vocab:
        return objects

    seen_surface = {o["surface"].lower() for o in objects}
    seen_canon = {o["object"] for o in objects}
    text = " " + caption.lower() + " "

    extra = []
    # longest tags first so "baby carriage" wins over "carriage"
    for tag in sorted(extra_vocab, key=len, reverse=True):
        if tag in seen_canon or tag in seen_surface:
            continue
        if f" {tag} " in text or f" {tag}s " in text or f" {tag}, " in text or f" {tag}." in text:
            if any(tag in e["object"] or e["object"] in tag for e in extra):
                continue                      # don't double-count nested tags
            extra.append({
                "object": tag,
                "surface": tag,
                "span_idx": -1,
                "in_coco": False,
            })

    for o in objects:
        o["in_coco"] = True
    return objects + extra
