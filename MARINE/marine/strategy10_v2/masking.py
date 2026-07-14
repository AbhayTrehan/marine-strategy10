"""
Sec 3.3, masking operator:

    "I_masked(w) = Mask(I, R(w)) replaces the pixels in R(w) with the per-channel
     mean pixel value of I, and the modified image is re-encoded in full through
     the vision encoder."

That is what happens here -- with two hardenings that close a leak the previous
build had, and a measurement that proves the leak is closed.

THE LEAK
--------
LLaVA's vision tower does not see pixels; it sees 576 patch tokens, one per 14x14
cell of the preprocessed 336x336 crop. Painting a box's exact pixels grey leaves
two channels open:

  (1) PARTIAL PATCHES. A patch the box only half-covers is still a single token,
      and half of that token is still the object. The encoder reads it.
  (2) RESIZE BLEED. The mask is painted at original resolution, then the image is
      bicubically resized to 336. Interpolation pulls unmasked neighbouring
      pixels *into* the masked area near its edge, and pushes masked-edge content
      outward. The grey region that arrives at the ViT is not the grey region we
      painted.

Neither is visible by eye. Both put object signal inside tokens that are supposed
to contain none, which inflates l_masked, which shrinks Delta, which is the one
number the whole method turns on.

THE FIX
-------
  (1) is fixed in attribution.box_to_patches: the region is defined on the PATCH
      GRID, and every patch the box touches is taken in full.
  (2) is fixed here by `enforce_on_pixel_values`: after the processor has done its
      resize/crop/normalise, the masked patches are overwritten AGAIN, directly on
      the normalised pixel_values tensor, with the normalised fill colour. This
      runs before pixel_values is handed to the vision tower, so the tower
      provably cannot see anything else there.

Both are applied. The PIL-level mask is kept (not replaced) because it is what the
spec literally prescribes, because it is what the HTML report displays, and
because keeping it lets us MEASURE the leak: `measure_leak` reports how much
signal survived in the masked patches with PIL-masking alone. If the geometry were
wrong, that number would be large -- so it is a live check on the mapping, not a
comment claiming the mapping is right.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def mean_pixel(image: Image.Image) -> Tuple[int, int, int]:
    """Per-channel mean pixel value of I, per Sec 3.3."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return tuple(int(round(float(c))) for c in arr.reshape(-1, 3).mean(axis=0))


def apply_mask(image: Image.Image, boxes: Sequence[Sequence[float]],
               fill: Tuple[int, int, int]) -> Tuple[Image.Image, float]:
    """Paint `boxes` (already patch-aligned, in ORIGINAL pixels) with `fill`."""
    img = image.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    touched = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        c0, r0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c1, r1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if c1 > c0 and r1 > r0:
            touched[r0:r1, c0:c1] = True

    arr[touched] = np.array(fill, dtype=arr.dtype)
    area = float(touched.sum()) / float(h * w) if h * w else 0.0
    return Image.fromarray(arr), area


# --------------------------------------------------------------------------- #
# pixel_values-level enforcement + verification
# --------------------------------------------------------------------------- #

def normalised_fill(fill: Tuple[int, int, int], image_processor) -> List[float]:
    """The fill colour as the vision tower will see it, after CLIP normalisation.

    This is exactly the value the processor itself would produce for a region of
    constant `fill`, so overwriting with it is indistinguishable from the region
    having genuinely been that colour all along -- we are not injecting a novel
    value the encoder has never seen.
    """
    mean = list(getattr(image_processor, "image_mean", [0.0, 0.0, 0.0]))
    std = list(getattr(image_processor, "image_std", [1.0, 1.0, 1.0]))
    rescale = 1.0 / 255.0 if getattr(image_processor, "do_rescale", True) else 1.0
    do_norm = getattr(image_processor, "do_normalize", True)

    out = []
    for c in range(3):
        v = float(fill[c]) * rescale
        if do_norm:
            v = (v - float(mean[c])) / float(std[c])
        out.append(v)
    return out


def _patch_slices(patch_idx: int, grid: int, side: int):
    r, c = divmod(int(patch_idx), grid)
    p = side // grid
    return slice(r * p, (r + 1) * p), slice(c * p, (c + 1) * p)


def measure_leak(pixel_values: torch.Tensor, patches: Sequence[int],
                 grid: int, fill_norm: Sequence[float]) -> Dict[str, float]:
    """How much signal is STILL in the masked patches after PIL-only masking?

    Returns, over the masked patches only:
      max_abs_dev : the largest per-pixel deviation from the fill colour, in
                    normalised units. 0 == a perfectly flat masked region.
      mean_abs_dev: the average such deviation.

    A perfectly executed PIL mask, resized without interpolation error, would give
    0.0 for both. In practice bicubic resize leaves a thin rim of non-zero
    deviation at the mask boundary -- which is precisely the leak that
    `enforce_on_pixel_values` then removes. If these numbers were LARGE (order of
    the data's own scale, ~1-2), the patch geometry would be wrong and the mask
    would be landing somewhere other than the object.
    """
    if not len(patches):
        return {"max_abs_dev": 0.0, "mean_abs_dev": 0.0}

    side = int(pixel_values.shape[-1])
    fill = torch.tensor(fill_norm, dtype=pixel_values.dtype,
                        device=pixel_values.device).view(3, 1, 1)

    devs = []
    for p in patches:
        rs, cs = _patch_slices(p, grid, side)
        region = pixel_values[0, :, rs, cs]
        devs.append((region - fill).abs())
    d = torch.cat([x.reshape(-1) for x in devs])
    return {"max_abs_dev": float(d.max()), "mean_abs_dev": float(d.mean())}


def enforce_on_pixel_values(pixel_values: torch.Tensor, patches: Sequence[int],
                            grid: int, fill_norm: Sequence[float]) -> torch.Tensor:
    """Overwrite every masked patch with the fill colour, in normalised space.

    Runs AFTER the processor and BEFORE the vision tower. After this call the
    masked patches contain the fill colour and nothing else -- no partial-patch
    remnant, no interpolation bleed. The object cannot reach the encoder.
    """
    if not len(patches):
        return pixel_values

    side = int(pixel_values.shape[-1])
    fill = torch.tensor(fill_norm, dtype=pixel_values.dtype,
                        device=pixel_values.device).view(3, 1, 1)

    out = pixel_values.clone()
    for p in patches:
        rs, cs = _patch_slices(p, grid, side)
        out[0, :, rs, cs] = fill
    return out


def verify_enforced(pixel_values: torch.Tensor, patches: Sequence[int],
                    grid: int, fill_norm: Sequence[float], tol: float = 1e-4) -> bool:
    """Assert the masked patches really are flat. Cheap, so we just always run it."""
    if not len(patches):
        return True
    return measure_leak(pixel_values, patches, grid, fill_norm)["max_abs_dev"] <= tol


# --------------------------------------------------------------------------- #
# Controlled masking (--control_mask)
# --------------------------------------------------------------------------- #

def sample_control_patches(patches, grid: int, rng, max_tries: int = 200):
    """A patch set of the SAME SIZE AND SHAPE as R(w), placed somewhere else.

    WHY THIS EXISTS
    ---------------
    Delta(w) = l(w) - l_masked(w) conflates two effects:
        (i)  "the pixels that support w are gone"        <- what we want
        (ii) "a chunk of the image is gone"              <- pure nuisance
    (ii) scales with masked AREA. In the first real run one object masked 6.8% of the
    image and another masked 77%; their Deltas are not on the same scale, and no
    threshold can fix that.

    The control mask removes (ii) by construction. Mask an equally-large, equally-
    shaped region SOMEWHERE ELSE and ask:

        S(w) = l_control(w) - l_masked(w)

    "Does deleting w's own region hurt w MORE than deleting an equally big irrelevant
    region?" The area effect appears in both terms and cancels. This is the standard
    correction for deletion-based attribution metrics, which are known to be
    meaningless without a matched baseline.

    The control region is a rigid translation of R(w) -- same shape, same patch count
    -- so it also controls for the region's geometry, not just its size.
    """
    patches = sorted(set(patches))
    if not patches:
        return []

    rows = [p // grid for p in patches]
    cols = [p % grid for p in patches]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    rel = [(r - r0, c - c0) for r, c in zip(rows, cols)]
    orig = set(patches)

    if h > grid or w > grid:
        return []

    # Prefer a placement that does not overlap R(w) at all.
    for _ in range(max_tries):
        nr = rng.randrange(0, grid - h + 1)
        nc = rng.randrange(0, grid - w + 1)
        cand = {(nr + dr) * grid + (nc + dc) for dr, dc in rel}
        if not (cand & orig):
            return sorted(cand)

    # Densely-masked image: fall back to the placement with the least overlap.
    best, best_ov = None, None
    for _ in range(max_tries):
        nr = rng.randrange(0, grid - h + 1)
        nc = rng.randrange(0, grid - w + 1)
        cand = {(nr + dr) * grid + (nc + dc) for dr, dc in rel}
        ov = len(cand & orig)
        if best_ov is None or ov < best_ov:
            best, best_ov = cand, ov
    return sorted(best) if best else []


def complement_patches(patches, grid: int):
    """Every patch EXCEPT R(w). The insertion / sufficiency counterfactual.

    WHY DELETION ALONE IS NOT ENOUGH
    --------------------------------
    Sec 3.4 says a hallucination is "a mention driven by language priors or
    SCENE-LEVEL PLAUSIBILITY rather than the pixels of I". Deletion cannot detect
    that. When you delete a hallucinated object's region, the scene that invented it
    -- the rain, the crowd, the desk, the co-occurring objects -- is all still there.
    The model keeps believing, Delta ~ 0, and you have learned only that the model
    did not need those particular pixels. You have NOT learned that it was leaning on
    context, because you never took the context away.

    Insertion does exactly that: mask everything EXCEPT R(w), so the ONLY thing the
    model can see is the region that allegedly supports w.

        REAL object        -> its pixels are still there   -> belief survives  -> l_keep HIGH
        HALLUCINATED object-> R(w) holds nothing, and the context that invented
                              it is now gone as well       -> belief collapses -> l_keep LOW

    The contrast that follows is the score:

        G(w) = l_keep(w) - l_del(w)
             = "is the evidence INSIDE this region, or in the scene around it?"

    Note what G does NOT contain: l_full. That matters more than it looks. l_full is
    where candidates and probes stop being exchangeable -- candidates are words the
    model CHOSE to say (so l_full is high by construction), probes are words it did
    not (so l_full sits at the floor, with no room to fall). Any score of the form
    l_full - l_masked inherits that asymmetry. A contrast between two MASKED
    conditions cancels it, along with the word's unigram prior and the masked-area
    effect.
    """
    r = set(patches)
    return [p for p in range(grid * grid) if p not in r]
