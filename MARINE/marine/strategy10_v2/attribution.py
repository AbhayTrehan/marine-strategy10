"""
Sec 3.3 -- Visual Attribution and the Masking Region R(w).

Two-tier, applied IDENTICALLY to candidates and probes:

  1. Detector-based localization. If s_det(w) >= tau_box, R(w) := OWL-ViT's top
     box for "a photo of a w."  (handled in detector.py; consumed here)

  2. Attention-based fallback. Otherwise R(w) is built from the LVLM's own
     cross-modal attention at the teacher-forced token w_1, aggregated over a
     fixed layer/head set A (Eq. 5), keeping the smallest top-ranked patch set
     whose cumulative attention mass exceeds rho.

The fiddly part is tier 2: LLaVA's 576 image tokens live on a 24x24 grid over
the *preprocessed* image (CLIP-L/14: resize shortest-edge->336, then centre-crop
336x336). To mask the corresponding pixels of I -- and Sec 3.3 is explicit that
we mask I and re-encode it in full -- we must invert that resize+crop. That
inverse is implemented exactly (matching HF's get_resize_output_image_size and
center_crop integer arithmetic) rather than approximated.
"""

from typing import Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------- #
# preprocessing geometry: patch grid  ->  original image pixels
# --------------------------------------------------------------------------- #

def compute_geometry(orig_w: int, orig_h: int, image_processor) -> Dict:
    """Replicates the CLIP image processor's resize + centre-crop arithmetic."""
    do_resize = getattr(image_processor, "do_resize", True)
    size = getattr(image_processor, "size", None) or {}
    shortest_edge = size.get("shortest_edge")
    size_h, size_w = size.get("height"), size.get("width")

    do_center_crop = getattr(image_processor, "do_center_crop", False)
    crop_size = getattr(image_processor, "crop_size", None) or {}
    crop_h, crop_w = crop_size.get("height"), crop_size.get("width")

    # --- resize -----------------------------------------------------------
    if do_resize and shortest_edge is not None:
        # HF get_resize_output_image_size(..., default_to_square=False)
        if orig_w <= orig_h:
            new_w = shortest_edge
            new_h = int(shortest_edge * orig_h / orig_w)
        else:
            new_h = shortest_edge
            new_w = int(shortest_edge * orig_w / orig_h)
    elif do_resize and size_h is not None and size_w is not None:
        new_h, new_w = size_h, size_w
    else:
        new_h, new_w = orig_h, orig_w

    # --- centre crop ------------------------------------------------------
    if do_center_crop and crop_h is not None and crop_w is not None:
        ch, cw = crop_h, crop_w
    else:
        ch, cw = new_h, new_w

    top = (new_h - ch) // 2
    left = (new_w - cw) // 2

    return {
        "crop_w": cw,
        "crop_h": ch,
        "left": left,
        "top": top,
        "scale_x": new_w / float(orig_w),
        "scale_y": new_h / float(orig_h),
    }


def patches_to_boxes(patch_indices: Sequence[int], grid: int,
                     orig_w: int, orig_h: int, image_processor) -> List[List[float]]:
    """Map flat patch indices (row-major over the grid x grid map) back to
    boxes in ORIGINAL image pixel coordinates."""
    g = compute_geometry(orig_w, orig_h, image_processor)
    px_w = g["crop_w"] / float(grid)
    px_h = g["crop_h"] / float(grid)

    boxes = []
    for idx in patch_indices:
        r, c = divmod(int(idx), grid)
        # patch box in crop coords
        x0c, x1c = c * px_w, (c + 1) * px_w
        y0c, y1c = r * px_h, (r + 1) * px_h
        # crop coords -> resized coords
        x0r, x1r = x0c + g["left"], x1c + g["left"]
        y0r, y1r = y0c + g["top"], y1c + g["top"]
        # resized coords -> original coords
        x0 = x0r / g["scale_x"]
        x1 = x1r / g["scale_x"]
        y0 = y0r / g["scale_y"]
        y1 = y1r / g["scale_y"]

        x0 = max(0.0, min(x0, orig_w))
        x1 = max(0.0, min(x1, orig_w))
        y0 = max(0.0, min(y0, orig_h))
        y1 = max(0.0, min(y1, orig_h))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


# --------------------------------------------------------------------------- #
# locating the image tokens inside the LM sequence
# --------------------------------------------------------------------------- #

def image_token_positions(input_ids_row: torch.Tensor, seq_len: int,
                          image_token_index: int, n_image_tokens: int) -> torch.Tensor:
    """Positions of the image tokens in the language model's sequence.

    Handles both transformers regimes:
      * modern: the processor already expanded <image> into n_image_tokens
        placeholders, so seq_len == input_ids length;
      * legacy: input_ids carries a single <image> and the model expands it
        internally, so seq_len == len(input_ids) - 1 + n_image_tokens.
    """
    pos = (input_ids_row == image_token_index).nonzero(as_tuple=True)[0]
    n_in = int(input_ids_row.shape[0])

    if n_in == seq_len:
        if int(pos.numel()) == n_image_tokens:
            return pos
        raise RuntimeError(
            f"Expected {n_image_tokens} image tokens in input_ids, found {int(pos.numel())}."
        )

    if int(pos.numel()) == 1 and seq_len == n_in - 1 + n_image_tokens:
        start = int(pos[0].item())
        return torch.arange(start, start + n_image_tokens, device=input_ids_row.device)

    raise RuntimeError(
        f"Cannot locate image tokens: len(input_ids)={n_in}, model seq_len={seq_len}, "
        f"n_image_tokens={n_image_tokens}, n_image_token_ids_found={int(pos.numel())}."
    )


# --------------------------------------------------------------------------- #
# Eq. 5: a_j(w), and the top-rho attention-mass region
# --------------------------------------------------------------------------- #

def aggregate_attention(attentions, row_idx: int, img_pos: torch.Tensor,
                        layers: Sequence[int], heads: Sequence[int],
                        skip_cls: bool) -> torch.Tensor:
    """a_j(w) = mean over (layer, head) in A of Attn(w_1 -> patch_j).  Eq. (5).

    `attentions` is the HF tuple of per-layer tensors [B, H, S, S].
    `row_idx` is the sequence position of the teacher-forced token w_1.
    """
    n_layers = len(attentions)
    layers = [l for l in layers if 0 <= l < n_layers]
    if not layers:
        layers = [n_layers // 2]

    acc: Optional[torch.Tensor] = None
    n = 0
    for li in layers:
        att = attentions[li]                      # [1, H, S, S]
        n_heads = att.shape[1]
        head_ids = [h for h in heads if 0 <= h < n_heads] if heads else range(n_heads)
        for h in head_ids:
            v = att[0, h, row_idx, :].float()     # [S]
            acc = v[img_pos] if acc is None else acc + v[img_pos]
            n += 1

    a = acc / max(n, 1)
    if skip_cls:
        a = a[1:]                                 # drop the CLS image token
    return a                                      # [n_patches]


def top_mass_patches(a: torch.Tensor, rho: float, max_patch_frac: float) -> torch.Tensor:
    """Smallest top-ranked patch set whose cumulative attention mass exceeds rho.

    a is renormalised over the image patches first. The spec's rho is a mass
    threshold on the attention *to the image*; raw LLaMA attention rows also
    spend mass on the BOS/system/text tokens (the well-known attention sink), so
    thresholding raw mass would be a threshold on "how visual is this token",
    not "which patches support w". Renormalising makes rho mean the intended
    thing and keeps the region size comparable across words.
    """
    a = a.clamp(min=0)
    total = a.sum()
    n_patches = int(a.numel())

    if float(total) <= 0:
        return torch.arange(min(1, n_patches), device=a.device)

    p = a / total
    order = torch.argsort(p, descending=True)
    cum = torch.cumsum(p[order], dim=0)

    k = int((cum < rho).sum().item()) + 1
    k = max(1, min(k, max(1, int(max_patch_frac * n_patches))))
    return order[:k]
