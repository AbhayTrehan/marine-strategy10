"""
Geometry between three coordinate frames:

    (a) ORIGINAL image pixels        -- what GroundingDINO's box is in
    (b) the PREPROCESSED 336x336 crop -- what CLIP-L/14 actually receives
    (c) the 24x24 PATCH GRID          -- what the ViT actually tokenises

Frame (c) is the one that matters, and the reason the previous build leaked.
LLaVA does not see pixels; it sees 576 patch tokens. Masking a box's exact
pixels leaves every partially-covered patch still carrying a slice of the object
straight into the encoder. So the region has to be defined in frame (c): every
patch the box touches, in full.

The mapping (a) -> (b) is CLIP's resize-shortest-edge-to-336 followed by a
336x336 centre crop. It is replicated here exactly (matching HF's
get_resize_output_image_size and center_crop integer arithmetic) rather than
approximated -- an off-by-a-few-pixels error here would put the mask on the
wrong patches, which is undetectable by eye but fatal to Delta.

Note that the centre crop DISCARDS part of the image. An object sitting in the
cropped-away margin is not visible to the LVLM at all, so it has no patches, so
masking it is a no-op and its Delta is necessarily 0. That is a real and correct
outcome, not a bug -- but it must be surfaced, so box_to_patches reports it.
"""

from typing import Dict, List, Sequence, Tuple


def compute_geometry(orig_w: int, orig_h: int, image_processor) -> Dict:
    """Replicates the CLIP image processor's resize + centre-crop arithmetic."""
    do_resize = getattr(image_processor, "do_resize", True)
    size = getattr(image_processor, "size", None) or {}
    shortest_edge = size.get("shortest_edge")
    size_h, size_w = size.get("height"), size.get("width")

    do_center_crop = getattr(image_processor, "do_center_crop", False)
    crop_size = getattr(image_processor, "crop_size", None) or {}
    crop_h, crop_w = crop_size.get("height"), crop_size.get("width")

    # --- resize ------------------------------------------------------------
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

    # --- centre crop --------------------------------------------------------
    if do_center_crop and crop_h is not None and crop_w is not None:
        ch, cw = crop_h, crop_w
    else:
        ch, cw = new_h, new_w

    top = (new_h - ch) // 2
    left = (new_w - cw) // 2

    return {
        "crop_w": cw, "crop_h": ch,
        "left": left, "top": top,
        "scale_x": new_w / float(orig_w),
        "scale_y": new_h / float(orig_h),
    }


def orig_box_to_crop_box(box: Sequence[float], g: Dict) -> Tuple[float, float, float, float]:
    """ORIGINAL pixel box -> coordinates inside the 336x336 crop (may fall outside)."""
    x0, y0, x1, y1 = box
    return (
        x0 * g["scale_x"] - g["left"],
        y0 * g["scale_y"] - g["top"],
        x1 * g["scale_x"] - g["left"],
        y1 * g["scale_y"] - g["top"],
    )


def box_to_patches(box: Sequence[float], grid: int, orig_w: int, orig_h: int,
                   image_processor) -> Tuple[List[int], Dict]:
    """Every patch index the box touches, plus diagnostics.

    "Touches" is deliberately inclusive: a patch that the box overlaps by even
    one pixel is masked IN FULL. Anything less would leave part of the object
    inside a token the ViT reads.
    """
    g = compute_geometry(orig_w, orig_h, image_processor)
    px_w = g["crop_w"] / float(grid)
    px_h = g["crop_h"] / float(grid)

    x0c, y0c, x1c, y1c = orig_box_to_crop_box(box, g)

    # clip the box to the visible crop
    vx0, vy0 = max(0.0, x0c), max(0.0, y0c)
    vx1, vy1 = min(float(g["crop_w"]), x1c), min(float(g["crop_h"]), y1c)

    info = {
        "outside_crop": not (vx1 > vx0 and vy1 > vy0),
        "clipped_by_crop": (x0c < 0 or y0c < 0
                            or x1c > g["crop_w"] or y1c > g["crop_h"]),
    }
    if info["outside_crop"]:
        info.update({"n_patches": 0, "patch_frac": 0.0})
        return [], info

    c0 = max(0, int(vx0 // px_w))
    r0 = max(0, int(vy0 // px_h))
    # ceil on the far edge: any patch the box reaches into is included
    c1 = min(grid, int(-(-vx1 // px_w)))
    r1 = min(grid, int(-(-vy1 // px_h)))
    c1 = max(c1, c0 + 1)
    r1 = max(r1, r0 + 1)

    patches = [r * grid + c for r in range(r0, r1) for c in range(c0, c1)]
    info.update({
        "n_patches": len(patches),
        "patch_frac": len(patches) / float(grid * grid),
        "patch_rect": [r0, r1, c0, c1],
    })
    return patches, info


def patches_to_orig_boxes(patch_indices: Sequence[int], grid: int,
                          orig_w: int, orig_h: int, image_processor) -> List[List[float]]:
    """Patch indices -> boxes in ORIGINAL image pixels.

    Used to (a) paint the mask onto the PIL image and (b) draw the actual masked
    region in the HTML report, so what the report shows is what the model saw.
    """
    g = compute_geometry(orig_w, orig_h, image_processor)
    px_w = g["crop_w"] / float(grid)
    px_h = g["crop_h"] / float(grid)

    boxes = []
    for idx in patch_indices:
        r, c = divmod(int(idx), grid)
        x0 = (c * px_w + g["left"]) / g["scale_x"]
        x1 = ((c + 1) * px_w + g["left"]) / g["scale_x"]
        y0 = (r * px_h + g["top"]) / g["scale_y"]
        y1 = ((r + 1) * px_h + g["top"]) / g["scale_y"]

        x0, x1 = max(0.0, min(x0, orig_w)), max(0.0, min(x1, orig_w))
        y0, y1 = max(0.0, min(y0, orig_h)), max(0.0, min(y1, orig_h))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def patches_bounding_box(patch_indices: Sequence[int], grid: int,
                         orig_w: int, orig_h: int, image_processor):
    """Single rect in ORIGINAL pixels enclosing all the given patches."""
    boxes = patches_to_orig_boxes(patch_indices, grid, orig_w, orig_h, image_processor)
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]
