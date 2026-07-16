"""
Loads the LVLM this repo already uses (llava-hf/llava-1.5-7b-hf).

Deviations from marine/utils/utils_model.py, all deliberate:

  1. fp16. utils_model.load_model() does `.from_pretrained(model_path).cuda()`,
     i.e. fp32 -> ~28 GB of VRAM. Strategy 10 (v2) has to co-reside with
     GroundingDINO and runs thousands of forward passes, so fp16 it is.
     `low_cpu_mem_usage=True` is deliberately NOT passed: it requires the
     `accelerate` package, which nothing else in this repo's model loading needs.

  2. The slow tokenizer (use_fast=False), matching utils_model.py exactly. The
     fast tokenizer reads tokenizer.json, whose on-disk schema is versioned by
     the `tokenizers` package and is not forward-compatible -- a tokenizer.json
     written by a newer release fails to load on an older one. The slow tokenizer
     reads vocab files directly and cannot hit that.

  3. num_additional_image_tokens=1 when constructing LlavaProcessor. Current
     transformers computes the <image> expansion as
         (h//patch)*(w//patch) + num_additional_image_tokens
         minus 1 if vision_feature_select_strategy == "default"
     The additive term is the vision tower's CLS token: emitted always, subtracted
     back out only for "default". Leaving it at its 0 default (which is what
     utils_model.py does) yields 575 image tokens for llava-1.5-7b-hf instead of
     576 -- off by one, no error raised, and every downstream alignment silently
     corrupted.

  4. NO attn_implementation="eager". The previous build forced eager because the
     attention-based fallback of Sec 3.3 needed real attention weights. That
     fallback is gone (GroundingDINO now localises every word), so we no longer
     request attentions at all, and the default (SDPA) kernel is both faster and
     lighter.
"""

import torch


def _build_processor(image_processor, tokenizer, model):
    from transformers import LlavaProcessor

    vcfg = model.config.vision_config
    strategy = getattr(model.config, "vision_feature_select_strategy", "default")

    try:
        return LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=vcfg.patch_size,
            vision_feature_select_strategy=strategy,
            num_additional_image_tokens=1,
        )
    except TypeError:
        # Older transformers: LlavaProcessor takes none of these, and therefore
        # does not expand <image> at all -- the model expands it internally
        # instead. scoring.py aligns from the end of the sequence, so both
        # regimes are handled.
        return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_lvlm(model_path: str, fp16: bool = True, device: str = "cuda"):
    from transformers import (AutoImageProcessor, AutoTokenizer,
                              LlavaForConditionalGeneration)

    dtype = torch.float16 if fp16 else torch.float32

    # attn_implementation="sdpa" is REQUIRED for attention occlusion to work.
    # The ViT occlusion is realised as an additive attention mask injected into the
    # vision tower's self-attention. SDPA and eager both honour an additive mask;
    # flash-attention-2 does NOT (it supports only causal/no mask) and would silently
    # ignore it, leaving the ViT half of the occlusion inert -- the object then leaks
    # into the LVLM through neighbouring patch tokens and the causal test is gutted
    # with no error. So we pin sdpa here rather than leave it to the version-dependent
    # default. (The AttentionOcclusion constructor also self-checks this at runtime.)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation="sdpa"
    )
    model = model.to(device)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    processor = _build_processor(image_processor, tokenizer, model)

    return model, processor.tokenizer, processor


def describe_visual_grid(model):
    """(grid, n_image_tokens) for the LVLM's vision tower.

    llava-1.5-7b-hf: CLIP-L/14 @336 -> grid=24 -> 576 patch tokens.
    """
    vcfg = model.config.vision_config
    grid = vcfg.image_size // vcfg.patch_size
    n_patches = grid * grid

    strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    if strategy == "full":
        return grid, n_patches + 1
    return grid, n_patches