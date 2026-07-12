"""
Loads the same LVLM this repo already uses (llava-hf/llava-1.5-7b-hf), with
deliberate deviations from marine/utils/utils_model.py:

  1. fp16, no low_cpu_mem_usage. utils_model.load_model() does
     `.from_pretrained(model_path).cuda()`, i.e. fp32 -> ~28 GB of VRAM.
     Strategy 10 (v2) has to co-reside with OWL-ViT and does thousands of
     forward passes, so fp16 is used here instead. `low_cpu_mem_usage=True`
     is deliberately NOT passed: it requires the `accelerate` package, which
     the rest of this repo's model loading does not depend on.

  2. attn_implementation="eager". The attention-based fallback of Sec 3.3 needs
     real attention weights; SDPA/FlashAttention do not return them.

  3. tokenizer via AutoTokenizer.from_pretrained(model_path, use_fast=False) --
     matching utils_model.py exactly, NOT AutoProcessor.from_pretrained. This
     was tried and reverted: AutoProcessor pulls in the FAST (Rust-backed,
     tokenizers-library) tokenizer by default, which reads the checkpoint's
     tokenizer.json. That file's on-disk schema is versioned by the
     `tokenizers` PyPI package and is NOT forward-compatible: a tokenizer.json
     written by a newer `tokenizers` release fails to load on an older one
     with "data did not match any variant of untagged enum ModelWrapper".
     use_fast=False selects the slow, pure-Python tokenizer, which reads
     vocab/merge files directly and never touches tokenizer.json at all --
     matching what the rest of this repo already does successfully.

  4. LlavaProcessor is still built with the correct patch_size /
     vision_feature_select_strategy / num_additional_image_tokens (see
     _build_processor below), so the <image> expansion bug described there is
     fixed WITHOUT needing AutoProcessor or any extra file downloads.
"""

import torch


def _build_processor(image_processor, tokenizer, model):
    """Construct LlavaProcessor with the fields that determine how many
    "<image>" placeholder tokens get inserted.

    On current transformers, LlavaProcessor computes:

        num_image_tokens = (h // patch_size) * (w // patch_size)
                           + num_additional_image_tokens
        if vision_feature_select_strategy == "default":
            num_image_tokens -= 1

    `num_additional_image_tokens` represents the vision tower's own CLS token
    (or equivalent): it is emitted for EVERY feature-select strategy and is
    only conditionally subtracted back out for "default". This is therefore a
    property of the vision tower architecture (standard CLIP-ViT: 1 CLS +
    patches), not of the strategy -- and is the same constant across
    strategies. Leaving it at its default of 0 (what happens if you construct
    LlavaProcessor by hand without setting it, as marine/utils/utils_model.py
    does) silently produces 575 expanded image tokens instead of the correct
    576 for llava-1.5-7b-hf's "default" strategy -- confirmed against a real
    LlavaProcessor instance, not just derived on paper.
    """
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
        # older transformers: LlavaProcessor doesn't accept these kwargs at
        # all, meaning it also doesn't try to expand <image> at construction
        # time -- the model expands it internally instead (the "legacy"
        # regime already handled by attribution.image_token_positions).
        return LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)


def load_lvlm(model_path: str, fp16: bool = True, device: str = "cuda"):
    from transformers import AutoImageProcessor, AutoTokenizer, LlavaForConditionalGeneration

    dtype = torch.float16 if fp16 else torch.float32

    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="eager", torch_dtype=dtype
        )
    except TypeError:
        # very old transformers without the attn_implementation kwarg -> eager
        # is the only implementation available anyway, so this is safe.
        model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)

    model = model.to(device)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    processor = _build_processor(image_processor, tokenizer, model)
    tokenizer = processor.tokenizer

    return model, tokenizer, processor


def describe_visual_grid(model):
    """Return (grid, n_image_tokens, skip_cls) for the LVLM's vision tower.

    For llava-1.5-7b-hf: CLIP-L/14 @336 -> grid=24, n_image_tokens=576,
    vision_feature_select_strategy='default' (CLS dropped) -> skip_cls=False.
    """
    vcfg = model.config.vision_config
    grid = vcfg.image_size // vcfg.patch_size
    n_patches = grid * grid

    strategy = getattr(model.config, "vision_feature_select_strategy", "default")
    if strategy == "full":
        # CLS token is kept as image token 0
        return grid, n_patches + 1, True
    return grid, n_patches, False
