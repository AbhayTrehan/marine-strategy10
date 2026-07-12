"""
Loads the same LVLM this repo already uses (llava-hf/llava-1.5-7b-hf), with two
deliberate deviations from marine/utils/utils_model.py:

  1. fp16 + low_cpu_mem_usage. utils_model.load_model() does
     `.from_pretrained(model_path).cuda()`, i.e. fp32 -> ~28 GB of VRAM.
     Strategy 10 (v2) has to co-resident OWL-ViT alongside the LVLM, and it does
     thousands of forward passes, so fp16 is loaded here instead. (We do NOT
     edit utils_model.py -- minimal changes outside this package.)

  2. attn_implementation="eager". The attention-based fallback of Sec 3.3 needs
     real attention weights; SDPA/FlashAttention do not return them.

Everything else (LlavaForConditionalGeneration + LlavaProcessor built from
AutoImageProcessor + slow AutoTokenizer) mirrors utils_model.py exactly, so the
model/tokenizer/processor behaviour is identical to the rest of the repo.
"""

import torch


def load_lvlm(model_path: str, fp16: bool = True, device: str = "cuda"):
    from transformers import (
        AutoImageProcessor,
        AutoTokenizer,
        LlavaForConditionalGeneration,
        LlavaProcessor,
    )

    dtype = torch.float16 if fp16 else torch.float32

    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, attn_implementation="eager", **kwargs
        )
    except TypeError:
        # very old transformers without attn_implementation kwarg -> eager is the default
        model = LlavaForConditionalGeneration.from_pretrained(model_path, **kwargs)

    model = model.to(device)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    # utils_model.py builds LlavaProcessor(image_processor, tokenizer) with nothing
    # else. On transformers >= ~4.44 the processor only expands <image> into the 576
    # image-token placeholders if it knows patch_size + the feature-select strategy;
    # without them it silently leaves a single <image> and the MODEL expands
    # internally. Both regimes are handled downstream (scoring.py aligns from the end
    # of the sequence, attribution.image_token_positions detects which regime it is
    # in), but we pass the values when the installed version accepts them so the
    # common case is the unambiguous one.
    vcfg = model.config.vision_config
    try:
        processor = LlavaProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=vcfg.patch_size,
            vision_feature_select_strategy=getattr(
                model.config, "vision_feature_select_strategy", "default"
            ),
        )
    except TypeError:
        # older transformers: LlavaProcessor takes only (image_processor, tokenizer)
        processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

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
