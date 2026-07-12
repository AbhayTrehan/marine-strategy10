"""
Loads the same LVLM this repo already uses (llava-hf/llava-1.5-7b-hf), with
deliberate deviations from marine/utils/utils_model.py:

  1. fp16, no low_cpu_mem_usage. utils_model.load_model() does
     `.from_pretrained(model_path).cuda()`, i.e. fp32 -> ~28 GB of VRAM.
     Strategy 10 (v2) has to co-reside with OWL-ViT and does thousands of
     forward passes, so fp16 is used here instead. `low_cpu_mem_usage=True`
     is deliberately NOT passed: it requires the `accelerate` package, which
     the rest of this repo's model loading does not depend on, so requiring
     it here would be an unnecessary new dependency for a feature (staged
     low-RAM loading) that only matters during model construction, not
     inference.

  2. attn_implementation="eager". The attention-based fallback of Sec 3.3 needs
     real attention weights; SDPA/FlashAttention do not return them.

  3. Processor via AutoProcessor.from_pretrained(model_path), NOT manually
     reconstructed from AutoImageProcessor + AutoTokenizer the way
     utils_model.py does. This matters more than it looks: on current
     transformers, LlavaProcessor expands "<image>" using

         num_image_tokens = (h // patch_size) * (w // patch_size)
                            + num_additional_image_tokens
         if vision_feature_select_strategy == "default":
             num_image_tokens -= 1

     `num_additional_image_tokens` is NOT something we can safely default --
     it exists precisely to account for whatever the checkpoint's vision
     tower prepends (typically a CLS token) before feature selection drops
     it, and its correct value is saved in the checkpoint's own
     processor_config.json, not derivable from the model config alone.
     Manually constructing LlavaProcessor(image_processor, tokenizer, ...)
     without it (as utils_model.py does, and as an earlier version of this
     function did) silently defaults it to 0, which for llava-1.5-7b-hf
     produces 575 expanded image tokens instead of the correct 576 -- off by
     one, no error raised, every downstream alignment silently corrupted.
     AutoProcessor.from_pretrained loads the checkpoint's actual saved
     config, so this can't drift regardless of transformers version.
"""

import torch


def load_lvlm(model_path: str, fp16: bool = True, device: str = "cuda"):
    from transformers import AutoProcessor, LlavaForConditionalGeneration

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

    processor = AutoProcessor.from_pretrained(model_path)
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
