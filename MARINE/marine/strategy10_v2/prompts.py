"""
Prompt construction for Strategy 10 (v2).

Two prompts are needed:

  * the Stage 1 task prompt x  (Sec 2)  -- "Generate a short caption of the image."
  * the fixed elicitation template c_elicit (Sec 3.1), under which EVERY word --
    candidate or probe -- is re-elicited and teacher-forced.

The spec writes c_elicit abstractly as

    Question: What objects are visible in this image?\n Answer: This image contains a

For LLaVA-1.5 we must express that inside the model's own chat template
(vicuna_v1), otherwise the model is being scored off-distribution. The faithful
adaptation is:

    {SYSTEM} USER: <image>\nWhat objects are visible in this image? ASSISTANT: This image contains a

and then the subword tokens of w are teacher-forced as the continuation.

This is task-independent and identical for candidates and probes, which is the
only property Sec 3.1 actually requires ("procedural symmetry").
"""

from .config import (
    VICUNA_V1_SYSTEM,
    ELICIT_QUESTION,
    ELICIT_ANSWER_PREFIX,
)


def _build_prompt_fallback(user_message: str) -> str:
    """Hand-rolled vicuna_v1 prompt, byte-identical to llava.conversation's
    SeparatorStyle.TWO rendering with an empty assistant turn."""
    return f"{VICUNA_V1_SYSTEM} USER: {user_message} ASSISTANT:"


def _build_prompt_llava(user_message: str) -> str:
    """Preferred path: use the exact conv_templates object the rest of this repo
    uses (marine/utils/utils_dataset.py), so our prompt cannot silently drift
    from the repo's."""
    from llava.conversation import conv_templates  # noqa: WPS433

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], user_message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_prompt(user_message: str) -> str:
    try:
        return _build_prompt_llava(user_message)
    except Exception:  # llava repo not on PYTHONPATH -- fall back
        return _build_prompt_fallback(user_message)


def task_prompt(task: str) -> str:
    """Stage 1: y = M_theta(. | x, I)."""
    return build_prompt(f"<image>\n{task}")


def elicitation_prefix() -> str:
    """c_elicit -- the prefix that every candidate/probe word is forced onto.

    Returns a string ENDING in 'This image contains a' (no trailing space); the
    word is appended as ' {w}' by the scorer, so the LLaMA tokenizer sees the
    natural leading-space subword split.
    """
    prompt = build_prompt(f"<image>\n{ELICIT_QUESTION}")
    return f"{prompt} {ELICIT_ANSWER_PREFIX}"
