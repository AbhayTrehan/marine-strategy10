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


# --------------------------------------------------------------------------- #
# Existence elicitation (--scores delta_lo)
# --------------------------------------------------------------------------- #

EXISTENCE_QUESTION = "Is there a {word} in this image? Please answer yes or no."


def existence_prefix(word: str) -> str:
    """Prompt whose next token is the model's yes/no verdict on w's existence.

    WHY SCORE EXISTENCE RATHER THAN LIKELIHOOD
    ------------------------------------------
    Eq. (4)'s l(w) is a length-normalised log-likelihood of the WORD, so it is
    contaminated by how probable that word is a priori. In the first real run
    "laptop" scored l = -0.30 (p = 0.74) while "person" scored l = -9.08
    (p = 0.0001) -- on the SAME image. A word the model already half-expects has
    nats of room to fall when you occlude it; a word it barely believes has none. So
    Delta measures (visual grounding x prior confidence), and the second factor is
    pure nuisance that varies wildly across words.

    A yes/no log-ODDS cancels it:

        LO(w | I) = log p("Yes" | I, "Is there a {w}...?")
                  - log p("No"  | I, "Is there a {w}...?")

    This is a log odds-ratio of a BELIEF IN EXISTENCE, not a token likelihood. It is
    on the same scale for every word regardless of that word's unigram frequency, and
    LLaVA's well-documented yes-bias sits in both terms and cancels in the difference
    Delta_LO = LO(w|I) - LO(w|I_masked). Same cost: one forward per image variant.
    """
    return build_prompt(f"<image>\n{EXISTENCE_QUESTION.format(word=word)}")
