"""Deterministic, answer-only prompts for the formal VQA evaluation.

The prompt contract is intentionally small.  Open questions request a short
answer and multiple-choice questions request one of A--D.  Neither template
asks the model to expose a rationale, so the decoded value can be passed
directly to :mod:`cross_model_fata.evaluation.output_parsing`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence


TaskType = Literal["open", "multiple_choice"]

OPEN_ANSWER_INSTRUCTION = "Respond with only a short answer. No explanation."
MC_ANSWER_INSTRUCTION = (
    "Respond with exactly one letter: A, B, C, or D. No explanation."
)


@dataclass(frozen=True)
class PromptSpec:
    """One image-first chat prompt and its deterministic decode budget."""

    task_type: TaskType
    question_text: str
    messages: tuple[dict[str, Any], ...]
    max_new_tokens: int


def _clean_question(question: str) -> str:
    text = str(question).strip()
    if not text:
        raise ValueError("question must not be empty")
    return text


def _append_options(question: str, options: Sequence[str]) -> str:
    """Append options only when the dataset question does not contain them."""

    if not options or "options:" in question.lower():
        return question
    if not 2 <= len(options) <= 4:
        raise ValueError("multiple-choice options must contain two to four entries")
    rows = [f"{chr(ord('A') + index)}. {str(option).strip()}" for index, option in enumerate(options)]
    return f"{question}\nOptions:\n" + "\n".join(rows)


def answer_only_question(
    task_type: TaskType,
    question: str,
    *,
    options: Sequence[str] = (),
) -> str:
    """Return the exact text portion placed after the image placeholder."""

    clean = _clean_question(question)
    if task_type == "open":
        if options:
            raise ValueError("open questions must not supply multiple-choice options")
        return f"{clean}\n{OPEN_ANSWER_INSTRUCTION}"
    if task_type == "multiple_choice":
        clean = _append_options(clean, options)
        return f"{clean}\n{MC_ANSWER_INSTRUCTION}"
    raise ValueError(f"unsupported task_type {task_type!r}")


def build_prompt_spec(
    task_type: TaskType,
    question: str,
    *,
    options: Sequence[str] = (),
    open_max_new_tokens: int = 16,
    multiple_choice_max_new_tokens: int = 4,
) -> PromptSpec:
    """Build a single-user, image-first InternVL chat message."""

    if open_max_new_tokens < 1 or multiple_choice_max_new_tokens < 1:
        raise ValueError("decode budgets must be positive")
    question_text = answer_only_question(task_type, question, options=options)
    messages = (
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question_text},
            ],
        },
    )
    max_new_tokens = (
        open_max_new_tokens if task_type == "open" else multiple_choice_max_new_tokens
    )
    return PromptSpec(task_type, question_text, messages, max_new_tokens)


def render_prompt(processor: Any, spec: PromptSpec) -> str:
    """Render the model's own chat template without loading media or tokenizing."""

    return processor.apply_chat_template(
        list(spec.messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_prompt_inputs(
    processor: Any,
    image: Any,
    spec: PromptSpec,
    *,
    crop_to_patches: bool = True,
    return_tensors: str = "pt",
    **processor_kwargs: Any,
) -> Any:
    """Render then process one image/question while preserving placeholder parity."""

    prompt = render_prompt(processor, spec)
    return processor(
        images=image,
        text=prompt,
        crop_to_patches=crop_to_patches,
        return_tensors=return_tensors,
        **processor_kwargs,
    )


__all__ = [
    "MC_ANSWER_INSTRUCTION",
    "OPEN_ANSWER_INSTRUCTION",
    "PromptSpec",
    "TaskType",
    "answer_only_question",
    "build_prompt_spec",
    "prepare_prompt_inputs",
    "render_prompt",
]
