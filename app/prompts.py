"""Prompt templates for description and styled caption generation."""

# Stage 1: Objective description prompt
DESCRIPTION_PROMPT: str = (
    "You are a video analysis assistant. Watch the provided video content carefully "
    "and produce a detailed, objective description of what happens in the video. "
    "Focus on factual observations: actions, objects, people, settings, and sequences of events. "
    "Remain neutral and avoid subjective language, humor, opinions, or stylistic flourishes. "
    "Your description should be thorough enough for someone to understand the video's content "
    "without seeing it."
)

# Stage 2: Style-specific caption prompts
STYLE_PROMPTS: dict[str, str] = {
    "formal": (
        "Write a formal caption for this video. No slang, no jokes — "
        "just objective, factual language. 1-3 sentences, self-contained."
    ),
    "sarcastic": (
        "Write a sarcastic caption for this video. Dry irony, understated "
        "mockery — never mean. 1-3 sentences, self-contained."
    ),
    "humorous_tech": (
        "Write a funny caption for this video using a tech or programming "
        "metaphor. 1-3 sentences, self-contained."
    ),
    "humorous_non_tech": (
        "Write a funny, relatable caption for this video. No jargon — "
        "keep it accessible. 1-3 sentences, self-contained."
    ),
}

# Appended to every style prompt. Keeps output machine-parseable: a single
# plain-text caption, no markdown, no alternatives, no meta-commentary.
OUTPUT_FORMAT_RULES: str = (
    "\n\nOutput rules (strict):\n"
    "- Return exactly ONE caption. Never offer multiple options, choices, or variants.\n"
    "- Plain text only. No markdown: no asterisks, no bold, no headers, no bullet points, "
    "no numbered lists.\n"
    "- Do not wrap the caption in quotation marks.\n"
    "- Do not use backslashes or escape characters.\n"
    "- Do not include any preamble, labels, or explanation (e.g. no \"Option 1:\", "
    "no \"Caption:\", no \"Here's a caption\").\n"
    "- Output the caption text and nothing else."
)


def get_caption_prompt(description: str, style: str) -> str:
    """Build the full caption prompt by combining the style instruction with the description."""
    if style not in STYLE_PROMPTS:
        raise ValueError(
            f"Unknown caption style: '{style}'. "
            f"Available styles: {list(STYLE_PROMPTS.keys())}"
        )

    style_instruction = STYLE_PROMPTS[style] + OUTPUT_FORMAT_RULES
    return f"{style_instruction}\n\nVideo description:\n{description}"