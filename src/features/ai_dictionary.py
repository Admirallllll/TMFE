from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AITermSpec:
    key: str
    pattern: str


AI_TERM_SPECS: tuple[AITermSpec, ...] = (
    AITermSpec("artificial_intelligence", r"(?i)\bartificial\s+intelligence\b"),
    AITermSpec("ai", r"(?i)(?<![A-Za-z0-9])a\.?i\.?(?![A-Za-z0-9])"),
    AITermSpec("machine_learning", r"(?i)\bmachine\s+learning\b"),
    AITermSpec("ml", r"(?i)(?<![A-Za-z0-9])m\.?l\.?(?![A-Za-z0-9])"),
    AITermSpec("deep_learning", r"(?i)\bdeep\s+learning\b"),
    AITermSpec("neural_network", r"(?i)\bneural\s+network(?:s)?\b"),
    AITermSpec("generative_ai", r"(?i)\bgenerative\s+ai\b"),
    AITermSpec("genai", r"(?i)(?<![A-Za-z0-9])genai(?![A-Za-z0-9])"),
    AITermSpec("large_language_model", r"(?i)\blarge\s+language\s+model(?:s)?\b"),
    AITermSpec("llm", r"(?i)(?<![A-Za-z0-9])llm(?:s)?(?![A-Za-z0-9])"),
    AITermSpec("foundation_model", r"(?i)\bfoundation\s+model(?:s)?\b"),
    AITermSpec("transformer", r"(?i)\btransformer(?:s)?\b"),
    AITermSpec("chatgpt", r"(?i)\bchatgpt\b"),
    AITermSpec("copilot", r"(?i)\bcopilot(?:s)?\b"),
    AITermSpec("prompt", r"(?i)\bprompt(?:s|ing|ed)?\b"),
    AITermSpec("rag", r"(?i)(?<![A-Za-z0-9])rag(?![A-Za-z0-9])"),
    AITermSpec("retrieval_augmented", r"(?i)\bretrieval[- ]augmented\b"),
)


def ai_term_patterns() -> dict[str, str]:
    return {s.key: s.pattern for s in AI_TERM_SPECS}


def ai_topic_keyword_set() -> set[str]:
    return {
        "ai",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "ml",
        "deep",
        "neural",
        "network",
        "generative",
        "genai",
        "llm",
        "language",
        "model",
        "foundation",
        "transformer",
        "chatgpt",
        "copilot",
        "prompt",
        "rag",
        "retrieval",
        "augmented",
    }

