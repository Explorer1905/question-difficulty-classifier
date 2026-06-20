"""
Question Difficulty Classifier
Classifies assessment questions into Easy / Medium / Hard.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from google import genai as genai_client

VALID_LABELS = {"Easy", "Medium", "Hard"}

BLOOM_LEVELS = {
    "easy": {
        "define", "list", "name", "state", "identify", "recall", "label",
        "what is", "who is", "when did", "where is", "match", "select", "recognize"
    },
    "medium": {
        "explain", "describe", "summarize", "classify", "compare", "apply",
        "calculate", "solve", "demonstrate", "illustrate", "interpret",
        "differentiate", "implement", "show how", "discuss", "outline"
    },
    "hard": {
        "analyze", "evaluate", "design", "synthesize", "justify", "critique",
        "derive", "prove", "argue", "construct", "formulate", "optimize",
        "assess", "propose", "investigate", "why does", "what would happen if"
    }
}

MULTI_TOPIC_CONNECTORS = {"and", "as well as", "in addition to", "along with", "combined with"}


@dataclass
class FeatureScores:
    word_count: int
    bloom_score: int          # -1 easy, 0 neutral, 1 medium, 2 hard
    multi_topic: bool
    has_subquestions: bool
    has_numeric_reasoning: bool
    technical_density: float


def extract_features(question: str) -> FeatureScores:
    text = question.strip()
    lower = text.lower()
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)

    bloom_score = 0
    for level, score in (("hard", 2), ("medium", 1), ("easy", -1)):
        if any(v in lower for v in BLOOM_LEVELS[level]):
            bloom_score = score
            break

    multi_topic = any(c in lower for c in MULTI_TOPIC_CONNECTORS) and word_count > 15
    has_subquestions = bool(re.search(r"\([a-zA-Z]\)|\bi+\)|\?.+\?", text))
    has_numeric_reasoning = bool(re.search(r"\d", text)) and bool(
        re.search(r"calculate|compute|solve|how many|how much|find the value", lower)
    )
    long_words = [w for w in words if len(w) > 7]
    technical_density = len(long_words) / max(word_count, 1)

    return FeatureScores(word_count, bloom_score, multi_topic,
                          has_subquestions, has_numeric_reasoning, technical_density)


def rule_based_difficulty(f: FeatureScores) -> str:
    """Fallback classifier — weighted heuristic scoring."""
    score = f.bloom_score * 2
    score += 2 if f.word_count > 40 else 1 if f.word_count > 20 else 0
    score += f.multi_topic + f.has_subquestions + f.has_numeric_reasoning
    score += f.technical_density > 0.25

    if score <= 0:
        return "Easy"
    elif score <= 3:
        return "Medium"
    return "Hard"


LLM_SYSTEM_PROMPT = """You are an expert assessment designer. Classify the given question's \
difficulty as exactly one of: Easy, Medium, Hard.

Guidelines:
- Easy: tests recall of a single fact or definition, no multi-step reasoning.
- Medium: requires understanding, application, or comparison of one or two concepts.
- Hard: requires multi-step reasoning, synthesis of multiple concepts, evaluation, or open-ended analysis.

If the question mixes multiple sub-topics, rate it by the most demanding sub-topic.
If the question is ambiguous or vague, rate it Medium.

Respond with ONLY this JSON on one line, nothing else:
{"difficulty": "Easy" | "Medium" | "Hard", "confidence": 0.0-1.0}
"""


def llm_classify(question: str, model_name: str = "gemini-2.5-flash") -> Optional[dict]:
    try:
        client = genai_client.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = client.models.generate_content(
            model=model_name,
            contents=f"Question: {question}",
            config={
                "system_instruction": LLM_SYSTEM_PROMPT,
                "temperature": 0.1,
                "max_output_tokens": 60,
            },
        )
        raw = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data if data.get("difficulty") in VALID_LABELS else None
    except Exception:
        return None


def classify_question(question: str, use_llm: bool = True) -> dict:
    """Main entry point. Returns {"difficulty": "Easy"|"Medium"|"Hard"}."""
    if not question or not question.strip():
        return {"difficulty": "Easy"}

    features = extract_features(question)
    rule_label = rule_based_difficulty(features)

    if use_llm:
        result = llm_classify(question)
        if result:
            order = {"Easy": 0, "Medium": 1, "Hard": 2}
            # If LLM and heuristics disagree by 2 levels, treat as ambiguous -> Medium
            if abs(order[result["difficulty"]] - order[rule_label]) >= 2:
                return {"difficulty": "Medium"}
            return {"difficulty": result["difficulty"]}

    return {"difficulty": rule_label}


if __name__ == "__main__":
    samples = [
        "What is the capital of France?",
        "Explain how photosynthesis converts light energy into chemical energy.",
        "Design an algorithm to optimize delivery routes for a fleet of trucks "
        "given dynamic traffic and fuel constraints, and justify your trade-offs.",
    ]
    for q in samples:
        print(q, "->", classify_question(q, use_llm=False))