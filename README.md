# Question Difficulty Classifier

Automatically classifies assessment questions into **Easy**, **Medium**, or **Hard**, as part of the Assessment Intelligence module.

## Overview

This classifier uses a hybrid approach:
- **Primary**: An LLM (Gemini) judges difficulty using a Bloom's Taxonomy-aligned prompt, with explicit instructions for handling ambiguous and multi-topic questions.
- **Fallback / sanity-check**: A rule-based heuristic scores questions using Bloom's verb level, word count, presence of sub-questions, multi-topic connectors, and numeric reasoning — used when no API key is available, or to flag cases where the LLM and heuristic disagree sharply.

## Output Format

```json
{"difficulty": "Medium"}
```

## Setup

```bash
pip install -r requirements.txt
```

To enable the LLM-based classifier, set your Gemini API key:
```bash
export GEMINI_API_KEY="your_key_here"      # macOS/Linux
$env:GEMINI_API_KEY="your_key_here"        # Windows PowerShell
```

## Usage

```python
from difficulty_classifier import classify_question

result = classify_question("Explain how photosynthesis works.")
print(result)  # {"difficulty": "Medium"}
```

Run the rule-based classifier without an API key:
```python
classify_question("What is the capital of France?", use_llm=False)
```

## Evaluation

```bash
python evaluate.py
```

**Result: 97.06% accuracy (33/34)** on the labeled test set in `test_questions.json`, exceeding the 80% target.

## Edge Case Handling

- **Ambiguous questions**: The LLM prompt explicitly defaults to Medium when a question is vague. As a safety net, if the LLM and rule-based heuristic disagree by 2 difficulty levels, the result is forced to Medium rather than trusting either signal blindly.
- **Multi-topic questions**: The LLM is instructed to rate by the most demanding sub-topic. The rule-based fallback detects topic-joining connectors ("and", "as well as", etc.) in longer questions and increases the difficulty score accordingly.

## Project Structure

```
question-difficulty-classifier/
├── difficulty_classifier.py   # Core classification logic
├── evaluate.py                 # Accuracy evaluation script
├── test_questions.json         # Labeled test set (34 questions)
├── requirements.txt
└── README.md
```

## Tech Stack

- Python
- Google Gemini API (`google-genai`)
- Rule-based NLP heuristics (Bloom's Taxonomy verb classification)
```
