import json
from difficulty_classifier import classify_question

with open("test_questions.json") as f:
    test_set = json.load(f)

correct = 0
mismatches = []
for item in test_set:
    pred = classify_question(item["question"], use_llm=False)["difficulty"]
    if pred == item["label"]:
        correct += 1
    else:
        mismatches.append((item["question"], item["label"], pred))

accuracy = correct / len(test_set)
print(f"Accuracy: {accuracy:.2%} ({correct}/{len(test_set)})\n")

if mismatches:
    print("Misclassified:")
    for q, actual, pred in mismatches:
        print(f"  - \"{q}\"\n    Expected: {actual} | Got: {pred}")