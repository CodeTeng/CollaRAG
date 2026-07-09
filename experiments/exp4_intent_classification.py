"""实验 4：意图分类准确率（Intent Classification Accuracy）

评估 TreeIntentClassifier 在 430 条人工标注集上的分类性能。

对应文档：docs/experiments/intent-classification-accuracy.md

Usage:
    uv run python experiments/exp4_intent_classification.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.utils import (
    PROJECT_ROOT,
    add_common_args,
    render_markdown_table,
    save_results,
    setup_experiment,
)

logger = logging.getLogger("chimera.exp4")

EXP_NAME = "exp4_intent_classification"

DATASET_INTENT_MAP = {
    "nq": "single_hop",
    "popqa": "single_hop",
    "two_wiki": "multi_hop",
    "hotpotqa": "multi_hop",
    "musique": "multi_hop",
    "asqa": "summarization",
}

SAMPLES_PER_DATASET = 60

OTHER_QUERIES = [
    "What is the weather like today in Tokyo?",
    "Calculate 127 times 83",
    "What time is it in New York?",
    "How do I cook pasta?",
    "Tell me a joke",
    "What is 2 + 2?",
    "How to install Python on Windows?",
    "What is the meaning of life?",
    "Convert 100 USD to EUR",
    "How tall is a giraffe in meters?",
    "What programming language should I learn first?",
    "How many calories in a banana?",
    "What is the speed of light?",
    "How to fix a flat tire?",
    "What is the chemical formula for water?",
    "How do I reset my password?",
    "What is the population of Mars?",
    "How to play chess?",
    "What day is Christmas?",
    "How to make coffee?",
    "What is the largest prime number?",
    "How to tie a necktie?",
    "What is the boiling point of alcohol?",
    "How many legs does a spider have?",
    "What is the capital of the Moon?",
    "How to say hello in Japanese?",
    "What is the distance from Earth to the Sun?",
    "How to change a light bulb?",
    "What is the square root of 144?",
    "How many hours in a week?",
    "What is the atomic number of gold?",
    "How to parallel park?",
    "What is the average lifespan of a cat?",
    "How to calculate BMI?",
    "What is the smallest country in the world?",
    "How to make scrambled eggs?",
    "What is the freezing point of water in Fahrenheit?",
    "How many continents are there?",
    "What is the speed of sound?",
    "How to fold a paper airplane?",
]

GREETING_QUERIES = [
    "Hello", "Hi there", "Hey", "Good morning", "Good afternoon",
    "Good evening", "Hello, how are you?", "Hi, nice to meet you",
    "Hey there!", "Greetings", "What's up?", "Howdy", "Good day",
    "Hello world", "Hi!", "Hey, how's it going?", "Yo", "Hola",
    "Bonjour", "Hallo", "Ciao", "How are you doing?",
    "Nice to see you", "Good to meet you", "What's going on?",
    "How do you do?", "Salutations", "Sup?", "Heya", "Ahoy",
]


def build_annotated_set(
    limit_per_dataset: int = SAMPLES_PER_DATASET,
    seed: int = 42,
) -> list[dict[str, str]]:
    random.seed(seed)
    annotated = []

    for ds_name, intent_label in DATASET_INTENT_MAP.items():
        qa_path = PROJECT_ROOT / "data" / ds_name / "qa.jsonl"
        if not qa_path.exists():
            logger.warning("Dataset %s not found at %s, skipping", ds_name, qa_path)
            continue

        questions = []
        with open(qa_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    q = row.get("question") or row.get("query") or row.get("q", "")
                    if q:
                        questions.append(q)

        sampled = random.sample(questions, min(limit_per_dataset, len(questions)))
        for q in sampled:
            annotated.append({
                "question": q,
                "ground_truth": intent_label,
                "source": ds_name,
            })

    for q in OTHER_QUERIES:
        annotated.append({"question": q, "ground_truth": "other", "source": "constructed"})
    for q in GREETING_QUERIES:
        annotated.append({"question": q, "ground_truth": "greeting", "source": "constructed"})

    random.shuffle(annotated)
    logger.info("Built annotated set: %d samples", len(annotated))
    return annotated


def classify_query(chimera, question: str) -> str:
    from chimera_rag.core.types import Query
    query = Query(text=question)
    classifier = chimera.query_pipeline.intent_classifier
    intent = classifier.classify(query)
    return intent.label


def compute_classification_metrics(
    predictions: list[str],
    references: list[str],
    labels: list[str],
) -> dict[str, Any]:
    per_label: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in labels
    }
    correct = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for pred, ref in zip(predictions, references):
        confusion[ref][pred] += 1
        if pred == ref:
            correct += 1
            per_label[ref]["tp"] += 1
        else:
            if pred in per_label:
                per_label[pred]["fp"] += 1
            per_label[ref]["fn"] += 1

    accuracy = correct / len(predictions) if predictions else 0
    total = len(predictions)

    per_label_metrics = {}
    for label in labels:
        tp = per_label[label]["tp"]
        fp = per_label[label]["fp"]
        fn = per_label[label]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_label_metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": tp + fn,
        }

    macro_p = sum(m["precision"] for m in per_label_metrics.values()) / len(labels)
    macro_r = sum(m["recall"] for m in per_label_metrics.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in per_label_metrics.values()) / len(labels)
    weighted_p = sum(m["precision"] * m["support"] for m in per_label_metrics.values()) / total if total else 0
    weighted_r = sum(m["recall"] * m["support"] for m in per_label_metrics.values()) / total if total else 0
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_label_metrics.values()) / total if total else 0

    return {
        "accuracy": round(accuracy, 3),
        "macro_avg": {"precision": round(macro_p, 3), "recall": round(macro_r, 3), "f1": round(macro_f1, 3)},
        "weighted_avg": {"precision": round(weighted_p, 3), "recall": round(weighted_r, 3), "f1": round(weighted_f1, 3)},
        "per_label": per_label_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "n_samples": total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Intent Classification Accuracy"
    )
    add_common_args(parser)
    parser.add_argument(
        "--config", default="configs/colla_rag_only.yaml",
        help="Config with TreeIntentClassifier enabled",
    )
    parser.add_argument(
        "--samples-per-dataset", type=int, default=SAMPLES_PER_DATASET,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = setup_experiment(args, EXP_NAME)

    annotated = build_annotated_set(
        limit_per_dataset=args.samples_per_dataset, seed=args.seed,
    )

    annotated_path = out_dir / "annotated_set.jsonl"
    with open(annotated_path, "w") as f:
        for item in annotated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    from chimera_rag import ChimeraRAG
    chimera = ChimeraRAG.from_config(str(PROJECT_ROOT / args.config))

    predictions, references, misclassifications = [], [], []

    for i, item in enumerate(annotated):
        pred = classify_query(chimera, item["question"])
        ref = item["ground_truth"]
        predictions.append(pred)
        references.append(ref)
        if pred != ref:
            misclassifications.append({
                "question": item["question"],
                "predicted": pred,
                "ground_truth": ref,
                "source": item["source"],
            })
        if (i + 1) % 50 == 0:
            logger.info("Classified %d/%d queries", i + 1, len(annotated))

    labels = ["greeting", "single_hop", "multi_hop", "summarization", "other"]
    metrics = compute_classification_metrics(predictions, references, labels)

    md_path = out_dir / "classification_report.md"
    with open(md_path, "w") as f:
        f.write("# Intent Classification Accuracy Report\n\n")
        f.write(f"Overall Accuracy: **{metrics['accuracy']:.3f}** ({metrics['n_samples']} samples)\n\n")
        headers = ["Intent Label", "Precision", "Recall", "F1", "Support"]
        rows = []
        for label, m in metrics["per_label"].items():
            rows.append([label, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", str(m["support"])])
        rows.append(["**Macro Avg**", f"{metrics['macro_avg']['precision']:.3f}",
                      f"{metrics['macro_avg']['recall']:.3f}", f"{metrics['macro_avg']['f1']:.3f}", str(metrics["n_samples"])])
        rows.append(["**Weighted Avg**", f"{metrics['weighted_avg']['precision']:.3f}",
                      f"{metrics['weighted_avg']['recall']:.3f}", f"{metrics['weighted_avg']['f1']:.3f}", str(metrics["n_samples"])])
        f.write(render_markdown_table(headers, rows))
        f.write("\n\n")

        if misclassifications:
            f.write("## Misclassification Analysis\n\n")
            direction_counts: dict[str, list] = defaultdict(list)
            for mc in misclassifications:
                direction_counts[f"{mc['ground_truth']} -> {mc['predicted']}"].append(mc["question"])
            mc_headers = ["Direction", "Count", "Example"]
            mc_rows = []
            for direction, questions in sorted(direction_counts.items(), key=lambda x: -len(x[1])):
                mc_rows.append([direction, str(len(questions)), questions[0][:80]])
            f.write(render_markdown_table(mc_headers, mc_rows))
            f.write("\n")

    save_results(
        {"metrics": metrics, "misclassifications": misclassifications},
        out_dir, "classification_results",
    )
    logger.info("Intent classification accuracy: %.1f%%", metrics["accuracy"] * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
