"""
ClarifySign - Recognizer Offline Evaluation
Computes quantitative recognition metrics on test data:
Top-1 accuracy, Top-3 accuracy, Confusion matrix, Average entropy, and Average margin.
Saves detailed JSON metrics report to results/recognizer_evaluation.json.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    PROCESSED_DATA_DIR,
    MODEL_PATH,
    LABELS_PATH,
    RESULTS_DIR
)
from core.recognizer import ISLRecognizer
from core.uncertainty import entropy, margin


def evaluate(data_dir=PROCESSED_DATA_DIR, out_file=RESULTS_DIR / "recognizer_evaluation.json"):
    recognizer = ISLRecognizer(model_path=MODEL_PATH, labels_path=LABELS_PATH)
    if not recognizer.available:
        print(f"Error: Model not found at {MODEL_PATH}. Train the model first.")
        return

    y_true = []
    y_pred = []
    top3_correct = 0
    total = 0
    entropies = []
    margins = []

    classes = recognizer.labels

    for cls in classes:
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            continue
        files = list(cls_dir.glob("*.npy"))
        for f in files:
            try:
                seq = np.load(f)
                candidates = recognizer.predict(seq, top_k=min(5, len(classes)))
                pred_label = candidates[0][0]

                y_true.append(cls)
                y_pred.append(pred_label)

                top_3_labels = [c[0] for c in candidates[:3]]
                if cls in top_3_labels:
                    top3_correct += 1

                entropies.append(entropy(candidates))
                margins.append(margin(candidates))
                total += 1
            except Exception as e:
                print(f"Error evaluating {f}: {e}")

    if total == 0:
        print("No evaluation sequences found.")
        return

    top1_acc = np.mean([1 if yt == yp else 0 for yt, yp in zip(y_true, y_pred)])
    top3_acc = top3_correct / total
    avg_entropy = float(np.mean(entropies))
    avg_margin = float(np.mean(margins))

    report_dict = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    result = {
        "dataset_path": str(data_dir),
        "total_evaluated_samples": total,
        "num_classes": len(classes),
        "classes": classes,
        "top1_accuracy": round(float(top1_acc), 4),
        "top3_accuracy": round(float(top3_acc), 4),
        "average_entropy_bits": round(avg_entropy, 4),
        "average_margin": round(avg_margin, 4),
        "classification_report": report_dict,
        "confusion_matrix": conf_mat
    }

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n--- Model Evaluation Summary ---")
    print(f"Total Evaluated Samples: {total}")
    print(f"Top-1 Accuracy:          {top1_acc*100:.2f}%")
    print(f"Top-3 Accuracy:          {top3_acc*100:.2f}%")
    print(f"Average Entropy:         {avg_entropy:.3f} bits")
    print(f"Average Top-2 Margin:    {avg_margin:.3f}")
    print(f"Report saved to:         {out_file}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ClarifySign recognizer")
    parser.add_argument("--data", default=str(PROCESSED_DATA_DIR), help="Data directory")
    parser.add_argument("--out", default=str(RESULTS_DIR / "recognizer_evaluation.json"), help="Output JSON path")
    args = parser.parse_args()

    evaluate(data_dir=Path(args.data), out_file=Path(args.out))


if __name__ == "__main__":
    main()
