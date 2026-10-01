"""
ClarifySign - Translation Dataset Preparation Module
Prepares gold-standard domain training pairs from core.translation.VERIFIED_SEMANTIC_DATABASE
augmented with IN22-Gen / IN22-Conv samples for fine-tuning IndicTrans2.

Base Model: ai4bharat/indictrans2-en-indic-dist-200M
Languages: Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Punjabi, Odia.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import LANGUAGES
from core.translation import VERIFIED_SEMANTIC_DATABASE, CONTEXTUAL_REALIZATIONS

DATA_DIR = ROOT_DIR / "data" / "translation"


def build_gold_domain_pairs() -> List[Dict[str, str]]:
    """
    Extracts gold-standard English -> Indic language pairs from the verified semantic database
    and contextual realizations for all 10 supported Indian languages.
    """
    pairs = []

    # 1. Semantic concept pairs
    for concept, lang_dict in VERIFIED_SEMANTIC_DATABASE.items():
        # Canonical English expressions
        en_phrases = [
            f"I want {concept}.",
            f"Please show me {concept}.",
            f"{concept.capitalize()}."
        ]
        for en in en_phrases:
            for lang, target_text in lang_dict.items():
                if lang in LANGUAGES:
                    pairs.append({
                        "source": en,
                        "target": target_text,
                        "target_language": lang,
                        "domain": "retail_gold",
                        "weight": 2.0
                    })

    # 2. Contextual realization pairs
    for act, lang_dict in CONTEXTUAL_REALIZATIONS.items():
        en_act = act.replace("_", " ").capitalize()
        for lang, target_text in lang_dict.items():
            if lang in LANGUAGES:
                pairs.append({
                    "source": f"Clarification: {en_act}",
                    "target": target_text,
                    "target_language": lang,
                    "domain": "dialogue_act_gold",
                    "weight": 2.0
                })

    return pairs


def build_augmented_pairs() -> List[Dict[str, str]]:
    """
    General-domain augmentation pairs reflecting IN22-Gen and IN22-Conv topics
    weighted lower than domain-specific retail concepts.
    """
    general_templates = [
        ("Good morning, how can I help you?", {
            "Hindi": "सुप्रभात, मैं आपकी क्या मदद कर सकता हूँ?",
            "Marathi": "शुभ सकाळ, मी तुम्हाला कशी मदत करू शकतो?",
            "Bengali": "সুপ্রভাত, আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
            "Tamil": "காலை வணக்கம், நான் உங்களுக்கு எவ்வாறு உதவ முடியும்?"
        }),
        ("What is the total price?", {
            "Hindi": "कुल कीमत क्या है?",
            "Marathi": "एकूण किंमत काय आहे?",
            "Bengali": "মোট মূল্য কত?",
            "Tamil": "மொத்த விலை என்ன?"
        }),
        ("Thank you for visiting our shop.", {
            "Hindi": "हमारी दुकान पर आने के लिए धन्यवाद।",
            "Marathi": "आमच्या दुकानाला भेट दिल्याबद्दल धन्यवाद.",
            "Bengali": "আমাদের দোকানে আসার জন্য ধন্যবাদ।",
            "Tamil": "எங்கள் கடைக்கு வருகை தந்தமைக்கு நன்றி."
        })
    ]

    pairs = []
    for en, lang_dict in general_templates:
        for lang, target_text in lang_dict.items():
            if lang in LANGUAGES:
                pairs.append({
                    "source": en,
                    "target": target_text,
                    "target_language": lang,
                    "domain": "general_augmentation",
                    "weight": 0.5
                })

    return pairs


def prepare_translation_dataset(out_dir: Path = DATA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = build_gold_domain_pairs()
    aug = build_augmented_pairs()
    all_pairs = gold + aug

    # Split 80% train, 10% val, 10% test deterministically
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_set = all_pairs[:n_train]
    val_set = all_pairs[n_train:n_train + n_val]
    test_set = all_pairs[n_train + n_val:]

    for name, dataset in [("train.jsonl", train_set), ("val.jsonl", val_set), ("test.jsonl", test_set)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "total_pairs": len(all_pairs),
        "gold_domain_pairs": len(gold),
        "general_augmentation_pairs": len(aug),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "languages_covered": list(LANGUAGES.keys())
    }

    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Translation dataset prepared at {out_dir}:")
    print(f"- Gold retail domain pairs: {len(gold)}")
    print(f"- Augmentation pairs:       {len(aug)}")
    print(f"- Train / Val / Test:       {len(train_set)} / {len(val_set)} / {len(test_set)}")
    return summary


if __name__ == "__main__":
    prepare_translation_dataset()
