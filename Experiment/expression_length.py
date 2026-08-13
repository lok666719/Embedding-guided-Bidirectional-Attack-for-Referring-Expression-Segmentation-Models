import json
import numpy as np
from collections import defaultdict

with open("all_expression_records.json", "r", encoding="utf-8") as f:
    records = json.load(f)


def get_length_group(prompt):
    n_words = len(prompt.strip().split())

    if n_words <= 3:
        return "Short (1-3)"
    elif n_words <= 6:
        return "Medium (4-6)"
    else:
        return "Long (>=7)"


def summarize(items):
    n = len(items)

    asr = 100.0 * np.mean([
        float(x["is_success"])
        for x in items
    ])

    miou_gt = np.mean([
        float(x["iou_foreground"])
        for x in items
    ])

    miou_utm = np.mean([
        float(x["iou_background"])
        for x in items
    ])

    return {
        "num_samples": n,
        "ASR@30": round(asr, 2),
        "mIoU-GT": round(miou_gt, 2),
        "mIoU-UTM": round(miou_utm, 2),
    }


# ============================================================
# Overall analysis
# ============================================================

groups = defaultdict(list)

for r in records:
    group = get_length_group(r["prompt"])
    groups[group].append(r)


order = [
    "Short (1-3)",
    "Medium (4-6)",
    "Long (>=7)"
]

overall_results = []

print("\n=== Expression Length Analysis ===")

for group in order:
    result = summarize(groups[group])

    result["length_group"] = group
    overall_results.append(result)

    print(
        f"{group:15s} "
        f"N={result['num_samples']:5d}  "
        f"ASR@30={result['ASR@30']:6.2f}  "
        f"mIoU-GT={result['mIoU-GT']:6.2f}  "
        f"mIoU-UTM={result['mIoU-UTM']:6.2f}"
    )


# ============================================================
# Per-dataset analysis
# ============================================================

dataset_results = {}

for dataset in ["refcoco", "refcoco+", "refcocog"]:

    subset = [
        r for r in records
        if r["dataset"] == dataset
    ]

    dataset_groups = defaultdict(list)

    for r in subset:
        group = get_length_group(r["prompt"])
        dataset_groups[group].append(r)

    dataset_results[dataset] = []

    print(f"\n=== {dataset} ===")

    for group in order:

        items = dataset_groups[group]

        if len(items) == 0:
            continue

        result = summarize(items)
        result["length_group"] = group

        dataset_results[dataset].append(result)

        print(
            f"{group:15s} "
            f"N={result['num_samples']:5d}  "
            f"ASR@30={result['ASR@30']:6.2f}  "
            f"mIoU-GT={result['mIoU-GT']:6.2f}  "
            f"mIoU-UTM={result['mIoU-UTM']:6.2f}"
        )


# ============================================================
# Save
# ============================================================

output = {
    "group_definition": {
        "Short": "1-3 words",
        "Medium": "4-6 words",
        "Long": ">=7 words",
        "Q1": 3,
        "median": 4,
        "Q3": 6
    },
    "overall": overall_results,
    "per_dataset": dataset_results
}

with open(
    "expression_length_analysis.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nSaved to expression_length_analysis.json")