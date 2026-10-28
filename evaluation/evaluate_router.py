import json
import time
import os
import sys
from collections import defaultdict
from typing import Callable, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "routing_benchmark.json")


def load_benchmark(path: str = BENCHMARK_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_domain_label(domain: str) -> str:
    """Normalize domain strings for comparison (e.g. government -> government_economic)."""
    d = domain.lower().strip()
    if d in ("government", "government_economic"):
        return "government_economic"
    return d


def evaluate_classifier(
    classify_fn: Callable[[str], Any],
    name: str = "Classifier",
    dataset: list[dict] | None = None,
) -> dict:
    """
    Evaluate a classification or routing function against the benchmark dataset.
    classify_fn can return:
      - str: e.g. "biomedical"
      - dict: e.g. {"domains": ["biomedical", "financial"], "primary": "biomedical", ...}
      - tuple/list: e.g. (["biomedical"], score)
    """
    if dataset is None:
        dataset = load_benchmark()

    latencies = []
    y_true_primary = []
    y_pred_primary = []
    multi_domain_stats = {"total": 0, "any_match": 0, "full_match": 0}
    confusion = defaultdict(lambda: defaultdict(int))
    domain_labels = set()

    for item in dataset:
        claim = item["claim"]
        expected_domains = [normalize_domain_label(d) for d in item["expected_domains"]]
        primary_expected = normalize_domain_label(item["primary_domain"])
        domain_labels.add(primary_expected)
        for ed in expected_domains:
            domain_labels.add(ed)

        t0 = time.perf_counter()
        raw_res = classify_fn(claim)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Parse result
        if isinstance(raw_res, str):
            pred_primary = normalize_domain_label(raw_res)
            pred_domains = [pred_primary]
        elif isinstance(raw_res, dict):
            pred_domains = [normalize_domain_label(d) for d in raw_res.get("domains", [])]
            pred_primary = normalize_domain_label(raw_res.get("primary_domain") or (pred_domains[0] if pred_domains else "unknown"))
        elif hasattr(raw_res, "primary_domain"):
            pred_primary = normalize_domain_label(raw_res.primary_domain)
            pred_domains = [normalize_domain_label(d) for d in getattr(raw_res, "domains", [pred_primary])]
        elif isinstance(raw_res, (list, tuple)):
            pred_domains = [normalize_domain_label(d) for d in raw_res]
            pred_primary = pred_domains[0] if pred_domains else "unknown"
        else:
            pred_primary = normalize_domain_label(str(raw_res))
            pred_domains = [pred_primary]

        domain_labels.add(pred_primary)
        for pd in pred_domains:
            domain_labels.add(pd)

        # Primary domain tracking
        y_true_primary.append(primary_expected)
        y_pred_primary.append(pred_primary)
        confusion[primary_expected][pred_primary] += 1

        # Multi-domain tracking
        if len(expected_domains) > 1:
            multi_domain_stats["total"] += 1
            if any(ed in pred_domains for ed in expected_domains):
                multi_domain_stats["any_match"] += 1
            if all(ed in pred_domains for ed in expected_domains):
                multi_domain_stats["full_match"] += 1

    # Single-domain accuracy (for claims where expected len == 1)
    single_domain_items = [(t, p) for item, t, p in zip(dataset, y_true_primary, y_pred_primary) if len(item["expected_domains"]) == 1]
    single_correct = sum(1 for t, p in single_domain_items if t == p)
    single_acc = (single_correct / len(single_domain_items)) if single_domain_items else 0.0

    # Overall top-1 primary accuracy across all 70 claims
    total_correct = sum(1 for t, p in zip(y_true_primary, y_pred_primary) if t == p)
    overall_acc = total_correct / len(dataset)

    # Per-domain metrics
    per_domain = {}
    f1_list = []
    clean_labels = sorted([lbl for lbl in domain_labels if lbl not in ("unknown", "news", "space")])

    for lbl in clean_labels:
        tp = sum(1 for t, p in zip(y_true_primary, y_pred_primary) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true_primary, y_pred_primary) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true_primary, y_pred_primary) if t == lbl and p != lbl)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_domain[lbl] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true_primary if t == lbl),
        }
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    # Latency
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else 0.0

    report = {
        "classifier_name": name,
        "dataset_size": len(dataset),
        "single_domain_count": len(single_domain_items),
        "single_domain_accuracy": round(single_acc, 4),
        "overall_primary_accuracy": round(overall_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_domain": per_domain,
        "multi_domain": {
            "total_multi_claims": multi_domain_stats["total"],
            "any_expected_domain_detected": multi_domain_stats["any_match"],
            "all_expected_domains_detected": multi_domain_stats["full_match"],
            "any_match_rate": round(multi_domain_stats["any_match"] / multi_domain_stats["total"], 4) if multi_domain_stats["total"] else 0.0,
            "full_match_rate": round(multi_domain_stats["full_match"] / multi_domain_stats["total"], 4) if multi_domain_stats["total"] else 0.0,
        },
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "p95": round(p95_lat, 2),
        },
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
    return report


if __name__ == "__main__":
    from app.domain_classifier import keyword_only_classify, classify_claim
    from app.router import route_claim

    print("Evaluating baseline keyword classifier...")
    baseline_res = evaluate_classifier(keyword_only_classify, name="Keyword-Only Baseline")

    print("Evaluating new hybrid semantic router...")
    hybrid_res = evaluate_classifier(route_claim, name="Hybrid Semantic Router")

    comparison = {
        "baseline": baseline_res,
        "hybrid_semantic": hybrid_res,
        "improvements": {
            "single_domain_accuracy_delta": round(hybrid_res["single_domain_accuracy"] - baseline_res["single_domain_accuracy"], 4),
            "overall_accuracy_delta": round(hybrid_res["overall_primary_accuracy"] - baseline_res["overall_primary_accuracy"], 4),
            "macro_f1_delta": round(hybrid_res["macro_f1"] - baseline_res["macro_f1"], 4),
            "multi_domain_full_match_delta": round(hybrid_res["multi_domain"]["full_match_rate"] - baseline_res["multi_domain"]["full_match_rate"], 4),
        }
    }

    report_path = os.path.join(os.path.dirname(__file__), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n=== EVALUATION REPORT ===")
    print(f"Dataset Size: {hybrid_res['dataset_size']} claims ({hybrid_res['single_domain_count']} single, {hybrid_res['multi_domain']['total_multi_claims']} multi)")
    print(f"Single-Domain Accuracy: {baseline_res['single_domain_accuracy']:.2%} -> {hybrid_res['single_domain_accuracy']:.2%} (+{comparison['improvements']['single_domain_accuracy_delta']:.2%})")
    print(f"Overall Accuracy:       {baseline_res['overall_primary_accuracy']:.2%} -> {hybrid_res['overall_primary_accuracy']:.2%} (+{comparison['improvements']['overall_accuracy_delta']:.2%})")
    print(f"Macro-F1 Score:         {baseline_res['macro_f1']:.2%} -> {hybrid_res['macro_f1']:.2%} (+{comparison['improvements']['macro_f1_delta']:.2%})")
    print(f"Multi-Domain Full Match: {baseline_res['multi_domain']['full_match_rate']:.2%} -> {hybrid_res['multi_domain']['full_match_rate']:.2%}")
    print(f"Latency (mean / p95):   {hybrid_res['latency_ms']['mean']} ms / {hybrid_res['latency_ms']['p95']} ms")
    print(f"Report saved to: {report_path}")
