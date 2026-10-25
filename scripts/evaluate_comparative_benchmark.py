#!/usr/bin/env python3
"""scripts/evaluate_comparative_benchmark.py

Comparative Evaluation Runner across Unimodal and Fusion Baselines for JIGSAWS.
Executes Phase A (Readiness & Alignment Audit), Phase B (Comparative Evaluation),
and Phase C (Hypothesis Analysis) under the official Leave-One-User-Out (LOUO) protocol.

Usage:
    python scripts/evaluate_comparative_benchmark.py --task Suturing
    python scripts/evaluate_comparative_benchmark.py --task all --audit-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark_comparison import (
    audit_task_multimodal_readiness,
    compare_model_benchmarks,
    export_comparison_csv,
    format_comparison_markdown,
    format_readiness_audit_markdown,
)
from src.utils.reproducibility import save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_comparative_benchmark")

ALL_TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run comparative evaluation and multimodal readiness audit for JIGSAWS baselines."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Suturing",
        choices=["Suturing", "Knot_Tying", "Needle_Passing", "all"],
        help="JIGSAWS task name, or 'all' to evaluate all tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/analysis",
        help="Directory to save audit and comparison reports.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Path to JIGSAWS dataset directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache/features",
        help="Path to visual feature cache directory.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Path to model checkpoints directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Path to artifacts containing baseline evaluation result files.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="ResNet18",
        help="Visual feature encoder name (default: ResNet18).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="capture1",
        help="Camera view (default: capture1).",
    )
    parser.add_argument(
        "--max-delta",
        type=int,
        default=500,
        help="Maximum allowed frame delta between video and kinematics for 30 Hz sync.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Perform readiness and alignment audit only, without executing model comparisons.",
    )
    return parser.parse_args()


def load_model_summary(summary_file: Path) -> Dict[str, Any]:
    """Load JSON baseline result file."""
    with open(summary_file, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_task(task: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Perform audit and comparative benchmark for a single task."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)

    logger.info("=" * 70)
    logger.info(f"AUDIT & COMPARATIVE EVALUATION: {task}")
    logger.info("=" * 70)

    # 1. Phase A: Audit Readiness and Alignment
    audit = audit_task_multimodal_readiness(
        task=task,
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        checkpoints_dir=args.checkpoints_dir,
        encoder_name=args.encoder,
        camera=args.camera,
        max_allowed_delta=args.max_delta,
    )

    audit_json_path = output_dir / f"benchmark_readiness_audit_{task}.json"
    audit_md_path = output_dir / f"benchmark_readiness_audit_{task}.md"

    save_json(audit, audit_json_path)
    md_content = format_readiness_audit_markdown(audit)
    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(f"Readiness Status for {task}: {audit['status']}")
    logger.info(f"Feature caches: {audit['cached_trials_count']}/{audit['total_trials']} verified.")
    logger.info(f"Audit report saved to: {audit_md_path}")

    if args.audit_only:
        return {"task": task, "status": audit["status"], "audit": audit}

    # 2. Check baseline summaries available in artifacts
    candidate_models: Dict[str, List[str]] = {
        "kinematics_mlp": [
            f"kinematics_mlp_{task}_results.json",
        ],
        "kinematics_gru": [
            f"kinematics_gru_{task}_results.json",
        ],
        "video_gru": [
            f"video_videogru_{task}_results.json",
            f"video_gru_{task}_results.json",
        ],
        "fixed_late_fusion": [
            f"late_fusion_{task}_results.json",
        ],
        "concat_fusion": [
            f"concat_fusion_{task}_results.json",
        ],
    }

    loaded_summaries: Dict[str, Dict[str, Any]] = {}
    missing_models: List[str] = []

    for m_id, candidates in candidate_models.items():
        found = False
        for c in candidates:
            p = artifacts_dir / c
            if p.is_file():
                try:
                    loaded_summaries[m_id] = load_model_summary(p)
                    found = True
                    break
                except Exception as e:
                    logger.warning(f"Error loading {p}: {e}")
        if not found:
            missing_models.append(m_id)

    logger.info(f"Available baseline summaries: {list(loaded_summaries.keys())}")
    if missing_models:
        logger.warning(f"Missing baseline summaries: {missing_models}")

    # 3. Phase B & C: Compare models if we have at least 2 models
    if len(loaded_summaries) >= 2:
        try:
            comparison = compare_model_benchmarks(loaded_summaries, task=task)
            comp_json_path = output_dir / f"fusion_comparison_{task}.json"
            comp_csv_path = output_dir / f"fusion_comparison_{task}.csv"
            comp_md_path = output_dir / f"fusion_comparison_{task}.md"

            save_json(comparison, comp_json_path)
            export_comparison_csv(comparison, comp_csv_path)
            comp_md = format_comparison_markdown(comparison)
            with open(comp_md_path, "w", encoding="utf-8") as f:
                f.write(comp_md)

            logger.info(f"Comparison report saved to: {comp_md_path}")
            logger.info(f"Comparison CSV saved to: {comp_csv_path}")

            return {
                "task": task,
                "status": audit["status"],
                "audit": audit,
                "comparison": comparison,
                "missing_models": missing_models,
            }
        except Exception as e:
            logger.error(f"Failed to compare models for {task}: {e}")
            return {
                "task": task,
                "status": "ERROR",
                "error": str(e),
                "audit": audit,
                "missing_models": missing_models,
            }
    else:
        logger.warning(
            f"Insufficient completed baselines for full comparative table (found {len(loaded_summaries)}). "
            f"Comparative benchmark remains BLOCKED."
        )
        return {
            "task": task,
            "status": "BLOCKED",
            "audit": audit,
            "missing_models": missing_models,
            "available_models": list(loaded_summaries.keys()),
        }


def main() -> None:
    args = parse_args()
    tasks = ALL_TASKS if args.task == "all" else [args.task]

    overall_results = {}
    for t in tasks:
        overall_results[t] = evaluate_task(t, args)

    print("\n" + "=" * 70)
    print("ITER-19 MULTIMODAL COMPARATIVE BENCHMARK STATUS SUMMARY")
    print("=" * 70)
    for t, res in overall_results.items():
        st = res.get("status", "UNKNOWN")
        print(f"Task: {t:<15} | Status: {st}")
        audit = res.get("audit", {})
        if audit:
            print(f"  - Cached features: {audit.get('cached_trials_count', 0)}/{audit.get('total_trials', 0)}")
            print(f"  - Alignment exceptions: {len(audit.get('alignment_exceptions', []))}")
        if "comparison" in res:
            recs = res["comparison"].get("records", [])
            print(f"  - Evaluated baselines ({len(recs)} models):")
            for r in recs:
                print(f"      * {r['model_name']:<25} | Acc: {r['accuracy']:.2f}% | F1: {r['macro_f1']:.2f}% | d_Acc: {r.get('delta_accuracy', 0.0):+.2f}%")
        elif "missing_models" in res:
            print(f"  - Missing baseline models: {', '.join(res['missing_models'])}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
