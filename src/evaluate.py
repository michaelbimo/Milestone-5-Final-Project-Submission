"""Evaluate Milestone 5 generative experiments and produce report-ready artifacts.

From the repository root:
    python src/evaluate.py

This script requires the experiment JSONL files produced by ``src/experiment_runner.py``.
It calculates transparent automatic grounding metrics, creates a human-review template, and
writes comparison charts for the final technical report and presentation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation_metrics import (
    archetype_keyword_coverage,
    non_empty,
    red_flag_coverage,
    risk_level_mentioned,
    source_token_support,
    unsupported_number_rate,
    word_count,
)
from utils.helpers import ensure_directory, write_json


RATING_COLUMNS = ["factuality_1_5", "relevance_1_5", "clarity_1_5", "usefulness_1_5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CFPB generative-model experiments.")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment file: {path}. Run `python src/experiment_runner.py` first."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Experiment file is empty: {path}")
    if "generated_summary" not in rows[0]:
        raise ValueError(
            f"{path} contains prepared prompts but no generated summaries. "
            "Run experiment_runner.py without --prepare-only."
        )
    return rows


def score_rows(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, float]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        summary = row.get("generated_summary", "")
        scored.append(
            {
                "sample_number": row.get("sample_number"),
                "complaint_id": row.get("complaint_id"),
                "prompt_style": row.get("prompt_style"),
                "summary": summary,
                "non_empty": float(non_empty(summary)),
                "summary_words": word_count(summary),
                "risk_mention": float(risk_level_mentioned(summary, row.get("risk_level", ""))),
                "red_flag_coverage": red_flag_coverage(summary, row.get("red_flags", [])),
                "archetype_keyword_coverage": archetype_keyword_coverage(
                    summary, row.get("archetype", "")
                ),
                "unsupported_number_rate": unsupported_number_rate(
                    summary, row.get("clean_text", "")
                ),
                "source_token_support": source_token_support(summary, row.get("clean_text", "")),
            }
        )
    frame = pd.DataFrame(scored)
    metrics = {
        "samples": int(len(frame)),
        "non_empty_output_rate": float(frame["non_empty"].mean()),
        "average_summary_words": float(frame["summary_words"].mean()),
        "risk_level_mention_rate": float(frame["risk_mention"].mean()),
        "mean_red_flag_coverage": float(frame["red_flag_coverage"].mean()),
        "mean_archetype_keyword_coverage": float(frame["archetype_keyword_coverage"].mean()),
        "mean_unsupported_number_rate": float(frame["unsupported_number_rate"].mean()),
        "mean_source_token_support": float(frame["source_token_support"].mean()),
    }
    return frame, metrics


def merge_human_scores(metrics: dict[str, dict[str, float]], template_path: Path) -> None:
    """Add aggregate human ratings when the reviewer has filled the template."""
    if not template_path.exists():
        return
    human = pd.read_csv(template_path)
    if not set(RATING_COLUMNS).issubset(human.columns):
        return
    numeric = human[RATING_COLUMNS].apply(pd.to_numeric, errors="coerce")
    human = pd.concat([human.drop(columns=RATING_COLUMNS), numeric], axis=1)
    for style, group in human.groupby("prompt_style"):
        if style not in metrics:
            continue
        style_scores = group[RATING_COLUMNS]
        if style_scores.notna().any().any():
            for column in RATING_COLUMNS:
                values = style_scores[column].dropna()
                if not values.empty:
                    metrics[style][f"human_{column}_mean"] = float(values.mean())
            all_values = style_scores.to_numpy(dtype=float)
            if np.isfinite(all_values).any():
                metrics[style]["human_overall_mean"] = float(np.nanmean(all_values))


def create_human_template(rows_by_style: dict[str, list[dict[str, Any]]], path: Path) -> None:
    records: list[dict[str, Any]] = []
    for style, rows in rows_by_style.items():
        for row in rows:
            records.append(
                {
                    "prompt_style": style,
                    "sample_number": row.get("sample_number"),
                    "complaint_id": row.get("complaint_id"),
                    "complaint": row.get("clean_text", ""),
                    "generated_summary": row.get("generated_summary", ""),
                    "factuality_1_5": "",
                    "relevance_1_5": "",
                    "clarity_1_5": "",
                    "usefulness_1_5": "",
                    "hallucination_found_yes_no": "",
                    "review_notes": "",
                }
            )
    pd.DataFrame(records).to_csv(path, index=False)


def make_metric_chart(comparison: pd.DataFrame, figures_dir: Path) -> None:
    metrics = [
        "risk_level_mention_rate",
        "mean_red_flag_coverage",
        "mean_archetype_keyword_coverage",
        "mean_source_token_support",
    ]
    labels = ["Risk mention", "Red-flag coverage", "Archetype coverage", "Source support"]
    available = [metric for metric in metrics if metric in comparison.columns]
    if not available:
        return

    ax = comparison.set_index("prompt_style")[available].T.plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Baseline vs. structured prompt: automatic grounding metrics")
    ax.set_xticklabels([labels[metrics.index(metric)] for metric in available], rotation=20, ha="right")
    ax.legend(title="Prompt style")
    plt.tight_layout()
    plt.savefig(figures_dir / "prompt_comparison_metrics.png", dpi=180)
    plt.close()


def make_length_chart(scored_frames: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    data = [frame["summary_words"].to_numpy() for frame in scored_frames.values()]
    labels = list(scored_frames.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Generated summary length (words)")
    ax.set_title("Summary-length distribution by prompt style")
    plt.tight_layout()
    plt.savefig(figures_dir / "summary_length_distribution.png", dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    results_dir = ensure_directory(PROJECT_ROOT / cfg["results"]["directory"])
    experiments_dir = PROJECT_ROOT / cfg["results"]["experiments_directory"]
    figures_dir = ensure_directory(PROJECT_ROOT / cfg["results"]["figures_directory"])
    template_path = PROJECT_ROOT / cfg["results"]["human_template_file"]

    rows_by_style: dict[str, list[dict[str, Any]]] = {}
    scored_frames: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, float]] = {}

    for style in cfg["experiment"]["prompt_styles"]:
        rows = read_jsonl(experiments_dir / f"{style}_prompt.jsonl")
        rows_by_style[style] = rows
        frame, style_metrics = score_rows(rows)
        scored_frames[style] = frame
        metrics[style] = style_metrics
        frame.to_csv(results_dir / f"{style}_sample_metrics.csv", index=False)

    # Create the human template only when it does not exist, so completed ratings are preserved.
    if not template_path.exists():
        create_human_template(rows_by_style, template_path)
        print(f"Created human-review template: {template_path}")

    merge_human_scores(metrics, template_path)

    comparison = pd.DataFrame(
        [{"prompt_style": style, **style_metrics} for style, style_metrics in metrics.items()]
    )
    comparison.to_csv(PROJECT_ROOT / cfg["results"]["comparison_file"], index=False)
    write_json(PROJECT_ROOT / cfg["results"]["metrics_file"], metrics)

    make_metric_chart(comparison, figures_dir)
    make_length_chart(scored_frames, figures_dir)

    print("\nFinal comparison:")
    print(comparison.to_string(index=False))
    print(f"\nMetrics: {PROJECT_ROOT / cfg['results']['metrics_file']}")
    print(f"Comparison CSV: {PROJECT_ROOT / cfg['results']['comparison_file']}")
    print(f"Figures: {figures_dir}")
    print(f"Human review template: {template_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
