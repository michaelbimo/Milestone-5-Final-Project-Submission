"""Evaluate Milestone 5 generative experiments and produce report-ready artifacts.

From the repository root:
    python src/evaluate.py

This evaluation separates:

1. Raw FLAN-T5 model performance
2. Validated summary performance after guardrails/fallback
3. Guardrail behavior and fallback rate
4. Final analyst-facing human evaluation

The experiment JSONL files must first be produced by:
    python src/experiment_runner.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

from utils.helpers import (
    ensure_directory,
    write_json,
)


RATING_COLUMNS = [
    "factuality_1_5",
    "relevance_1_5",
    "clarity_1_5",
    "usefulness_1_5",
]


# ============================================================
# Configuration
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CFPB fraud-intelligence "
            "generative-model experiments."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/eval_config.yaml",
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    """Load a YAML configuration file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
        )

    return value


# ============================================================
# Read experiment outputs
# ============================================================


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read experiment JSONL output."""

    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment file: {path}. "
            "Run `python src/experiment_runner.py` first."
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            if line.strip():
                rows.append(
                    json.loads(line)
                )

    if not rows:
        raise ValueError(
            f"Experiment file is empty: {path}"
        )

    if "generated_summary" not in rows[0]:
        raise ValueError(
            f"{path} contains prompts but no "
            "generated summaries. Run experiment_runner.py "
            "without --prepare-only."
        )

    return rows


# ============================================================
# Automatic summary metrics
# ============================================================


def score_rows(
    rows: list[dict[str, Any]],
    summary_field: str,
    evaluation_variant: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate one summary field across experiment rows.

    summary_field can be:
        generated_summary
        validated_summary

    generated_summary measures raw FLAN-T5 behavior.

    validated_summary measures the text actually used after
    the validation/fallback guardrail.
    """

    scored: list[dict[str, Any]] = []

    for row in rows:

        summary = str(
            row.get(
                summary_field,
                row.get(
                    "generated_summary",
                    "",
                ),
            )
            or ""
        )

        clean_text = str(
            row.get(
                "clean_text",
                "",
            )
            or ""
        )

        validation_issues = row.get(
            "validation_issues",
            [],
        )

        if isinstance(
            validation_issues,
            list,
        ):
            validation_issues_text = "; ".join(
                str(issue)
                for issue in validation_issues
            )
        else:
            validation_issues_text = str(
                validation_issues
            )

        scored.append(
            {
                "evaluation_variant":
                    evaluation_variant,

                "sample_number":
                    row.get(
                        "sample_number"
                    ),

                "complaint_id":
                    row.get(
                        "complaint_id"
                    ),

                "prompt_style":
                    row.get(
                        "prompt_style"
                    ),

                "summary_field":
                    summary_field,

                "summary":
                    summary,

                "summary_source":
                    row.get(
                        "summary_source",
                        "flan_t5",
                    ),

                "generation_valid":
                    row.get(
                        "generation_valid"
                    ),

                "validation_issues":
                    validation_issues_text,

                "non_empty":
                    float(
                        non_empty(
                            summary
                        )
                    ),

                "summary_words":
                    word_count(
                        summary
                    ),

                "risk_mention":
                    float(
                        risk_level_mentioned(
                            summary,
                            row.get(
                                "risk_level",
                                "",
                            ),
                        )
                    ),

                "red_flag_coverage":
                    red_flag_coverage(
                        summary,
                        row.get(
                            "red_flags",
                            [],
                        ),
                    ),

                "archetype_keyword_coverage":
                    archetype_keyword_coverage(
                        summary,
                        row.get(
                            "archetype",
                            "",
                        ),
                    ),

                "unsupported_number_rate":
                    unsupported_number_rate(
                        summary,
                        clean_text,
                    ),

                "source_token_support":
                    source_token_support(
                        summary,
                        clean_text,
                    ),
            }
        )

    frame = pd.DataFrame(
        scored
    )

    metrics = {
        "samples":
            int(
                len(frame)
            ),

        "non_empty_output_rate":
            float(
                frame[
                    "non_empty"
                ].mean()
            ),

        "average_summary_words":
            float(
                frame[
                    "summary_words"
                ].mean()
            ),

        "risk_level_mention_rate":
            float(
                frame[
                    "risk_mention"
                ].mean()
            ),

        "mean_red_flag_coverage":
            float(
                frame[
                    "red_flag_coverage"
                ].mean()
            ),

        "mean_archetype_keyword_coverage":
            float(
                frame[
                    "archetype_keyword_coverage"
                ].mean()
            ),

        "mean_unsupported_number_rate":
            float(
                frame[
                    "unsupported_number_rate"
                ].mean()
            ),

        "mean_source_token_support":
            float(
                frame[
                    "source_token_support"
                ].mean()
            ),
    }

    return frame, metrics


# ============================================================
# Guardrail / fallback evaluation
# ============================================================


def evaluate_guardrail(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure validator and fallback behavior."""

    total = len(rows)

    valid_count = sum(
        bool(
            row.get(
                "generation_valid",
                True,
            )
        )
        for row in rows
    )

    fallback_count = sum(
        row.get(
            "summary_source"
        )
        == "extractive_fallback"
        for row in rows
    )

    issue_counts: Counter[str] = (
        Counter()
    )

    for row in rows:

        issues = row.get(
            "validation_issues",
            [],
        )

        if isinstance(
            issues,
            list,
        ):

            for issue in issues:
                issue_counts[
                    str(issue)
                ] += 1

    return {
        "samples":
            total,

        "valid_generation_count":
            valid_count,

        "invalid_generation_count":
            total - valid_count,

        "valid_generation_rate":
            (
                valid_count / total
                if total
                else 0.0
            ),

        "fallback_count":
            fallback_count,

        "fallback_rate":
            (
                fallback_count / total
                if total
                else 0.0
            ),

        "validation_issue_counts":
            dict(
                issue_counts
            ),
    }


# ============================================================
# Human evaluation
# ============================================================


def create_final_human_template(
    rows_by_style:
        dict[
            str,
            list[
                dict[
                    str,
                    Any,
                ]
            ],
        ],
    path: Path,
) -> None:
    """Create final human-review template.

    Baseline:
        reviewers see raw FLAN-T5 output.

    Structured system:
        reviewers see final analyst_output after
        validation/fallback plus deterministic fraud signals.
    """

    records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    # --------------------------------------------------------
    # Baseline raw FLAN-T5 outputs
    # --------------------------------------------------------

    baseline_rows = rows_by_style.get(
        "baseline",
        [],
    )

    for row in baseline_rows:

        records.append(
            {
                "evaluation_variant":
                    "baseline_raw",

                "prompt_style":
                    "baseline",

                "sample_number":
                    row.get(
                        "sample_number"
                    ),

                "complaint_id":
                    row.get(
                        "complaint_id"
                    ),

                "complaint":
                    row.get(
                        "clean_text",
                        "",
                    ),

                "summary_source":
                    "flan_t5",

                "raw_generated_summary":
                    row.get(
                        "generated_summary",
                        "",
                    ),

                "validated_summary":
                    "",

                "output_for_review":
                    row.get(
                        "generated_summary",
                        "",
                    ),

                "factuality_1_5":
                    "",

                "relevance_1_5":
                    "",

                "clarity_1_5":
                    "",

                "usefulness_1_5":
                    "",

                "hallucination_found_yes_no":
                    "",

                "review_notes":
                    "",
            }
        )

    # --------------------------------------------------------
    # Final structured system outputs
    # --------------------------------------------------------

    structured_rows = rows_by_style.get(
        "structured",
        [],
    )

    for row in structured_rows:

        records.append(
            {
                "evaluation_variant":
                    "structured_final_system",

                "prompt_style":
                    "structured",

                "sample_number":
                    row.get(
                        "sample_number"
                    ),

                "complaint_id":
                    row.get(
                        "complaint_id"
                    ),

                "complaint":
                    row.get(
                        "clean_text",
                        "",
                    ),

                "summary_source":
                    row.get(
                        "summary_source",
                        "",
                    ),

                "raw_generated_summary":
                    row.get(
                        "generated_summary",
                        "",
                    ),

                "validated_summary":
                    row.get(
                        "validated_summary",
                        "",
                    ),

                "output_for_review":
                    row.get(
                        "analyst_output",
                        row.get(
                            "validated_summary",
                            "",
                        ),
                    ),

                "factuality_1_5":
                    "",

                "relevance_1_5":
                    "",

                "clarity_1_5":
                    "",

                "usefulness_1_5":
                    "",

                "hallucination_found_yes_no":
                    "",

                "review_notes":
                    "",
            }
        )

    pd.DataFrame(
        records
    ).to_csv(
        path,
        index=False,
    )


def read_final_human_scores(
    template_path: Path,
) -> dict[str, dict[str, float]]:
    """Aggregate completed final human-review scores."""

    if not template_path.exists():
        return {}

    human = pd.read_csv(
        template_path
    )

    required_columns = {
        "evaluation_variant",
        *RATING_COLUMNS,
    }

    if not required_columns.issubset(
        human.columns
    ):
        return {}

    numeric = human[
        RATING_COLUMNS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    human = pd.concat(
        [
            human.drop(
                columns=RATING_COLUMNS
            ),
            numeric,
        ],
        axis=1,
    )

    results: dict[
        str,
        dict[
            str,
            float,
        ],
    ] = {}

    for variant, group in human.groupby(
        "evaluation_variant"
    ):

        style_results: dict[
            str,
            float,
        ] = {}

        for column in RATING_COLUMNS:

            values = (
                group[
                    column
                ]
                .dropna()
            )

            if not values.empty:

                style_results[
                    f"human_{column}_mean"
                ] = float(
                    values.mean()
                )

        all_values = group[
            RATING_COLUMNS
        ].to_numpy(
            dtype=float
        )

        if np.isfinite(
            all_values
        ).any():

            style_results[
                "human_overall_mean"
            ] = float(
                np.nanmean(
                    all_values
                )
            )

        # Hallucination rate
        if (
            "hallucination_found_yes_no"
            in group.columns
        ):

            hallucination = (
                group[
                    "hallucination_found_yes_no"
                ]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            answered = hallucination.isin(
                [
                    "yes",
                    "no",
                ]
            )

            if answered.any():

                style_results[
                    "human_hallucination_rate"
                ] = float(
                    (
                        hallucination[
                            answered
                        ]
                        == "yes"
                    ).mean()
                )

        if style_results:
            results[
                str(
                    variant
                )
            ] = style_results

    return results


# ============================================================
# Charts
# ============================================================


def make_metric_chart(
    comparison: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Create report-ready comparison chart."""

    metrics = [
        "mean_red_flag_coverage",
        "mean_archetype_keyword_coverage",
        "mean_source_token_support",
    ]

    labels = [
        "Red-flag coverage",
        "Archetype coverage",
        "Source support",
    ]

    available = [
        metric
        for metric in metrics
        if metric
        in comparison.columns
    ]

    if not available:
        return

    chart_data = (
        comparison
        .set_index(
            "evaluation_variant"
        )[available]
        .T
    )

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 5.5),
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Score (0–1)"
    )

    ax.set_title(
        "Fraud-summary grounding metrics"
    )

    ax.set_xticklabels(
        [
            labels[
                metrics.index(
                    metric
                )
            ]
            for metric
            in available
        ],
        rotation=20,
        ha="right",
    )

    ax.legend(
        title=(
            "Evaluation variant"
        )
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "final_system_comparison_metrics.png",
        dpi=180,
    )

    plt.close()


def make_length_chart(
    frames:
        dict[
            str,
            pd.DataFrame,
        ],
    figures_dir: Path,
) -> None:
    """Compare summary-length distributions."""

    if not frames:
        return

    data = [
        frame[
            "summary_words"
        ].to_numpy()
        for frame
        in frames.values()
    ]

    labels = list(
        frames.keys()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.boxplot(
        data,
        labels=labels,
    )

    ax.set_ylabel(
        "Summary length (words)"
    )

    ax.set_title(
        "Summary-length distribution"
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "final_summary_length_distribution.png",
        dpi=180,
    )

    plt.close()


def make_guardrail_chart(
    guardrail:
        dict[
            str,
            Any,
        ],
    figures_dir: Path,
) -> None:
    """Visualize accepted versus fallback outputs."""

    accepted = int(
        guardrail.get(
            "valid_generation_count",
            0,
        )
    )

    fallback = int(
        guardrail.get(
            "fallback_count",
            0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 4.5)
    )

    ax.bar(
        [
            "Accepted FLAN-T5",
            "Fallback",
        ],
        [
            accepted,
            fallback,
        ],
    )

    ax.set_ylabel(
        "Number of samples"
    )

    ax.set_title(
        "Structured generation guardrail behavior"
    )

    plt.tight_layout()

    plt.savefig(
        figures_dir
        / "guardrail_fallback_counts.png",
        dpi=180,
    )

    plt.close()


# ============================================================
# Main evaluation
# ============================================================


def main() -> int:
    """Run complete Milestone 5 evaluation."""

    args = parse_args()

    cfg = load_yaml(
        PROJECT_ROOT
        / args.config
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    results_dir = ensure_directory(
        PROJECT_ROOT
        / cfg[
            "results"
        ][
            "directory"
        ]
    )

    experiments_dir = (
        PROJECT_ROOT
        / cfg[
            "results"
        ][
            "experiments_directory"
        ]
    )

    figures_dir = ensure_directory(
        PROJECT_ROOT
        / cfg[
            "results"
        ][
            "figures_directory"
        ]
    )

    # Keep old human template untouched.
    original_template_path = (
        PROJECT_ROOT
        / cfg[
            "results"
        ][
            "human_template_file"
        ]
    )

    final_human_template_path = (
        original_template_path.with_name(
            "human_evaluation_final_template.csv"
        )
    )

    # --------------------------------------------------------
    # Load experiment outputs
    # --------------------------------------------------------

    rows_by_style: dict[
        str,
        list[
            dict[
                str,
                Any,
            ]
        ],
    ] = {}

    for style in cfg[
        "experiment"
    ][
        "prompt_styles"
    ]:

        rows = read_jsonl(
            experiments_dir
            / f"{style}_prompt.jsonl"
        )

        rows_by_style[
            style
        ] = rows

    baseline_rows = rows_by_style.get(
        "baseline",
        [],
    )

    structured_rows = rows_by_style.get(
        "structured",
        [],
    )

    # --------------------------------------------------------
    # 1. Raw FLAN-T5 model evaluation
    # --------------------------------------------------------

    baseline_raw_frame, baseline_raw_metrics = (
        score_rows(
            baseline_rows,
            summary_field=(
                "generated_summary"
            ),
            evaluation_variant=(
                "baseline_raw"
            ),
        )
    )

    structured_raw_frame, structured_raw_metrics = (
        score_rows(
            structured_rows,
            summary_field=(
                "generated_summary"
            ),
            evaluation_variant=(
                "structured_raw"
            ),
        )
    )

    # --------------------------------------------------------
    # 2. Validated/fallback system evaluation
    # --------------------------------------------------------

    baseline_validated_frame, baseline_validated_metrics = (
        score_rows(
            baseline_rows,
            summary_field=(
                "validated_summary"
            ),
            evaluation_variant=(
                "baseline_validated"
            ),
        )
    )

    structured_validated_frame, structured_validated_metrics = (
        score_rows(
            structured_rows,
            summary_field=(
                "validated_summary"
            ),
            evaluation_variant=(
                "structured_validated"
            ),
        )
    )

    # --------------------------------------------------------
    # Save sample-level metrics
    # --------------------------------------------------------

    baseline_raw_frame.to_csv(
        results_dir
        / "baseline_sample_metrics.csv",
        index=False,
    )

    structured_raw_frame.to_csv(
        results_dir
        / "structured_sample_metrics.csv",
        index=False,
    )

    baseline_validated_frame.to_csv(
        results_dir
        / "baseline_validated_sample_metrics.csv",
        index=False,
    )

    structured_validated_frame.to_csv(
        results_dir
        / "structured_validated_sample_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Guardrail evaluation
    # --------------------------------------------------------

    baseline_guardrail = (
        evaluate_guardrail(
            baseline_rows
        )
    )

    structured_guardrail = (
        evaluate_guardrail(
            structured_rows
        )
    )

    # --------------------------------------------------------
    # 4. Build comparison dataframe
    # --------------------------------------------------------

    comparison_rows = [
        {
            "evaluation_variant":
                "baseline_raw",
            "prompt_style":
                "baseline",
            **baseline_raw_metrics,
        },
        {
            "evaluation_variant":
                "structured_raw",
            "prompt_style":
                "structured",
            **structured_raw_metrics,
        },
        {
            "evaluation_variant":
                "baseline_validated",
            "prompt_style":
                "baseline",
            **baseline_validated_metrics,
        },
        {
            "evaluation_variant":
                "structured_validated",
            "prompt_style":
                "structured",
            **structured_validated_metrics,
        },
    ]

    comparison = pd.DataFrame(
        comparison_rows
    )

    # Main comparison file already used by project.
    comparison_path = (
        PROJECT_ROOT
        / cfg[
            "results"
        ][
            "comparison_file"
        ]
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # 5. Create final human-evaluation template
    # --------------------------------------------------------

    if not final_human_template_path.exists():

        create_final_human_template(
            rows_by_style,
            final_human_template_path,
        )

        print(
            "Created final human-review template:"
        )

        print(
            final_human_template_path
        )

    else:

        print(
            "Preserving existing final human-review template:"
        )

        print(
            final_human_template_path
        )

    human_scores = (
        read_final_human_scores(
            final_human_template_path
        )
    )

    # --------------------------------------------------------
    # 6. Complete metrics JSON
    # --------------------------------------------------------

    final_metrics: dict[
        str,
        Any,
    ] = {

        "raw_model": {
            "baseline":
                baseline_raw_metrics,

            "structured":
                structured_raw_metrics,
        },

        "validated_summary": {
            "baseline":
                baseline_validated_metrics,

            "structured":
                structured_validated_metrics,
        },

        "guardrail": {
            "baseline":
                baseline_guardrail,

            "structured":
                structured_guardrail,
        },

        "human_final_output":
            human_scores,
    }

    metrics_path = (
        PROJECT_ROOT
        / cfg[
            "results"
        ][
            "metrics_file"
        ]
    )

    write_json(
        metrics_path,
        final_metrics,
    )

    # --------------------------------------------------------
    # 7. Save guardrail table
    # --------------------------------------------------------

    guardrail_table = pd.DataFrame(
        [
            {
                "prompt_style":
                    "baseline",

                "samples":
                    baseline_guardrail[
                        "samples"
                    ],

                "valid_generation_count":
                    baseline_guardrail[
                        "valid_generation_count"
                    ],

                "valid_generation_rate":
                    baseline_guardrail[
                        "valid_generation_rate"
                    ],

                "fallback_count":
                    baseline_guardrail[
                        "fallback_count"
                    ],

                "fallback_rate":
                    baseline_guardrail[
                        "fallback_rate"
                    ],
            },

            {
                "prompt_style":
                    "structured",

                "samples":
                    structured_guardrail[
                        "samples"
                    ],

                "valid_generation_count":
                    structured_guardrail[
                        "valid_generation_count"
                    ],

                "valid_generation_rate":
                    structured_guardrail[
                        "valid_generation_rate"
                    ],

                "fallback_count":
                    structured_guardrail[
                        "fallback_count"
                    ],

                "fallback_rate":
                    structured_guardrail[
                        "fallback_rate"
                    ],
            },
        ]
    )

    guardrail_table.to_csv(
        results_dir
        / "guardrail_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Generate report figures
    # --------------------------------------------------------

    make_metric_chart(
        comparison,
        figures_dir,
    )

    make_length_chart(
        {
            "Baseline raw":
                baseline_raw_frame,

            "Structured raw":
                structured_raw_frame,

            "Structured validated":
                structured_validated_frame,
        },
        figures_dir,
    )

    make_guardrail_chart(
        structured_guardrail,
        figures_dir,
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        "\nAutomatic comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nStructured guardrail:"
    )

    print(
        json.dumps(
            structured_guardrail,
            indent=2,
        )
    )

    print(
        "\nMetrics JSON:"
    )

    print(
        metrics_path
    )

    print(
        "\nComparison CSV:"
    )

    print(
        comparison_path
    )

    print(
        "\nGuardrail CSV:"
    )

    print(
        results_dir
        / "guardrail_metrics.csv"
    )

    print(
        "\nFigures:"
    )

    print(
        figures_dir
    )

    print(
        "\nFinal human evaluation template:"
    )

    print(
        final_human_template_path
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )