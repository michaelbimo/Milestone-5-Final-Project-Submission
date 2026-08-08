"""Run the frozen Milestone 5 pipeline on an unseen final test set.

This script:

1. Loads the same preprocessed CFPB dataset used during development.
2. Explicitly excludes all complaint IDs used for development/tuning.
3. Selects a new deterministic 30-complaint sample.
4. Runs the frozen baseline and structured pipelines.
5. Validates generated summaries.
6. Applies the extractive fallback where required.
7. Writes all results into results/final_test/.

Run from the repository root:

    python src/final_test_runner.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================
# Imports from the frozen project pipeline
# ============================================================

from src.data_loader import (
    load_preprocessed_dataset,
    select_representative_samples,
)

from src.experiment_runner import (
    build_analyst_output,
    build_baseline_prompt,
    build_rows,
    load_yaml,
)

from src.feature_builder import (
    build_generation_prompt,
)

from src.model_runner import (
    generate_summaries,
    resolve_device,
)

from src.output_validation import (
    build_extractive_fallback,
    validate_generated_summary,
)

from utils.helpers import (
    ensure_directory,
    set_global_seed,
    utc_timestamp,
    write_json,
    write_jsonl,
)


# ============================================================
# Arguments
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen CFPB fraud-intelligence "
            "pipeline on an unseen final test set."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/final_eval_config.yaml",
    )

    parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Select and save the unseen final test set "
            "without loading FLAN-T5."
        ),
    )

    return parser.parse_args()


# ============================================================
# Development-set exclusion
# ============================================================


def load_excluded_ids(
    path: Path,
) -> set[str]:
    """Load complaint IDs that must not appear in the final test."""

    if not path.exists():

        raise FileNotFoundError(
            "Development complaint ID file not found: "
            f"{path}"
        )

    excluded: set[str] = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        for line in handle:

            value = line.strip()

            if value:
                excluded.add(
                    value
                )

    if not excluded:

        raise ValueError(
            "Development complaint ID file is empty."
        )

    return excluded


# ============================================================
# Overlap verification
# ============================================================


def verify_no_overlap(
    selected_ids: set[str],
    excluded_ids: set[str],
) -> None:
    """Fail immediately if final-test complaints overlap development."""

    overlap = (
        selected_ids
        & excluded_ids
    )

    if overlap:

        raise RuntimeError(
            "FINAL TEST CONTAMINATION DETECTED.\n"
            "The following complaint IDs were already used "
            "during development:\n"
            + "\n".join(
                sorted(overlap)
            )
        )


# ============================================================
# Main
# ============================================================


def main() -> int:
    """Run the unseen final-test experiment."""

    args = parse_args()

    # --------------------------------------------------------
    # Load configurations
    # --------------------------------------------------------

    eval_cfg = load_yaml(
        PROJECT_ROOT
        / args.config
    )

    model_cfg_file = load_yaml(
        PROJECT_ROOT
        / args.model_config
    )

    seed = int(
        eval_cfg[
            "project"
        ][
            "seed"
        ]
    )

    set_global_seed(
        seed
    )

    # --------------------------------------------------------
    # Read model/data configuration
    # --------------------------------------------------------

    data_cfg = model_cfg_file[
        "data"
    ]

    id_column = data_cfg[
        "id_column"
    ]

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "LOADING CFPB DATASET"
    )

    print(
        "=" * 80
    )

    frame = load_preprocessed_dataset(
        PROJECT_ROOT
        / data_cfg[
            "path"
        ],
        text_column=data_cfg[
            "text_column"
        ],
        raw_text_column=data_cfg[
            "raw_text_column"
        ],
        minimum_words=int(
            data_cfg[
                "minimum_words"
            ]
        ),
    )

    print(
        f"Usable complaints before exclusion: {len(frame):,}"
    )

    # --------------------------------------------------------
    # Load development IDs
    # --------------------------------------------------------

    exclusion_file = (
        PROJECT_ROOT
        / eval_cfg[
            "experiment"
        ][
            "exclude_complaint_ids_file"
        ]
    )

    excluded_ids = load_excluded_ids(
        exclusion_file
    )

    print(
        f"Development complaint IDs to exclude: "
        f"{len(excluded_ids)}"
    )

    # --------------------------------------------------------
    # Verify complaint ID column exists
    # --------------------------------------------------------

    if id_column not in frame.columns:

        raise ValueError(
            f"Complaint ID column '{id_column}' "
            "was not found in the dataset."
        )

    # Normalize IDs before comparison.
    frame[
        id_column
    ] = (
        frame[
            id_column
        ]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove development complaints
    # --------------------------------------------------------

    development_present = (
        frame[
            id_column
        ]
        .isin(
            excluded_ids
        )
    )

    development_found_count = int(
        development_present.sum()
    )

    print(
        "Development complaints located in dataset: "
        f"{development_found_count}"
    )

    eligible_frame = (
        frame.loc[
            ~development_present
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Eligible complaints after exclusion: "
        f"{len(eligible_frame):,}"
    )

    # --------------------------------------------------------
    # Select final-test size
    # --------------------------------------------------------

    number_of_samples = (
        args.num_samples
        or int(
            eval_cfg[
                "experiment"
            ][
                "number_of_samples"
            ]
        )
    )

    # --------------------------------------------------------
    # Select new representative final test sample
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SELECTING UNSEEN FINAL TEST SET"
    )

    print(
        "=" * 80
    )

    sample_frame = (
        select_representative_samples(
            eligible_frame,
            number_of_samples=number_of_samples,
            seed=seed,
            group_column=data_cfg[
                "group_column"
            ],
            id_column=id_column,
        )
    )

    # --------------------------------------------------------
    # Hard overlap check
    # --------------------------------------------------------

    selected_ids = set(
        sample_frame[
            id_column
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    verify_no_overlap(
        selected_ids,
        excluded_ids,
    )

    if len(
        selected_ids
    ) != len(
        sample_frame
    ):

        raise RuntimeError(
            "Duplicate complaint IDs were selected "
            "for the final test."
        )

    print(
        f"Final test complaints selected: "
        f"{len(sample_frame)}"
    )

    print(
        "Development overlap: 0"
    )

    print(
        f"Final-test seed: {seed}"
    )

    # --------------------------------------------------------
    # Build frozen structured records
    # --------------------------------------------------------

    base_rows = build_rows(
        sample_frame,
        data_cfg,
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    results_dir = ensure_directory(
        PROJECT_ROOT
        / eval_cfg[
            "results"
        ][
            "experiments_directory"
        ]
    )

    final_root = ensure_directory(
        PROJECT_ROOT
        / eval_cfg[
            "results"
        ][
            "directory"
        ]
    )

    # --------------------------------------------------------
    # Save selected final-test IDs BEFORE inference
    # --------------------------------------------------------

    final_ids_path = (
        final_root
        / "final_test_complaint_ids.txt"
    )

    with final_ids_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for complaint_id in (
            sample_frame[
                id_column
            ]
            .astype(str)
            .tolist()
        ):

            handle.write(
                complaint_id
                + "\n"
            )

    print(
        f"Saved final-test IDs: {final_ids_path}"
    )

    # --------------------------------------------------------
    # Save final-test sample table
    # --------------------------------------------------------

    sample_csv_path = (
        final_root
        / "final_test_sample.csv"
    )

    sample_frame.to_csv(
        sample_csv_path,
        index=False,
    )

    print(
        f"Saved final-test sample: {sample_csv_path}"
    )

    # --------------------------------------------------------
    # Model configuration
    # --------------------------------------------------------

    model_name = str(
        eval_cfg[
            "model"
        ][
            "pretrained_name"
        ]
    )

    model_settings = dict(
        eval_cfg[
            "model"
        ]
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    experiment_meta: dict[
        str,
        Any,
    ] = {

        "experiment_type":
            "unseen_final_test",

        "started_at_utc":
            utc_timestamp(),

        "model":
            model_name,

        "number_of_samples":
            len(
                base_rows
            ),

        "seed":
            seed,

        "development_ids_excluded":
            len(
                excluded_ids
            ),

        "development_ids_found_in_dataset":
            development_found_count,

        "development_overlap":
            0,

        "prepare_only":
            bool(
                args.prepare_only
            ),
    }

    # --------------------------------------------------------
    # Run both experiment conditions
    # --------------------------------------------------------

    device = (
        "not loaded"
    )

    for prompt_style in (
        eval_cfg[
            "experiment"
        ][
            "prompt_styles"
        ]
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"RUNNING: {prompt_style.upper()}"
        )

        print(
            "=" * 80
        )

        # Independent copies of the exact same 30 complaints.
        rows = [
            dict(
                row
            )
            for row
            in base_rows
        ]

        # ----------------------------------------------------
        # Build prompts using frozen pipeline
        # ----------------------------------------------------

        if (
            prompt_style
            == "baseline"
        ):

            prompts = [
                build_baseline_prompt(
                    row
                )
                for row
                in rows
            ]

        elif (
            prompt_style
            == "structured"
        ):

            prompts = [
                build_generation_prompt(
                    row
                )
                for row
                in rows
            ]

        else:

            raise ValueError(
                "Unsupported prompt style: "
                f"{prompt_style}"
            )

        # Store exact prompt for reproducibility.
        for row, prompt in zip(
            rows,
            prompts,
            strict=True,
        ):

            row[
                "prompt_style"
            ] = prompt_style

            row[
                "prompt"
            ] = prompt

            row[
                "evaluation_split"
            ] = (
                "unseen_final_test"
            )

        # ----------------------------------------------------
        # Run FLAN-T5
        # ----------------------------------------------------

        if not args.prepare_only:

            device = resolve_device(
                str(
                    model_settings.get(
                        "device",
                        "auto",
                    )
                )
            )

            summaries = generate_summaries(
                prompts=prompts,
                model_source=model_name,
                device=device,
                model_config=model_settings,
            )

            # ------------------------------------------------
            # Frozen validation + fallback logic
            # ------------------------------------------------

            for row, summary in zip(
                rows,
                summaries,
                strict=True,
            ):

                # Raw FLAN-T5 output.
                raw_summary = (
                    summary.strip()
                )

                row[
                    "generated_summary"
                ] = raw_summary

                # Validate.
                (
                    is_valid,
                    validation_issues,
                ) = (
                    validate_generated_summary(
                        raw_summary
                    )
                )

                row[
                    "generation_valid"
                ] = is_valid

                row[
                    "validation_issues"
                ] = (
                    validation_issues
                )

                # Model output or grounded fallback.
                if is_valid:

                    validated_summary = (
                        raw_summary
                    )

                    summary_source = (
                        "flan_t5"
                    )

                else:

                    validated_summary = (
                        build_extractive_fallback(
                            row[
                                "clean_text"
                            ]
                        )
                    )

                    summary_source = (
                        "extractive_fallback"
                    )

                row[
                    "validated_summary"
                ] = (
                    validated_summary
                )

                row[
                    "summary_source"
                ] = (
                    summary_source
                )

                # ------------------------------------------------
                # Final analyst-facing output
                # ------------------------------------------------

                if (
                    prompt_style
                    == "structured"
                ):

                    row[
                        "analyst_output"
                    ] = (
                        build_analyst_output(
                            row,
                            validated_summary,
                        )
                    )

                else:

                    row[
                        "analyst_output"
                    ] = (
                        validated_summary
                    )

        # ----------------------------------------------------
        # Save experiment JSONL
        # ----------------------------------------------------

        output_path = (
            results_dir
            / f"{prompt_style}_prompt.jsonl"
        )

        write_jsonl(
            output_path,
            rows,
        )

        print(
            f"Saved {prompt_style} final test: "
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Final overlap verification from generated rows
    # --------------------------------------------------------

    verify_no_overlap(
        {
            str(
                row[
                    "complaint_id"
                ]
            )
            for row
            in base_rows
        },
        excluded_ids,
    )

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    experiment_meta[
        "device"
    ] = device

    experiment_meta[
        "finished_at_utc"
    ] = (
        utc_timestamp()
    )

    metadata_path = (
        results_dir
        / "experiment_metadata.json"
    )

    write_json(
        metadata_path,
        experiment_meta,
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL TEST COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"Samples: {len(base_rows)}"
    )

    print(
        f"Seed: {seed}"
    )

    print(
        "Development overlap: 0"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Results: {final_root}"
    )

    print(
        "\nMetadata:"
    )

    print(
        json.dumps(
            experiment_meta,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )