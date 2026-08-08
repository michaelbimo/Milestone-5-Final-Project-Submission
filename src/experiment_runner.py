"""Run the Milestone 5 prompt-comparison experiment on a fixed CFPB sample.

From the repository root:
    python src/experiment_runner.py

The script compares a simple summarization prompt against the structured
fraud-intelligence prompt while holding the sample set, model, and decoding
settings constant.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Make project modules importable when this file is run directly:
# python src/experiment_runner.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import yaml

from src.data_loader import (
    load_preprocessed_dataset,
    select_representative_samples,
)
from src.feature_builder import (
    build_generation_prompt,
    build_structured_record,
)
from src.model_runner import (
    generate_summaries,
    resolve_device,
)
from utils.helpers import (
    ensure_directory,
    set_global_seed,
    utc_timestamp,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare baseline and structured FLAN-T5 prompts."
    )

    parser.add_argument(
        "--config",
        default="configs/eval_config.yaml",
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
        "--model-name",
        default=None,
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Create identical experiment inputs/prompts "
            "without downloading the model."
        ),
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
        )

    return value


def build_baseline_prompt(
    record: dict[str, Any],
) -> str:
    """Create the simple baseline summarization prompt."""

    return (
        "Summarize the following consumer complaint in two concise "
        "sentences for an analyst. "
        "Do not invent facts.\n"
        f"Complaint: {record['clean_text']}\n"
        "Summary:"
    )


def build_rows(
    sample_frame,
    data_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert the selected dataframe rows into structured records."""

    rows: list[dict[str, Any]] = []

    id_col = data_cfg["id_column"]
    text_col = data_cfg["text_column"]

    for sample_number, (_, row) in enumerate(
        sample_frame.iterrows(),
        start=1,
    ):
        structured = build_structured_record(
            row=row,
            text_column=text_col,
            max_input_words=int(
                data_cfg["max_input_words"]
            ),
        )

        rows.append(
            {
                "sample_number": sample_number,
                "complaint_id": (
                    str(row[id_col])
                    if id_col in row
                    else f"sample-{sample_number}"
                ),
                "product": str(
                    row.get("Product", "Unknown")
                ),
                "sub_product": str(
                    row.get("Sub-product", "Unknown")
                ),
                **structured,
            }
        )

    return rows


def build_analyst_output(
    row: dict[str, Any],
    generated_summary: str,
) -> str:
    """Create the final analyst-facing fraud intelligence output.

    Structured fields such as risk level, archetype, and red flags are
    produced deterministically by the pipeline. FLAN-T5 is responsible
    only for the natural-language summary.

    Keeping these two layers separate makes the system more reliable and
    allows the raw model generation to be evaluated independently.
    """

    red_flags = row.get("red_flags", [])

    if red_flags:
        red_flag_text = "; ".join(
            str(flag) for flag in red_flags
        )
    else:
        red_flag_text = (
            "No explicit red flags detected"
        )

    return (
        f"Risk level: {row['risk_level']}\n"
        f"Suspected pattern: {row['archetype']}\n"
        f"Red flags: {red_flag_text}\n"
        f"Summary: {generated_summary.strip()}"
    )


def main() -> int:
    """Run the baseline-vs-structured experiment."""

    args = parse_args()

    # ---------------------------------------------------------
    # Load configurations
    # ---------------------------------------------------------

    eval_cfg = load_yaml(
        PROJECT_ROOT / args.config
    )

    model_cfg_file = load_yaml(
        PROJECT_ROOT / args.model_config
    )

    seed = int(
        eval_cfg["project"]["seed"]
    )

    set_global_seed(seed)

    # ---------------------------------------------------------
    # Load the preprocessed CFPB dataset
    # ---------------------------------------------------------

    data_cfg = model_cfg_file["data"]

    frame = load_preprocessed_dataset(
        PROJECT_ROOT / data_cfg["path"],
        text_column=data_cfg["text_column"],
        raw_text_column=data_cfg["raw_text_column"],
        minimum_words=int(
            data_cfg["minimum_words"]
        ),
    )

    # ---------------------------------------------------------
    # Select a fixed, reproducible evaluation sample
    # ---------------------------------------------------------

    n_samples = (
        args.num_samples
        or int(
            eval_cfg["experiment"][
                "number_of_samples"
            ]
        )
    )

    sample_frame = (
        select_representative_samples(
            frame,
            number_of_samples=n_samples,
            seed=seed,
            group_column=data_cfg[
                "group_column"
            ],
            id_column=data_cfg[
                "id_column"
            ],
        )
    )

    # Build the same underlying records once.
    # Both prompt styles use identical complaints.
    base_rows = build_rows(
        sample_frame,
        data_cfg,
    )

    # ---------------------------------------------------------
    # Experiment configuration
    # ---------------------------------------------------------

    results_dir = ensure_directory(
        PROJECT_ROOT
        / eval_cfg["results"][
            "experiments_directory"
        ]
    )

    model_name = (
        args.model_name
        or str(
            eval_cfg["model"][
                "pretrained_name"
            ]
        )
    )

    model_settings = dict(
        eval_cfg["model"]
    )

    experiment_meta = {
        "started_at_utc": utc_timestamp(),
        "model": model_name,
        "number_of_samples": len(
            base_rows
        ),
        "seed": seed,
        "prepare_only": bool(
            args.prepare_only
        ),
    }

    # ---------------------------------------------------------
    # Run each prompt condition
    # ---------------------------------------------------------

    for prompt_style in eval_cfg[
        "experiment"
    ]["prompt_styles"]:

        # Copy records so each experimental
        # condition remains independent.
        rows = [
            dict(row)
            for row in base_rows
        ]

        # -----------------------------------------------------
        # Build prompts
        # -----------------------------------------------------

        if prompt_style == "baseline":

            prompts = [
                build_baseline_prompt(row)
                for row in rows
            ]

        elif prompt_style == "structured":

            prompts = [
                build_generation_prompt(row)
                for row in rows
            ]

        else:
            raise ValueError(
                "Unsupported prompt style: "
                f"{prompt_style}"
            )

        # Save the prompt used for every record.
        for row, prompt in zip(
            rows,
            prompts,
            strict=True,
        ):
            row["prompt_style"] = (
                prompt_style
            )
            row["prompt"] = prompt

        # -----------------------------------------------------
        # Model inference
        # -----------------------------------------------------

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

            # -------------------------------------------------
            # Store raw generation and final analyst output
            # -------------------------------------------------

            for row, summary in zip(
                rows,
                summaries,
                strict=True,
            ):

                # Keep the original FLAN-T5 output.
                #
                # This field should be used for evaluation of
                # the generative model itself.
                row["generated_summary"] = (
                    summary.strip()
                )

                if prompt_style == "structured":

                    # The final system output combines the
                    # deterministic fraud signals with the
                    # generated narrative summary.
                    row["analyst_output"] = (
                        build_analyst_output(
                            row,
                            summary,
                        )
                    )

                else:

                    # The baseline has no deterministic
                    # analyst-intelligence layer.
                    row["analyst_output"] = (
                        summary.strip()
                    )

        else:

            device = "not loaded"

        # -----------------------------------------------------
        # Save this experimental condition
        # -----------------------------------------------------

        output_path = (
            results_dir
            / f"{prompt_style}_prompt.jsonl"
        )

        write_jsonl(
            output_path,
            rows,
        )

        print(
            f"Saved {prompt_style} "
            f"experiment: {output_path}"
        )

    # ---------------------------------------------------------
    # Save experiment metadata
    # ---------------------------------------------------------

    experiment_meta["device"] = device

    experiment_meta[
        "finished_at_utc"
    ] = utc_timestamp()

    write_json(
        results_dir
        / "experiment_metadata.json",
        experiment_meta,
    )

    print(
        json.dumps(
            experiment_meta,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
