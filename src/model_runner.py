"""Single-command end-to-end inference runner for CFPB fraud intelligence.

From the repository root, run:
    python src/model_runner.py
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

import numpy as np
import yaml

from src.data_loader import load_preprocessed_dataset, select_representative_samples
from src.feature_builder import build_generation_prompt, build_structured_record
from utils.helpers import ensure_directory, set_global_seed, utc_timestamp, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CFPB fraud-intelligence summaries.")
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Configuration path relative to the repository root.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Override sample count.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate data and save prepared prompts without downloading a model.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")
    return config


def resolve_model_source(config: dict[str, Any]) -> str:
    model_cfg = config["model"]
    local_path = PROJECT_ROOT / str(model_cfg.get("local_finetuned_path", ""))
    if model_cfg.get("prefer_local_finetuned_model", True) and local_path.is_dir():
        required_markers = [local_path / "config.json"]
        if all(marker.exists() for marker in required_markers):
            return str(local_path)
    return str(model_cfg["pretrained_name"])


def resolve_device(requested: str) -> str:
    import torch

    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no GPU is available.")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    return requested


def generate_summaries(
    prompts: list[str],
    model_source: str,
    device: str,
    model_config: dict[str, Any],
) -> list[str]:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing model dependencies. Run: pip install -r requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, torch_dtype=dtype)
    model.to(device)
    model.eval()

    batch_size = int(model_config["batch_size"])
    generated_text: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_config["max_input_tokens"]),
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=int(model_config["max_new_tokens"]),
                num_beams=int(model_config["num_beams"]),
                do_sample=False,
                early_stopping=True,
            )
        generated_text.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

    return [text.strip() for text in generated_text]


def build_output_rows(sample_frame, data_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    id_col = data_cfg["id_column"]
    text_col = data_cfg["text_column"]
    for sample_number, (_, row) in enumerate(sample_frame.iterrows(), start=1):
        structured = build_structured_record(
            row=row,
            text_column=text_col,
            max_input_words=int(data_cfg["max_input_words"]),
        )
        rows.append(
            {
                "sample_number": sample_number,
                "complaint_id": str(row[id_col]) if id_col in row else f"sample-{sample_number}",
                "product": str(row.get("Product", "Unknown")),
                "sub_product": str(row.get("Sub-product", "Unknown")),
                **structured,
                "prompt": build_generation_prompt(structured),
            }
        )
    return rows


def render_text_report(rows: list[dict[str, Any]]) -> str:
    blocks = ["CFPB GENERATIVE FRAUD INTELLIGENCE — PRELIMINARY SAMPLES", "=" * 68, ""]
    for row in rows:
        flags = "; ".join(row["red_flags"]) if row["red_flags"] else "None extracted"
        blocks.extend(
            [
                f"SAMPLE {row['sample_number']}",
                f"Complaint ID: {row['complaint_id']}",
                f"Product / Sub-product: {row['product']} / {row['sub_product']}",
                f"Provisional archetype: {row['archetype']}",
                f"Red flags: {flags}",
                f"Risk: {row['risk_level']} (score {row['risk_score']})",
                "Generated summary:",
                row.get("generated_summary", "[not generated: prepare-only mode]"),
                "-" * 68,
                "",
            ]
        )
    return "\n".join(blocks).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    set_global_seed(seed)

    data_cfg = config["data"]
    dataset_path = PROJECT_ROOT / data_cfg["path"]
    frame = load_preprocessed_dataset(
        path=dataset_path,
        text_column=data_cfg["text_column"],
        raw_text_column=data_cfg["raw_text_column"],
        minimum_words=int(data_cfg["minimum_words"]),
    )

    number_of_samples = args.num_samples or int(config["inference"]["number_of_samples"])
    sample_frame = select_representative_samples(
        frame=frame,
        number_of_samples=number_of_samples,
        seed=seed,
        group_column=data_cfg["group_column"],
        id_column=data_cfg["id_column"],
    )
    rows = build_output_rows(sample_frame, data_cfg)

    output_cfg = config["outputs"]
    output_dir = ensure_directory(PROJECT_ROOT / output_cfg["directory"])
    prepared_path = output_dir / "prepared_inputs.jsonl"
    write_jsonl(prepared_path, rows)

    model_source = resolve_model_source(config)
    device = "not loaded"
    started_at = utc_timestamp()

    if not args.prepare_only:
        device = resolve_device(str(config["model"].get("device", "auto")))
        summaries = generate_summaries(
            prompts=[row["prompt"] for row in rows],
            model_source=model_source,
            device=device,
            model_config=config["model"],
        )
        for row, summary in zip(rows, summaries, strict=True):
            row["generated_summary"] = summary

        text_path = output_dir / output_cfg["text_file"]
        text_path.write_text(render_text_report(rows), encoding="utf-8")
        write_jsonl(output_dir / output_cfg["jsonl_file"], rows)

        non_empty = [bool(row["generated_summary"].strip()) for row in rows]
        word_counts = [len(row["generated_summary"].split()) for row in rows]
        risk_hits = [row["risk_level"].lower() in row["generated_summary"].lower() for row in rows]
        quality_checks = {
            "non_empty_output_rate": float(np.mean(non_empty)),
            "average_summary_words": float(np.mean(word_counts)),
            "risk_level_mention_rate": float(np.mean(risk_hits)),
        }
    else:
        quality_checks = {"status": "prepare-only; model inference was skipped"}

    metadata = {
        "started_at_utc": started_at,
        "finished_at_utc": utc_timestamp(),
        "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
        "dataset_rows_loaded": int(len(frame)),
        "samples_selected": int(len(rows)),
        "model_source": model_source,
        "device": device,
        "prepare_only": bool(args.prepare_only),
        "quality_checks": quality_checks,
    }
    write_json(output_dir / output_cfg["metadata_file"], metadata)

    description = (
        "# Preliminary output description\n\n"
        f"The pipeline selected **{len(rows)}** representative CFPB complaint narratives and "
        f"used **{model_source}** with deterministic beam-search decoding. Each prompt included "
        "the cleaned complaint, a provisional transparent archetype, extracted red flags, and a "
        "rule-based risk level. The outputs are preliminary and require human review because the "
        "generator is not an independent fact checker and the provisional archetypes are not the "
        "final trained classifier labels.\n\n"
        f"Run mode: **{'prepare only' if args.prepare_only else 'full generation'}**.\n"
    )
    (output_dir / output_cfg["description_file"]).write_text(description, encoding="utf-8")

    print(f"Loaded {len(frame):,} preprocessed complaints.")
    print(f"Selected {len(rows)} representative samples.")
    if args.prepare_only:
        print(f"Prepared prompts saved to: {prepared_path}")
    else:
        print(f"Generated samples saved to: {output_dir / output_cfg['text_file']}")
        print(json.dumps(quality_checks, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
