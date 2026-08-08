# Outputs

Run the following command from the repository root:

```bash
python src/model_runner.py
```

A successful full run creates:

- `samples.txt` — 10 readable generated fraud-intelligence samples
- `samples.jsonl` — the same records in structured JSON Lines format
- `run_metadata.json` — model, device, sample count, and basic checks
- `output_description.md` — a concise description of what was generated
- `prepared_inputs.jsonl` — the structured prompts used for inference

The repository includes prepared inputs for reproducibility. Generated model outputs must be
produced in Google Colab before final submission.
