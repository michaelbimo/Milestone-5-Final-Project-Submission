# CFPB Generative Fraud Intelligence — End-to-End Model Pipeline

This repository demonstrates an executable end-to-end NLP system for CFPB consumer complaint
narratives. It loads the cleaned dataset from Milestone 3, constructs transparent fraud signals,
runs a pretrained FLAN-T5 sequence-to-sequence model, and saves 10 analyst-oriented preliminary
summaries under `outputs/`.

## Objective

The system turns an unstructured complaint narrative into a concise fraud-intelligence summary.
Each model prompt includes:

1. the cleaned complaint narrative;
2. a provisional scam archetype;
3. rule-based red flags;
4. a transparent risk level.

The provisional archetype is rule based in this milestone so that the complete generator pipeline
is runnable before the final DistilBERT/RoBERTa classifier is accepted. The runner automatically
switches to `models/generator/` when a valid fine-tuned FLAN-T5 checkpoint is available.

## Required single-command behavior

From the repository root:

```bash
python src/model_runner.py
```

The command:

- loads `data/processed/complaints_clean.csv.gz`;
- selects 10 deterministic, representative complaint samples;
- loads `google/flan-t5-small` or a local fine-tuned checkpoint;
- generates summaries with deterministic beam search;
- saves text, JSONL, metadata, and an output description in `outputs/`.

## Repository structure

```text
project-root/
├── src/
│   ├── data_loader.py
│   ├── feature_builder.py
│   ├── model_runner.py
│   ├── preprocessing.py
│   └── red_flags.py
├── utils/
│   └── helpers.py
├── configs/
│   └── model_config.yaml
├── outputs/
│   ├── README.md
│   ├── prepared_inputs.jsonl
│   ├── samples.txt                 # generated after a full run
│   ├── samples.jsonl               # generated after a full run
│   ├── run_metadata.json
│   └── output_description.md
├── notebooks/
│   └── demo_pipeline_colab.ipynb
├── data/
│   └── processed/
│       └── complaints_clean.csv.gz
├── Dockerfile
├── requirements.txt
├── requirements-full.txt
└── README.md
```

## Google Colab setup

1. Push this project to your GitHub repository.
2. Open `notebooks/demo_pipeline_colab.ipynb` in Colab.
3. Change the runtime to a **T4 GPU**.
4. Set `REPO_URL`, `BRANCH`, and `PROJECT_FOLDER` in the notebook.
5. For a private repository, add a Colab secret named `GITHUB_TOKEN`.
6. Run all cells.

The notebook first runs a no-model validation and then executes:

```bash
python src/model_runner.py
```

## Local setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/model_runner.py
```

To validate the data pipeline without downloading FLAN-T5:

```bash
python src/model_runner.py --prepare-only
```

To change the sample count:

```bash
python src/model_runner.py --num-samples 5
```

## Configuration

Edit `configs/model_config.yaml` to change:

- pretrained or local model source;
- number of representative samples;
- batch size and decoding parameters;
- data and output paths;
- CPU/GPU selection.

The default model is `google/flan-t5-small`, selected because it is instruction-tuned, matches the
text-to-text task, and is practical for a standard Colab GPU. Decoding uses four beams with
sampling disabled to improve reproducibility.

## Preliminary results

The pipeline is configured to generate 10 preliminary summaries. Each output records the source
complaint ID, product, provisional archetype, extracted red flags, risk level, and generated
summary. After the full Colab run, the exact samples are available in `outputs/samples.txt` and
`outputs/samples.jsonl`.

This is a zero-shot preliminary system unless `models/generator/` contains a fine-tuned checkpoint.
Expected limitations include omitted details, awkward phrasing, and occasional mismatch between
the generated wording and rule-based signals. Outputs must therefore be reviewed by a human and
must not be treated as final fraud determinations.

## Reproducing outputs

Reproducibility controls include:

- fixed seed (`42`);
- deterministic sample selection;
- deterministic beam-search generation (`do_sample: false`);
- versioned YAML configuration;
- structured run metadata;
- the exact prompts saved to `outputs/prepared_inputs.jsonl`.

## Known issues and limitations

- The CFPB dataset does not contain analyst-written reference summaries.
- Provisional archetypes are rule based until the final classifier is integrated.
- FLAN-T5 may paraphrase or omit a red flag; generated text is not a fact checker.
- The first run requires internet access to download the Hugging Face checkpoint.
- Colab runtimes are temporary, so download or commit the small output files before disconnecting.

## Docker

Build and run from the project root:

```bash
docker build -t cfpb-fraud-intelligence .
docker run --rm -v "$PWD/outputs:/app/outputs" cfpb-fraud-intelligence
```

The default container uses CPU. GPU execution requires an NVIDIA-compatible container runtime.

## Full experimental environment

`requirements.txt` contains only the dependencies needed for the end-to-end runner. Use
`requirements-full.txt` for BERTopic discovery, classifier training, FLAN-T5 fine-tuning, and the
broader Milestone 3 notebooks.

---

## Milestone 5 — Final evaluation workflow

Milestone 5 adds a reproducible experiment and evaluation layer to the end-to-end generator.
The final experiment compares two prompt conditions on the **same deterministic CFPB sample**:

1. **Baseline prompt** — asks FLAN-T5 to summarize the complaint without structured fraud signals.
2. **Structured prompt** — supplies the complaint, provisional archetype, extracted red flags, and
   risk level before requesting the analyst-oriented summary.

This comparison isolates the contribution of the structured prompting strategy while holding the
sample set, model, seed, and decoding settings constant.

### Run the final experiment in Google Colab

From the repository root:

```bash
python src/experiment_runner.py --num-samples 30
python src/evaluate.py
```

For a quick no-model validation:

```bash
python src/experiment_runner.py --num-samples 10 --prepare-only
```

### Final evaluation artifacts

After a full run, the repository produces:

```text
results/
├── experiments/
│   ├── baseline_prompt.jsonl
│   ├── structured_prompt.jsonl
│   └── experiment_metadata.json
├── figures/
│   ├── prompt_comparison_metrics.png
│   └── summary_length_distribution.png
├── baseline_sample_metrics.csv
├── structured_sample_metrics.csv
├── human_evaluation_template.csv
├── model_comparison.csv
└── metrics.json
```

### Automatic metrics

Because the CFPB corpus does not contain analyst-written reference summaries, the final automatic
evaluation emphasizes transparent grounding proxies rather than claiming reference-based accuracy:

- non-empty output rate;
- average summary length;
- risk-level mention rate;
- red-flag coverage;
- provisional-archetype keyword coverage;
- source-token support;
- unsupported-number rate (a hallucination warning proxy).

These metrics are complemented by a human rubric with **1–5 ratings for factuality, relevance,
clarity, and analyst usefulness**. The evaluator creates `results/human_evaluation_template.csv`.
After ratings are entered, rerun `python src/evaluate.py` to include human-score averages in the
final comparison.

### Important interpretation note

The lexical grounding metrics are diagnostic proxies, not substitutes for expert factual review.
A valid paraphrase can lower source-token overlap, while a fluent but incorrect sentence can still
look lexically similar to the complaint. The final report should therefore interpret automatic and
human evaluation together.
