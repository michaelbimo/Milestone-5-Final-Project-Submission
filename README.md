# CFPB Generative Fraud Intelligence — End-to-End Model Pipeline

This repository implements an end-to-end NLP pipeline for analyzing CFPB consumer complaint narratives and producing grounded, analyst-oriented fraud intelligence.

The system combines:

- cleaned CFPB complaint narratives;
- FLAN-T5 abstractive summarization;
- deterministic fraud archetype classification;
- rule-based red-flag detection;
- transparent risk scoring;
- generated-output validation;
- extractive fallback for unreliable generations;
- automatic and human evaluation.

The final system was evaluated on a held-out set of **30 previously unseen CFPB complaints with zero overlap with the development set**.

---

## Objective

The goal is to transform an unstructured CFPB consumer complaint into a concise analyst-oriented output containing:

```text
Risk level: ...
Suspected pattern: ...
Red flags: ...
Summary: ...
```

The architecture intentionally separates generative and deterministic components.

FLAN-T5 is responsible for generating the natural-language summary, while fraud archetypes, red flags, and risk scores are produced using transparent rule-based logic.

A validation layer checks the generated summary before it is included in the final output. If generation is unreliable, the system replaces it with a grounded extractive fallback derived directly from the complaint.

This design reduces dependence on unconstrained generation while preserving interpretable fraud-analysis signals.

---

# System Architecture

```text
CFPB Complaint
      │
      ▼
Text Cleaning / Normalization
      │
      ├───────────────────────────────┐
      │                               │
      ▼                               ▼
FLAN-T5 Summary Generation     Rule-Based Fraud Signals
      │                         • Archetype
      ▼                         • Red Flags
Output Validator               • Risk Score
      │
      ├── Valid ──────────────► Generated Summary
      │
      └── Invalid ────────────► Extractive Fallback
                                      │
                                      ▼
                        Final Analyst-Facing Output
```

The validator detects common generation failures including:

- instruction echo;
- redaction-dominated output;
- repeated sentences;
- extremely short outputs.

---

# Repository Structure

```text
project-root/
├── src/
│   ├── data_loader.py
│   ├── feature_builder.py
│   ├── model_runner.py
│   ├── preprocessing.py
│   ├── red_flags.py
│   ├── output_validation.py
│   ├── experiment_runner.py
│   ├── final_test_runner.py
│   ├── evaluation_metrics.py
│   └── evaluate.py
│
├── utils/
│   └── helpers.py
│
├── configs/
│   ├── model_config.yaml
│   ├── eval_config.yaml
│   ├── final_eval_config.yaml
│   └── development_complaint_ids.txt
│
├── data/
│   └── processed/
│       └── complaints_clean.csv.gz
│
├── outputs/
│   ├── prepared_inputs.jsonl
│   ├── samples.txt
│   ├── samples.jsonl
│   ├── run_metadata.json
│   └── output_description.md
│
├── results/
│   ├── experiments/
│   ├── figures/
│   └── final_test/
│       ├── experiments/
│       ├── figures/
│       ├── final_test_complaint_ids.txt
│       ├── final_test_sample.csv
│       ├── model_comparison.csv
│       ├── guardrail_metrics.csv
│       ├── metrics.json
│       └── human_evaluation_final_template.csv
│
├── notebooks/
│   ├── demo_pipeline_colab.ipynb
│   └── final_evaluation_colab.ipynb
│
├── Dockerfile
├── requirements.txt
├── requirements-full.txt
└── README.md
```

---

# Dataset

The pipeline uses the CFPB Consumer Complaint Database.

After preprocessing and validation, the project contains:

```text
25,576 usable complaint narratives
```

Processing includes:

- narrative normalization;
- removal of unusable short records;
- duplicate handling;
- normalization of CFPB redaction markers;
- whitespace cleanup.

The processed dataset is stored at:

```text
data/processed/complaints_clean.csv.gz
```

---

# Model

The default generator is:

```text
google/flan-t5-small
```

FLAN-T5 was selected because it:

- supports instruction-style text-to-text generation;
- can run on standard Google Colab GPU resources;
- provides a reproducible open-source baseline;
- is appropriate for controlled summarization experiments.

Generation uses deterministic beam search rather than stochastic sampling.

The model is used only for natural-language summarization. Fraud classification and risk signals are handled separately by transparent rule-based components.

---

# Fraud Intelligence Signals

## Fraud Archetype

`src/feature_builder.py` assigns conservative provisional fraud categories such as:

- account takeover or unauthorized transaction;
- identity theft or synthetic identity fraud;
- impersonation or phishing scam;
- fake check or overpayment scam;
- investment or cryptocurrency scam;
- romance or relationship manipulation scam;
- remote-access or technical-support scam;
- rental, marketplace, or purchase scam;
- debt-collection or government impostor scam;
- money-transfer or payment-app scam;
- other suspected fraud or disputed transaction.

The classifier intentionally requires contextual evidence instead of relying on single keywords whenever possible.

If sufficient evidence is not available, the system defaults to:

```text
other suspected fraud or disputed transaction
```

rather than making a more aggressive fraud determination.

---

## Red Flags

`src/red_flags.py` identifies evidence-backed indicators including:

- urgent or threatening language;
- impersonation of a trusted organization;
- request for credentials or verification codes;
- request to transfer money;
- unexpected contact;
- remote device access;
- investment return promises;
- romance manipulation;
- counterfeit or fake checks;
- consumer denial of transaction authorization.

The rules are intentionally conservative to reduce unsupported analyst-facing claims.

---

## Risk Scoring

Risk is computed transparently from the detected red flags and fraud archetype.

The system reports:

```text
Low
Medium
High
```

Risk scores are deterministic and reproducible rather than generated by the language model.

They should be interpreted as preliminary triage indicators, not confirmed fraud determinations.

---

# Output Validation and Grounded Fallback

Small generative models can occasionally produce unusable outputs such as:

```text
Do not invent facts.
```

or:

```text
[REDACTED]
```

The system therefore validates every generated summary.

If the output fails validation, the system replaces it with a fraud-focused extractive fallback built directly from the original complaint.

Example:

```text
Raw FLAN-T5 output:
Do not invent facts.

Fallback:
Two unauthorized debit card transactions were made using my bank
debit card through a digital wallet. I immediately reported these
transactions to the bank and filed a fraud claim.
```

This guardrail is a key component of the final system.

---

# Basic End-to-End Run

From the repository root:

```bash
python src/model_runner.py
```

The command:

- loads the cleaned CFPB dataset;
- selects deterministic representative complaint samples;
- builds fraud signals;
- runs FLAN-T5;
- saves generated summaries and metadata under `outputs/`.

To validate the pipeline without loading FLAN-T5:

```bash
python src/model_runner.py --prepare-only
```

To change the sample count:

```bash
python src/model_runner.py --num-samples 5
```

---

# Development Experiment

The development experiment can be reproduced with:

```bash
python src/experiment_runner.py --num-samples 30
python src/evaluate.py
```

The experiment evaluates both:

1. raw FLAN-T5 outputs;
2. validated outputs after guardrail intervention.

The original 30-sample experiment was subsequently treated as a **development set** because its outputs were used to refine validation, red-flag, and archetype rules.

It is therefore not used as the final unbiased test set.

---

# Held-Out Final Evaluation

To avoid evaluation leakage, the final system uses a separate held-out test procedure.

The 30 development complaint IDs are stored in:

```text
configs/development_complaint_ids.txt
```

The final evaluation runner explicitly removes those complaints before selecting the test sample.

The final-test configuration uses:

```text
Seed: 2026
Final samples: 30
Development IDs excluded: 30
Development overlap: 0
```

First validate the test-set selection:

```bash
python src/final_test_runner.py --prepare-only
```

Then run the frozen pipeline:

```bash
python src/final_test_runner.py
```

Run automatic evaluation with:

```bash
python src/evaluate.py --config configs/final_eval_config.yaml
```

This writes the final evaluation artifacts to:

```text
results/final_test/
```

---

# Final Automatic Results

The final evaluation was conducted on **30 unseen CFPB complaints with zero development overlap**.

| Evaluation Variant | Source Token Support | Unsupported Number Rate | Avg. Summary Words |
|---|---:|---:|---:|
| Baseline Raw | 0.809 | 0.000 | 16.8 |
| Structured Raw | 0.772 | 0.000 | 15.5 |
| Baseline + Validation | 0.842 | 0.000 | 22.7 |
| **Structured + Validation** | **0.964** | **0.000** | 27.0 |

The raw structured-generation condition did **not** outperform the baseline.

However, the complete guarded pipeline substantially improved grounding:

```text
Baseline raw source support:       0.809
Final validated source support:    0.964
Absolute improvement:              +0.155
```

This corresponds to approximately a **19% relative increase in source-token support**.

The result suggests that the main improvement comes from the complete system architecture—especially output validation and grounded fallback—rather than prompt wording alone.

---

# Guardrail Results

On the unseen final test set:

```text
Total samples:                 30
Accepted FLAN-T5 outputs:      20
Validator-triggered fallback:  10
Fallback rate:              33.3%
```

Detected validation issues included:

| Validation Issue | Count |
|---|---:|
| Instruction echo | 6 |
| Redaction-dominated output | 4 |
| Repeated sentence | 1 |
| Too short | 1 |

A single output may trigger more than one validation condition.

The fallback rate should therefore be interpreted as the percentage of samples requiring guardrail intervention, not as a direct model-accuracy measurement.

---

# Human Evaluation

The final unseen outputs were also evaluated using a human-review rubric.

Each output was scored from **1 to 5** for:

- factuality;
- relevance;
- clarity;
- analyst usefulness.

A separate binary field records whether the output contains an unsupported or hallucinated claim.

## Final Human Evaluation Results

| Metric | Baseline Raw | Structured Final System |
|---|---:|---:|
| Factuality | 3.90 | **4.10** |
| Relevance | 2.77 | **3.43** |
| Clarity | 3.33 | **4.07** |
| Usefulness | 2.30 | **2.90** |
| Unsupported / hallucinated claim rate | **3.3%** | 26.7% |

The structured system improved the average human rating across all four 1–5 dimensions.

The largest gains were observed in relevance, clarity, and usefulness because the final output provides explicit fraud-analysis structure rather than only a generated sentence.

However, the structured system also produced a higher unsupported-claim rate.

Manual review indicated that many of these errors originated from **rule-based archetype or red-flag interpretations**, rather than unsupported numerical generation.

This represents an important limitation and an area for future improvement.

---

# Interpretation of Results

The experiment supports three main conclusions.

### 1. Prompting alone was not sufficient

The structured raw FLAN-T5 output achieved lower source-token support than the baseline:

```text
Baseline raw:    0.809
Structured raw:  0.772
```

Therefore, the project does not claim that structured prompting alone improved generation.

### 2. Guardrails substantially improved grounding

After output validation and extractive fallback, structured source support increased to:

```text
0.964
```

The final architecture therefore performed better as a **hybrid generative + deterministic system** than as a standalone language-model prompt.

### 3. Structured analyst outputs improve usability but introduce classification risk

Human evaluation showed improvements in:

```text
Factuality:   3.90 → 4.10
Relevance:    2.77 → 3.43
Clarity:      3.33 → 4.07
Usefulness:   2.30 → 2.90
```

At the same time, the rule-based fraud classification layer occasionally produced unsupported fraud interpretations.

Future work should therefore focus on improving archetype and red-flag classification while preserving the grounding benefits of the validation/fallback architecture.

---

# Evaluation Metrics

Because the CFPB dataset does not contain analyst-written reference summaries, reference-based metrics such as ROUGE are not treated as ground-truth measures of accuracy.

Instead, the project reports transparent diagnostic metrics including:

- non-empty output rate;
- average summary length;
- red-flag coverage;
- archetype keyword coverage;
- source-token support;
- unsupported-number rate;
- validator-triggered fallback rate;
- human factuality;
- human relevance;
- human clarity;
- human usefulness;
- human unsupported-claim review.

These metrics should be interpreted together rather than as independent measures of model accuracy.

---

# Reproducibility

The project includes several reproducibility controls:

- deterministic random seeds;
- deterministic representative sampling;
- fixed development complaint IDs;
- explicit development/test separation;
- zero-overlap verification;
- deterministic beam-search decoding;
- version-controlled YAML configuration;
- saved prompts;
- saved experiment metadata;
- saved complaint IDs for the final test set.

Development experiment:

```text
seed = 42
```

Held-out final evaluation:

```text
seed = 2026
development overlap = 0
```

The final test complaint IDs are saved to:

```text
results/final_test/final_test_complaint_ids.txt
```

---

# Google Colab Setup

1. Push the repository to GitHub.
2. Open the project notebook in Google Colab.
3. Change the runtime to a **T4 GPU**.
4. Clone or pull the repository.
5. Install the required dependencies.
6. Run the desired pipeline.

Example:

```bash
git clone <REPOSITORY_URL>
cd Milestone-5-Final-Project-Submission

pip install -r requirements.txt
python src/model_runner.py
```

For final evaluation:

```bash
python src/final_test_runner.py
python src/evaluate.py --config configs/final_eval_config.yaml
```

---

# Local Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python src/model_runner.py
```

---

# Docker

Build from the project root:

```bash
docker build -t cfpb-fraud-intelligence .
```

Run:

```bash
docker run --rm \
  -v "$PWD/outputs:/app/outputs" \
  cfpb-fraud-intelligence
```

The default container uses CPU.

GPU execution requires an NVIDIA-compatible container runtime.

---

# Known Limitations

The current system has several important limitations:

- CFPB complaints do not include analyst-written reference summaries.
- FLAN-T5-small sometimes produces instruction echoes or incomplete summaries.
- Approximately one-third of unseen structured generations required fallback.
- Extractive fallbacks improve grounding but may be less fluent than abstractive summaries.
- Rule-based fraud archetypes can miss emerging or linguistically unusual scam patterns.
- Rule-based fraud categories can occasionally overinterpret complaint language.
- Risk scores are heuristic triage indicators rather than calibrated fraud probabilities.
- Human evaluation was conducted on a relatively small 30-complaint held-out sample.
- Source-token support measures lexical grounding and does not prove semantic factual correctness.

Outputs should therefore be treated as **analyst decision-support signals**, not automated fraud determinations.

---

# Future Work

Potential extensions include:

- replacing rule-based archetype detection with a validated transformer classifier;
- confidence-calibrated fraud classification;
- improved distinction between authorized scams and unauthorized transactions;
- semantic entailment-based hallucination detection;
- larger blinded human evaluation;
- comparison with larger instruction-tuned language models;
- automatic confidence scoring;
- retrieval of related historical complaint patterns;
- fraud-trend clustering and emerging-scam discovery.

---

# Conclusion

This project demonstrates that reliable fraud-intelligence generation requires more than prompting a language model.

On an unseen 30-complaint test set with **zero overlap with development data**, the raw structured prompt did not outperform the baseline. However, combining FLAN-T5 with deterministic fraud signals, output validation, and a grounded extractive fallback increased source-token support from **0.809 to 0.964**.

Human evaluation also showed improvements in factuality, relevance, clarity, and usefulness.

The results support a hybrid architecture in which generative models provide natural-language summarization while deterministic controls and validation mechanisms improve traceability and grounding.

The main remaining limitation is the accuracy of rule-based fraud interpretation, which provides a clear direction for future work.
