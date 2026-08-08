# Milestone 3 Method and Implementation Notes

## 1. Objective

The system has three connected tasks:

1. Discover coherent scam archetypes from unlabeled complaint narratives.
2. Train a classifier to assign new complaints to the reviewed archetypes.
3. Generate a concise analyst-ready summary from the narrative and structured signals.

## 2. Review of the original notebook

The original notebook successfully loads 25,777 narratives, normalizes text, embeds complaints with
`all-MiniLM-L6-v2`, and runs UMAP + HDBSCAN through BERTopic. Its recorded baseline is:

- 5 non-noise topics
- 2,922 noise complaints
- DBCV relative validity: 0.040
- mean C_v coherence: 0.447

However, topic 0 contains 19,812 complaints, while several topic keywords are mostly common words.
This means the numeric clusters are not yet a defensible archetype label set. The revised notebook
adds a stopword-aware representation, parameter benchmark, corpus audit, deduplication, and a manual
review gate.


## 2A. Attached dataset audit

The actual CSV contains 25,777 rows. Deterministic cleaning removes
94 very short narratives and
107 exact normalized duplicates, leaving
25,576 records.

A regex scam indicator matches only 65.8% of retained records.
Manual inspection found clear false negatives, so it is used only as an audit feature rather than as
a hard inclusion rule. The complete cleaned working corpus proceeds to BERTopic, where noise handling
and human review determine which discovered topics become final archetypes.

## 3. Method selection

| Task | Methods compared | Selected approach | Rationale |
|---|---|---|---|
| Discovery | LDA conceptually; BERTopic configurations empirically | Sentence Transformer + UMAP + HDBSCAN + BERTopic | Semantic embeddings handle paraphrases; HDBSCAN does not force every complaint into a cluster. |
| Classification | TF-IDF logistic regression, DistilBERT, RoBERTa | Select by validation macro F1 and inference cost | The baseline tests whether Transformers add meaningful value; macro F1 protects minority archetypes. |
| Generation | Deterministic template, zero-shot FLAN-T5, fine-tuned FLAN-T5 | Fine-tuned FLAN-T5 only after human targets exist | Encoder-decoder generation fits structured text-to-text mapping, but requires valid target summaries. |

## 4. Preliminary experiments

### Discovery

Benchmark HDBSCAN settings with:

- DBCV
- number of clusters
- noise rate
- largest-cluster share
- representative-document review

The best numeric score is not automatically accepted. A team member must read representative
complaints and assign a coherent label to every retained topic.

### Classification

Run the same fixed splits for:

- TF-IDF + balanced logistic regression
- DistilBERT with and without weighted loss
- RoBERTa with and without weighted loss

Use a small stratified sample for the development run, then rerun the selected configurations on the
full training set.

### Generation

Compare:

- deterministic template baseline
- zero-shot FLAN-T5
- fine-tuned FLAN-T5

Weak template targets are acceptable for pipeline testing but not for final factuality claims. Create
and freeze a human-written test set before model or prompt tuning.

## 5. Reproducibility controls

- Random seed fixed at 42
- Configuration stored in `configs/project_config.json`
- Exact duplicate narratives removed before splitting
- Fixed stratified train/validation/test files saved to disk
- Embeddings cached to avoid accidental changes between clustering experiments
- Model outputs, reports, and prediction files saved under `results/`

## 6. Acceptance criteria before final submission

- Topic labels are human-readable and documented
- No unexplained giant cluster dominates the dataset
- C_v coherence is improved or the limitation is discussed honestly
- Classifier final run uses non-provisional labels
- Generator evaluation uses a held-out human target set
- README instructions work from a clean environment
- Repository contains meaningful commits and reviewed pull requests
