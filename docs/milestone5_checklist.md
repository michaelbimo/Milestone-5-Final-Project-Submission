# Milestone 5 Finalization Checklist

## Code and reproducibility

- [ ] `python src/model_runner.py` runs successfully from the repository root.
- [ ] `python src/experiment_runner.py` generates baseline and structured-prompt outputs.
- [ ] `python src/evaluate.py` generates final metrics, CSVs, figures, and human-review template.
- [ ] `requirements.txt` installs the final runtime/evaluation dependencies.
- [ ] `configs/model_config.yaml` and `configs/eval_config.yaml` are committed.
- [ ] Generated checkpoints/caches are excluded from Git.

## Final results

- [ ] At least 10 representative generated samples are committed under `outputs/`.
- [ ] `results/model_comparison.csv` contains the final baseline-vs-structured results.
- [ ] `results/metrics.json` contains the machine-readable final metrics.
- [ ] Human evaluation template has been reviewed and scored for a reasonable subset.
- [ ] Final figures are committed under `results/figures/`.

## Documentation

- [ ] README explains the problem, final architecture, setup, execution, evaluation, and limitations.
- [ ] README includes the final quantitative results after the Colab experiment is complete.
- [ ] README distinguishes preliminary BERTopic/classifier work from the final integrated pipeline.
- [ ] Technical report PDF is committed under `docs/` if the course permits report submission through GitHub.
- [ ] Repository is public or instructors are added as collaborators.

## Submission

- [ ] Clone/test the repository in a fresh Colab runtime.
- [ ] Run the documented commands from a clean session.
- [ ] Confirm generated outputs and charts are reproducible.
- [ ] Verify no GitHub tokens, API keys, or private credentials are committed.
