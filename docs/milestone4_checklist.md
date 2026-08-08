# End-to-End Model Pipeline Checklist

- [x] Preprocessed data loader under `src/`
- [x] Modular feature and prompt builder
- [x] Pretrained FLAN-T5 model loading
- [x] Inference on 10 representative samples
- [x] Single command: `python src/model_runner.py`
- [x] Text and JSONL output writers
- [x] Run metadata and preliminary quality checks
- [x] Shared helpers under `utils/`
- [x] YAML configuration under `configs/`
- [x] Google Colab demo notebook
- [x] README with setup, usage, reproduction, results, and limitations
- [x] Optional Dockerfile
- [ ] Run the full model in Colab and commit the four generated output files
- [ ] Add two or three concrete observations from `outputs/samples.txt` to the README
