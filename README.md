# gpt2_finetuned_vs_fewShots


# GPT-2 Few-Shot vs Fine-Tuned on AG News

This repo contains code, notebook, a short report and sample outputs for my project comparing few-shot prompting vs fine-tuning of GPT-2 on the AG News dataset.

## Contents
- `notebooks/gpt2_agnews_finetune.ipynb` — Colab-ready notebook.
- `scripts/` — optional Python scripts to reproduce training/generation/evaluation.
- `report/report.pdf` — 2-page report summarizing methods and results.
- `outputs/` — sample generation outputs (pretrained vs fine-tuned).
- `requirements.txt` — Python packages and versions to install.

## Quick start (run in Colab)
1. Open the notebook in Colab: `File -> Upload notebook` and choose `notebooks/gpt2_agnews_finetune.ipynb`.
2. Set Runtime → Change runtime type → GPU (preferably A100/T4).
3. Run the first cell to install dependencies and then run cells in order.

> NOTE: If you want to re-run full fine-tuning, use an A100 / large GPU and expect several hours. Checkpoints and final model are large — see below for storage options.

## Reproducibility
- Random seed used: `SEED = 42`.
- Block size: 512 tokens.
- Fine-tuning: GPT-2 small, 20 epochs, lr=3e-5, per_device_batch_size=64, gradient_accumulation_steps=4.

## Large files / model checkpoints
I do **not** store model checkpoints in the repo. If you need the final checkpoint:
- Option A: upload to a cloud (Google Drive) and add link in `video/link.txt`.
- Option B: push model to Hugging Face Hub (recommended). See `scripts/push_to_hf.sh` for example.

## How to cite / use
- My reproduction is built from the NeurIPS 2020 paper: Brown et al., "Language Models are Few-Shot Learners".
- See `report/report.pdf` for details and experimental results.

## License
MIT — see `LICENSE`.

