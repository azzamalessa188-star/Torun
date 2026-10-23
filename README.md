# Torun

**Torun** is an Arabic-focused coding-model adaptation project. It starts from an existing code model and provides a reproducible pipeline for preparing Arabic technical instruction data, fine-tuning with QLoRA, and evaluating Arabic and coding behavior.

> **Status:** Initial research/training scaffold. No fine-tuning has been run yet, and no performance improvement is claimed.

## Base model

The initial candidate is [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct). Review the model card and its license before use or redistribution. Torun's training code does not include or redistribute the base-model weights.

## Goals

- Improve usefulness in Arabic technical conversations and programming tasks.
- Preserve code quality and instruction following while adding Arabic-focused examples.
- Make data preparation, training, and evaluation repeatable.
- Keep dataset sources, licenses, and evaluation results documented.

## Repository layout

- `training/train.py` — QLoRA supervised fine-tuning entry point.
- `training/prepare_data.py` — validate and format JSONL conversation data.
- `evaluation/evaluate.py` — lightweight generation smoke test.
- `configs/qlora.yaml` — starter training configuration.
- `data/README.md` — dataset format and data governance notes.

## Quick start

Use a Python 3.10 or 3.11 environment and a supported NVIDIA GPU for 4-bit QLoRA training.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Prepare your own licensed dataset in JSONL format (see [data/README.md](data/README.md)), then validate it:

```bash
python training/prepare_data.py --input data/train.jsonl --output data/train.prepared.jsonl
```

Start training with the default config:

```bash
python training/train.py --config configs/qlora.yaml --train-file data/train.prepared.jsonl
```

Run a quick inference check against a saved adapter or merged model:

```bash
python evaluation/evaluate.py --model Qwen/Qwen2.5-Coder-7B-Instruct
```

The default inference command checks the base model. Pass `--model` with your local output path to test a fine-tuned checkpoint.

## Important limitations

- Fine-tuning is not the same as training a model from scratch.
- Arabic capability must be measured; it must not be assumed from multilingual support alone.
- A 7B model with QLoRA still needs substantial GPU memory and compatible CUDA/bitsandbytes versions. Training may require adjusting sequence length, batch size, or gradient accumulation.
- Do not commit model weights, API keys, private datasets, or personal data.
- Do not use scraped or generated data unless its provenance, permissions, and quality have been reviewed.

## Evaluation before claiming success

Compare the untouched base model and the fine-tuned adapter on the same held-out prompts. Include Arabic explanation, Arabic debugging, code generation, tests/pass rates, hallucination checks, and regression checks in English/code-only tasks. Record model revision, dataset version, hyperparameters, hardware, and metrics.

## License

The repository's code license and the base model's license are separate. See [LICENSE](LICENSE) for this repository's code license and the base model card for model terms.