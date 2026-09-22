# SER Fine-tuning with Qwen2-Audio + LoRA

Speaker-independent k-fold cross-validation for Speech Emotion Recognition
on EmoDB and RAVDESS using Qwen2-Audio-7B-Instruct with LoRA adapters.

## Project structure

```
ser_project/
├── train.py          # Main training + evaluation entry point
├── gridsearch.py     # Hyperparameter grid search runner
├── requirements.txt
└── src/
    ├── prompts.py    # Prompt templates, label maps, output normalisation
    ├── data.py       # Dataset loaders, filename parsers, fold builder
    ├── model.py      # Model/processor loading, LoRA attachment
    ├── training.py   # Training loop with gradient accumulation
    ├── evaluation.py # Inference, UAR/ACC/WF1 metrics
    └── utils.py      # Seed, plotting, CSV/JSON helpers
```

## Setup (Berta HPC)

```bash
conda activate ser
pip install -r requirements.txt
```

## Single training run

```bash
# EmoDB (German, 5-fold)
python train.py \
    --dataset emodb \
    --data_dir /ssd/4/u/baz9201/data/EMODB \
    --output_dir /stor/u/baz9201/ser_results/emodb_baseline \
    --model_path Qwen/Qwen2-Audio-7B-Instruct

# RAVDESS (English, 6-fold)
python train.py \
    --dataset ravdess \
    --data_dir /ssd/4/u/baz9201/data/RAVDESS/audio_speech_actors_01-24 \
    --output_dir /stor/u/baz9201/ser_results/ravdess_baseline
```

## Grid search

```bash
python gridsearch.py \
    --dataset emodb \
    --data_dir /ssd/4/u/baz9201/data/EMODB \
    --output_root /stor/u/baz9201/ser_results/gridsearch_emodb \
    --lora_r 8 16 32 \
    --lr 1e-4 5e-5 \
    --epochs 3 5 \
    --seeds 42 123
```

Results are saved per run as `summary.json` / `summary.csv`, and aggregated
into a timestamped master `gridsearch_results_YYYYMMDD_HHMMSS.csv`.

## Key arguments (`train.py`)

| Flag | Default | Description |
|---|---|---|
| `--dataset` | — | `emodb` or `ravdess` |
| `--data_dir` | — | Root folder of the dataset |
| `--output_dir` | — | Where to save results |
| `--model_path` | `Qwen/Qwen2-Audio-7B-Instruct` | HF model ID or local path |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 3 | Training epochs |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--train_prompt` | `normal` | Prompt used during training |
| `--seed` | 123 | Random seed |
| `--save_model` | False | Save LoRA adapter per fold |

## Metrics

- **UAR** (Unweighted Average Recall = Macro Recall) – primary metric.
  Treats all classes equally; robust to class imbalance.
- **ACC** – overall accuracy
- **W-F1** – weighted F1 (weighted by class frequency)
