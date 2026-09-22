#!/usr/bin/env python3
"""
train.py – Main entry point for SER fine-tuning with Qwen2-Audio.

Supports EmoDB and RAVDESS datasets with speaker-independent k-fold
cross-validation.

Example usage (EmoDB, default settings):
  python train.py \\
      --dataset emodb \\
      --data_dir /ssd/4/u/baz9201/data/EMODB \\
      --output_dir /stor/u/baz9201/ser_results/emodb_baseline \\
      --model_path Qwen/Qwen2-Audio-7B-Instruct

Example usage (RAVDESS, custom hyperparameters):
  python train.py \\
      --dataset ravdess \\
      --data_dir /ssd/4/u/baz9201/data/RAVDESS/audio_speech_actors_01-24 \\
      --output_dir /stor/u/baz9201/ser_results/ravdess_r32 \\
      --lora_r 32 --lora_alpha 64 \\
      --lr 5e-5 --epochs 5
"""

import argparse
import os
import sys
import json
import time

import torch

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SER fine-tuning with Qwen2-Audio + LoRA")

    # Dataset
    p.add_argument("--dataset",    required=True, choices=["emodb", "ravdess"],
                   help="Which dataset to use")
    p.add_argument("--data_dir",   required=True,
                   help="Path to dataset root directory")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save checkpoints, results and plots")

    # Model
    p.add_argument("--model_path", default="Qwen/Qwen2-Audio-7B-Instruct",
                   help="HuggingFace model ID or local path")

    # LoRA
    p.add_argument("--lora_r",       type=int,   default=16)
    p.add_argument("--lora_alpha",   type=int,   default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Training
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Learning rate (AdamW)")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--grad_accum",    type=int,   default=4,
                   help="Gradient accumulation steps (effective batch size)")
    p.add_argument("--train_prompt",  default="normal",
                   help="Prompt template name to use during training")

    # Cross-validation
    p.add_argument("--held_out_per_fold", type=int, default=None,
                   help="Speakers held out per fold "
                        "(default: 2 for EmoDB, 4 for RAVDESS)")
    p.add_argument("--seed", type=int, default=123)

    # Misc
    p.add_argument("--eval_all_prompts", action="store_true", default=True,
                   help="Evaluate with all prompt templates after training "
                        "(default: True)")
    p.add_argument("--no_eval_all_prompts", dest="eval_all_prompts",
                   action="store_false")
    p.add_argument("--save_model", action="store_true", default=False,
                   help="Save LoRA adapter weights per fold")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--no_verbose", dest="verbose", action="store_false")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------
    if args.held_out_per_fold is None:
        args.held_out_per_fold = 2 if args.dataset == "emodb" else 4

    run_name = (
        f"{args.dataset}_seed{args.seed}"
        f"_R{args.lora_r}_DO{args.lora_dropout}"
        f"_AS{args.grad_accum}"
        f"_lr{args.lr}"
        f"_ep{args.epochs}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Run: {run_name}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Imports (deferred to avoid slow startup when just showing --help)
    # ------------------------------------------------------------------
    from src.utils import set_seed
    set_seed(args.seed)

    from src.data import load_dataset, build_speaker_folds, make_audio_cache
    from src.model import load_base_model, attach_lora
    from src.prompts import get_prompt_templates, get_labels
    from src.training import train_one_fold
    from src.evaluation import evaluate_all_prompts
    from src.utils import (
        aggregate_fold_results, save_results_json, save_summary_csv,
        plot_training_curves, plot_prompt_generalisation,
    )

    # ------------------------------------------------------------------
    # Load data and build folds
    # ------------------------------------------------------------------
    print(f"Loading {args.dataset} from {args.data_dir} …")
    samples = load_dataset(args.dataset, args.data_dir)
    print(f"  {len(samples)} samples loaded.")

    folds = build_speaker_folds(
        samples,
        held_out_per_fold=args.held_out_per_fold,
        seed=args.seed,
    )
    print(f"  {len(folds)} folds (holding out {args.held_out_per_fold} speakers each).")

    prompt_templates = get_prompt_templates(args.dataset)
    labels           = get_labels(args.dataset)
    train_prompt_text = prompt_templates[args.train_prompt]

    # ------------------------------------------------------------------
    # K-fold loop
    # ------------------------------------------------------------------
    all_loss_histories: list = []
    all_fold_results:   list = []

    for fold_idx, (train_samples, test_samples) in enumerate(folds):
        print(f"\n{'─'*50}")
        print(f"Fold {fold_idx+1}/{len(folds)}  |  "
              f"train={len(train_samples)}  test={len(test_samples)}")
        print(f"{'─'*50}")

        # Fresh model + LoRA for each fold
        model, processor = load_base_model(
            model_name_or_path=args.model_path,
            dtype=torch.bfloat16,
        )
        model, target_modules = attach_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        if args.verbose:
            model.print_trainable_parameters()

        audio_cache = make_audio_cache()

        # Train
        t0 = time.time()
        loss_history = train_one_fold(
            model=model,
            processor=processor,
            train_samples=train_samples,
            prompt_text=train_prompt_text,
            dataset=args.dataset,
            train_prompt_name=args.train_prompt,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            grad_accum_steps=args.grad_accum,
            audio_cache=audio_cache,
            verbose=args.verbose,
        )
        all_loss_histories.append(loss_history)
        print(f"  Training done in {(time.time()-t0)/60:.1f} min")

        # Optionally save adapter
        if args.save_model:
            fold_model_dir = os.path.join(args.output_dir, f"fold{fold_idx+1}_adapter")
            model.save_pretrained(fold_model_dir)
            print(f"  Adapter saved to {fold_model_dir}")

        # Evaluate
        t1 = time.time()
        prompts_to_eval = prompt_templates if args.eval_all_prompts else {args.train_prompt: train_prompt_text}
        fold_results = evaluate_all_prompts(
            model=model,
            processor=processor,
            test_samples=test_samples,
            prompt_templates=prompts_to_eval,
            dataset=args.dataset,
            labels=labels,
            audio_cache=audio_cache,
            verbose=args.verbose,
        )
        all_fold_results.append(fold_results)
        print(f"  Evaluation done in {(time.time()-t1)/60:.1f} min")

        # Per-fold JSON
        fold_out = os.path.join(args.output_dir, f"fold{fold_idx+1}_results.json")
        save_results_json(fold_results, fold_out)

        # Clean up GPU memory
        del model, processor, audio_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Aggregate and save
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Aggregating results …")

    summary = aggregate_fold_results(all_fold_results)
    save_results_json(summary, os.path.join(args.output_dir, "summary.json"))
    save_summary_csv(summary, os.path.join(args.output_dir, "summary.csv"))

    # Plots
    plot_training_curves(
        all_loss_histories,
        os.path.join(args.output_dir, "training_curves.png"),
        title=f"Training loss — {run_name}",
    )
    plot_prompt_generalisation(
        all_fold_results,
        labels,
        os.path.join(args.output_dir, "prompt_generalisation.png"),
        title=f"UAR per prompt — {run_name}",
    )

    # Print final table
    print(f"\n{'─'*60}")
    print(f"{'Prompt':<20}  {'UAR':>8}  {'±':>6}  {'ACC':>8}  {'WF1':>8}")
    print(f"{'─'*60}")
    for pname, m in summary.items():
        print(
            f"{pname:<20}  "
            f"{m['mean_uar']*100:>7.2f}%  "
            f"±{m['std_uar']*100:>5.2f}  "
            f"{m['mean_acc']*100:>7.2f}%  "
            f"{m['mean_wf1']*100:>7.2f}%"
        )
    print(f"{'─'*60}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
