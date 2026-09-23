"""
evaluation.py – Inference, metric computation (UAR / ACC / W-F1) and
confusion matrix generation.
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Single-sample prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_prediction(
    model,
    processor,
    sample: Dict,
    prompt_text: str,
    dataset: str,
    audio_cache: Optional[Dict] = None,
    max_new_tokens: int = 10,
) -> str:
    """
    Run greedy generation for a single sample and return the normalized
    canonical emotion label (e.g. 'anger', 'neutral', …).

    Returns 'unknown' if the output cannot be mapped.
    """
    from src.data import load_audio
    from src.model import build_conversation_text
    from src.prompts import normalize_label

    # Build prompt-only text (no target)
    text = build_conversation_text(processor, prompt_text, target_text=None)

    audio = load_audio(sample["path"], processor, cache=audio_cache)
    text = build_conversation_text(processor, prompt_text, target_text=None)
    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    input_ids              = inputs["input_ids"].to(device)
    attention_mask         = inputs["attention_mask"].to(device)
    input_features         = inputs.get("input_features")
    feature_attention_mask = inputs.get("feature_attention_mask")
    if input_features is not None:
        input_features = input_features.to(device)
    if feature_attention_mask is not None:
        feature_attention_mask = feature_attention_mask.to(device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    raw_text   = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    predicted  = normalize_label(raw_text, dataset)
    return predicted


# ---------------------------------------------------------------------------
# Fold-level evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    model,
    processor,
    test_samples: List[Dict],
    prompt_text: str,
    dataset: str,
    labels: List[str],
    audio_cache: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate *model* on *test_samples* using *prompt_text*.

    Returns a dict with:
      - predictions: List[str]  (one per sample)
      - true_labels: List[str]
      - uar:   float  (Unweighted Average Recall, i.e. macro recall)
      - acc:   float  (overall accuracy)
      - wf1:   float  (weighted F1)
      - per_class_recall: Dict[str, float]
    """
    model.eval()
    predictions: List[str] = []
    true_labels: List[str] = []

    for i, sample in enumerate(test_samples):
        try:
            pred = generate_prediction(
                model, processor, sample, prompt_text, dataset,
                audio_cache=audio_cache,
            )
        except Exception as e:
            if verbose:
                print(f"  [eval skip] {sample['path']}: {e}")
            pred = "unknown"

        predictions.append(pred)
        true_labels.append(sample["emotion"])

        if verbose and (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(test_samples)}")

    metrics = compute_metrics(true_labels, predictions, labels)
    metrics["predictions"] = predictions
    metrics["true_labels"]  = true_labels
    return metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    true_labels: List[str],
    predictions: List[str],
    labels: List[str],
) -> Dict:
    """
    Compute UAR (macro recall), overall accuracy and weighted F1.

    Labels not in *labels* (e.g. 'unknown') count as wrong.
    """
    from sklearn.metrics import (
        recall_score, accuracy_score, f1_score, confusion_matrix
    )

    # Map to indices; unknown → -1 (will be wrong for every metric)
    label2idx = {l: i for i, l in enumerate(labels)}
    y_true = [label2idx.get(l, -1) for l in true_labels]
    y_pred = [label2idx.get(p, -1) for p in predictions]

    # Filter out any -1 entries for UAR/WF1 (mark them as wrong by keeping)
    # sklearn handles unknown classes gracefully with zero_division=0
    uar  = recall_score(y_true, y_pred, average="macro",    labels=list(range(len(labels))), zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    wf1  = f1_score(y_true, y_pred, average="weighted", labels=list(range(len(labels))), zero_division=0)

    # Per-class recall
    per_class = recall_score(
        y_true, y_pred, average=None,
        labels=list(range(len(labels))), zero_division=0
    )
    per_class_recall = {labels[i]: float(per_class[i]) for i in range(len(labels))}

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    return {
        "uar":               float(uar),
        "acc":               float(acc),
        "wf1":               float(wf1),
        "per_class_recall":  per_class_recall,
        "confusion_matrix":  cm,
    }


# ---------------------------------------------------------------------------
# Multi-prompt evaluation (all prompts at once)
# ---------------------------------------------------------------------------

def evaluate_all_prompts(
    model,
    processor,
    test_samples: List[Dict],
    prompt_templates: Dict[str, str],
    dataset: str,
    labels: List[str],
    audio_cache: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate *model* using every prompt in *prompt_templates*.

    Returns a dict mapping prompt_name → metrics dict.
    """
    results = {}
    for prompt_name, prompt_text in prompt_templates.items():
        if verbose:
            print(f"\n  Evaluating prompt '{prompt_name}' …")
        metrics = evaluate_fold(
            model, processor, test_samples, prompt_text,
            dataset, labels, audio_cache=audio_cache, verbose=verbose,
        )
        results[prompt_name] = metrics
        if verbose:
            print(
                f"  [{prompt_name}] UAR={metrics['uar']:.4f}  "
                f"ACC={metrics['acc']:.4f}  WF1={metrics['wf1']:.4f}"
            )
    return results
