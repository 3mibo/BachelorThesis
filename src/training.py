"""
training.py – Training loop with gradient accumulation.

One training example is processed at a time (variable-length audio prevents
real batching).  Gradient accumulation simulates a larger effective batch.
"""

from typing import Dict, List, Optional
import torch


# ---------------------------------------------------------------------------
# Per-sample helpers
# ---------------------------------------------------------------------------

def build_train_example(
    sample: Dict,
    processor,
    prompt_text: str,
    target_text: str,
    audio_cache: Optional[Dict] = None,
) -> Dict:
    """
    Build a single training example (input_ids, labels, attention_mask,
    input_features, feature_attention_mask).

    The loss is masked for the prompt portion so that only the target token(s)
    are trained.
    """
    from src.data import load_audio
    from src.model import build_conversation_text

    # ------------------------------------------------------------------
    # 1. Build the full chat text (prompt + target)
    # ------------------------------------------------------------------
    full_text = build_conversation_text(processor, prompt_text, target_text)

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    text_inputs = processor.tokenizer(full_text, return_tensors="pt")
    input_ids = text_inputs["input_ids"]

    # ------------------------------------------------------------------
    # 3. Build labels: mask everything up to (and including) the
    #    <|im_start|>assistant newline token so only the answer is in loss.
    # ------------------------------------------------------------------
    labels = input_ids.clone()

    # Find the assistant-turn start.  We tokenise the prompt-only text and
    # use its length as the mask boundary.
    prompt_only_text = build_conversation_text(processor, prompt_text, target_text=None)
    prompt_ids = processor.tokenizer(prompt_only_text, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]
    labels[:, :prompt_len] = -100          # mask prompt tokens from loss

    # ------------------------------------------------------------------
    # 4. Load audio features
    # ------------------------------------------------------------------
    audio_data = load_audio(sample["path"], processor, cache=audio_cache)

    return {
        "input_ids":              input_ids,
        "labels":                 labels,
        "attention_mask":         text_inputs["attention_mask"],
        "input_features":         audio_data["input_features"],
        "feature_attention_mask": audio_data["feature_attention_mask"],
    }


def move_to_device(example: Dict, device) -> Dict:
    """Move all tensors in *example* to *device*, skipping None values."""
    return {
        k: v.to(device) if v is not None else None
        for k, v in example.items()
    }


# ---------------------------------------------------------------------------
# One-fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    model,
    processor,
    train_samples: List[Dict],
    prompt_text: str,
    dataset: str,
    train_prompt_name: str,
    num_epochs: int,
    learning_rate: float,
    grad_accum_steps: int,
    audio_cache: Optional[Dict] = None,
    device=None,
    verbose: bool = True,
) -> List[float]:
    """
    Train *model* on *train_samples* for *num_epochs*.

    Returns a list of per-step average losses (one value per accumulation
    step boundary, i.e. every *grad_accum_steps* samples).
    """
    from src.prompts import get_train_target

    if device is None:
        device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    model.train()
    loss_history: List[float] = []

    for epoch in range(num_epochs):
        import random
        random.shuffle(train_samples)

        optimizer.zero_grad()
        accum_loss = 0.0
        step = 0

        for i, sample in enumerate(train_samples):
            target_text = get_train_target(sample["emotion"], train_prompt_name, dataset)

            try:
                example = build_train_example(
                    sample, processor, prompt_text, target_text, audio_cache
                )
            except Exception as e:
                if verbose:
                    print(f"  [skip] {sample['path']}: {e}")
                continue

            example = move_to_device(example, device)

            outputs = model(
                input_ids=example["input_ids"],
                attention_mask=example["attention_mask"],
                input_features=example["input_features"],
                feature_attention_mask=example["feature_attention_mask"],
                labels=example["labels"],
            )

            loss = outputs.loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
            step += 1

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                avg = accum_loss
                loss_history.append(avg)
                if verbose:
                    global_step = epoch * len(train_samples) + i + 1
                    print(
                        f"  Epoch {epoch+1}/{num_epochs}  "
                        f"step {global_step}/{len(train_samples)*num_epochs}  "
                        f"loss={avg:.4f}"
                    )
                accum_loss = 0.0

        # Flush any remaining accumulated gradients at end of epoch
        if step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(accum_loss)
            accum_loss = 0.0

    return loss_history
