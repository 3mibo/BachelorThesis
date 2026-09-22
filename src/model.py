"""
model.py – Model loading, LoRA attachment and conversation-text construction.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base model + processor
# ---------------------------------------------------------------------------

def load_base_model(
    model_name_or_path: str = "Qwen/Qwen2-Audio-7B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple:
    """
    Load Qwen2-Audio model and processor.

    Returns (model, processor).
    """
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    return model, processor


# ---------------------------------------------------------------------------
# LoRA attachment
# ---------------------------------------------------------------------------

def attach_lora(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Tuple:
    """
    Attach LoRA adapters to the language-model decoder (q_proj / v_proj).

    The audio tower is intentionally excluded so only the text decoder
    is fine-tuned.

    Returns (peft_model, list_of_target_module_names).
    """
    from peft import LoraConfig, get_peft_model

    target_modules = [
        name
        for name, module in model.named_modules()
        if (
            isinstance(module, nn.Linear)
            and name.endswith(("q_proj", "v_proj"))
            and "audio_tower" not in name
        )
    ]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    return model, target_modules


# ---------------------------------------------------------------------------
# Conversation / chat-template helpers
# ---------------------------------------------------------------------------

def build_conversation_text(
    processor,
    prompt_text: str,
    target_text: Optional[str] = None,
    audio_placeholder: str = "<|audio_bos|><|AUDIO|><|audio_eos|>",
) -> str:
    """
    Build the full input string using Qwen2-Audio's ChatML template.

    If *target_text* is provided the assistant turn is appended (for training).
    If not, only the user turn is built (for inference).

    The audio placeholder is expected by the tokeniser to mark the audio
    position; it must match what the processor uses internally.
    """
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_placeholder},
                {"type": "text",  "text": prompt_text},
            ],
        }
    ]
    if target_text is not None:
        conversation.append({"role": "assistant", "content": target_text})

    # apply_chat_template renders the ChatML string
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=(target_text is None),
        tokenize=False,
    )
    return text
