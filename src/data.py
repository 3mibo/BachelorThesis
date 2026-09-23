"""
data.py – Dataset loading, filename parsing and speaker-fold construction.

Supports EmoDB and RAVDESS.
"""

import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# EmoDB constants
# ---------------------------------------------------------------------------

EMODB_EMOTION_MAP = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}

EMODB_SPEAKERS = {
    3: "M", 8: "F", 9: "F", 10: "M",
    11: "M", 12: "M", 13: "F", 14: "F",
    15: "M", 16: "F",
}


def parse_emodb_filename(path: str) -> Optional[Dict]:
    """
    Parse a single EmoDB .wav path.

    EmoDB filename format:  <speaker_id:2><text_id:2><emotion_code><version>.wav
    e.g. 03a01Wa.wav  →  speaker=03, emotion=W (anger)

    Returns a dict with keys: path, actor, gender, emotion
    or None if the filename cannot be parsed.
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    try:
        speaker_id = int(stem[:2])
    except ValueError:
        return None
    # emotion code is the first uppercase letter after position 2
    emotion_code = next(
        (c for c in stem[2:] if c in EMODB_EMOTION_MAP), None
    )
    if emotion_code is None:
        return None
    emotion = EMODB_EMOTION_MAP[emotion_code]
    gender = "male" if EMODB_SPEAKERS.get(speaker_id) == "M" else "female"
    return {
        "path":   path,
        "actor":  speaker_id,
        "gender": gender,
        "emotion": emotion,
    }


def load_emodb(data_dir: str) -> List[Dict]:
    """Load all valid EmoDB samples from *data_dir*."""
    samples = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(data_dir, fname)
        sample = parse_emodb_filename(fpath)
        if sample is not None:
            samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# RAVDESS constants
# ---------------------------------------------------------------------------

# Emotion code (1-indexed) → canonical label
RAVDESS_EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}


def parse_ravdess_filename(path: str) -> Optional[Dict]:
    """
    Parse a single RAVDESS .wav path.

    RAVDESS filename format (1-indexed, zero-padded):
      Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    e.g. 03-01-06-01-02-01-12.wav
         modality=03 (audio-only), emotion=06 (fearful), actor=12 (female)

    Only modality==03 (audio-only speech) samples are kept.
    Returns a dict with keys: path, actor, gender, emotion, intensity
    or None if the filename cannot be parsed or is not audio-only speech.
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality  = int(parts[0])
        emotion   = int(parts[2])
        intensity = int(parts[3])
        actor_id  = int(parts[6])
    except ValueError:
        return None
    if modality != 3:          # keep only audio-only speech
        return None
    emotion_label = RAVDESS_EMOTION_MAP.get(emotion)
    if emotion_label is None:
        return None
    gender = "male" if actor_id % 2 == 1 else "female"
    return {
        "path":      path,
        "actor":     actor_id,
        "gender":    gender,
        "emotion":   emotion_label,
        "intensity": intensity,
    }


def load_ravdess(data_dir: str) -> List[Dict]:
    """
    Load all valid RAVDESS samples.

    *data_dir* should point to the folder that contains Actor_01 … Actor_24
    subdirectories (the extracted RAVDESS audio_speech_actors_01-24 folder).
    """
    samples = []
    for actor_dir in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in sorted(os.listdir(actor_path)):
            if not fname.lower().endswith(".wav"):
                continue
            fpath = os.path.join(actor_path, fname)
            sample = parse_ravdess_filename(fpath)
            if sample is not None:
                samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Generic dataset loader
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, data_dir: str) -> List[Dict]:
    """Load samples for *dataset* ('emodb' or 'ravdess') from *data_dir*."""
    if dataset == "emodb":
        return load_emodb(data_dir)
    elif dataset == "ravdess":
        return load_ravdess(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Speaker-independent fold construction
# ---------------------------------------------------------------------------

def build_speaker_folds(
    samples: List[Dict],
    held_out_per_fold: int,
    seed: int = 42,
) -> List[Tuple[List[Dict], List[Dict]]]:
    """
    Build speaker-independent folds.

    Each fold holds out *held_out_per_fold* speakers (aiming for gender
    balance: half male, half female).  The remaining speakers form the
    training set.

    Returns a list of (train_samples, test_samples) tuples.

    EmoDB:  held_out_per_fold=2  → 5 folds  (10 speakers total)
    RAVDESS: held_out_per_fold=4 → 6 folds  (24 speakers total)
    """
    rng = random.Random(seed)

    # Group speakers by gender
    male_speakers   = sorted({s["actor"] for s in samples if s["gender"] == "male"})
    female_speakers = sorted({s["actor"] for s in samples if s["gender"] == "female"})

    rng.shuffle(male_speakers)
    rng.shuffle(female_speakers)

    half = held_out_per_fold // 2
    n_folds = min(len(male_speakers) // half, len(female_speakers) // half)

    # Build speaker→samples index
    speaker_index: Dict[int, List[Dict]] = defaultdict(list)
    for s in samples:
        speaker_index[s["actor"]].append(s)

    folds = []
    for i in range(n_folds):
        held_male   = male_speakers[i * half : (i + 1) * half]
        held_female = female_speakers[i * half : (i + 1) * half]
        held_set    = set(held_male + held_female)

        test_samples  = [s for s in samples if s["actor"] in held_set]
        train_samples = [s for s in samples if s["actor"] not in held_set]
        folds.append((train_samples, test_samples))

    return folds


# ---------------------------------------------------------------------------
# Audio loading (cached)
# ---------------------------------------------------------------------------

def make_audio_cache() -> Dict:
    """Return a fresh (empty) audio cache dict."""
    return {}


def load_audio(path: str, processor, cache=None):
    import librosa
    if cache is not None and path in cache:
        return cache[path]
    target_sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    if cache is not None:
        cache[path] = audio
    return audio
