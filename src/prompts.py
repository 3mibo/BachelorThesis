"""
prompts.py – Prompt templates, label maps and output normalisation.

Both EmoDB (German primary) and RAVDESS (English primary) are supported.
"""

# ---------------------------------------------------------------------------
# EmoDB
# ---------------------------------------------------------------------------

EMODB_LABELS = ["anger", "boredom", "disgust", "fear", "happiness", "sadness", "neutral"]

DE_LABEL_MAP = {
    "anger":     "Ärger",
    "boredom":   "Langeweile",
    "disgust":   "Ekel",
    "fear":      "Angst",
    "happiness": "Freude",
    "sadness":   "Trauer",
    "neutral":   "Neutral",
}

EMODB_PROMPT_TEMPLATES = {
    # --- German ---
    "normal": (
        "Klassifiziere die Emotion in dieser Sprachaufnahme. "
        "Antworte NUR mit einem der folgenden Wörter: "
        "Ärger, Langeweile, Ekel, Angst, Freude, Trauer, Neutral."
    ),
    "prosodic": (
        "Analysiere Tonhöhe, Tempo, Energie und Stimmqualität dieser Sprachaufnahme. "
        "Klassifiziere die Emotion. "
        "Antworte NUR mit einem der folgenden Wörter: "
        "Ärger, Langeweile, Ekel, Angst, Freude, Trauer, Neutral."
    ),
    "expert": (
        "Du bist ein Experte für Sprachemotion. "
        "Analysiere diese Aufnahme auf prosodische und paralinguistische Merkmale "
        "und entscheide, welche der folgenden Emotionen am besten passt: "
        "Ärger, Langeweile, Ekel, Angst, Freude, Trauer, Neutral. "
        "Antworte ausschließlich mit dem passenden Wort."
    ),
    # --- English ---
    "normal_en": (
        "Classify the emotion in this speech recording. "
        "Reply with ONLY one of the following words: "
        "anger, boredom, disgust, fear, happiness, sadness, neutral."
    ),
    "prosodic_en": (
        "Analyse pitch, tempo, energy and voice quality in this speech recording. "
        "Classify the emotion. "
        "Reply with ONLY one of the following words: "
        "anger, boredom, disgust, fear, happiness, sadness, neutral."
    ),
    "expert_en": (
        "You are an expert in speech emotion recognition. "
        "Analyse this recording for prosodic and paralinguistic cues "
        "and decide which of the following emotions fits best: "
        "anger, boredom, disgust, fear, happiness, sadness, neutral. "
        "Reply with that single word only."
    ),
}

# Map every variant the model might output → canonical English label
EMODB_ALIAS_TABLE = {
    # German originals
    "ärger":      "anger",
    "arger":      "anger",
    "langeweile": "boredom",
    "ekel":       "disgust",
    "angst":      "fear",
    "freude":     "happiness",
    "trauer":     "sadness",
    "neutral":    "neutral",
    # English
    "anger":      "anger",
    "boredom":    "boredom",
    "disgust":    "disgust",
    "fear":       "fear",
    "happiness":  "happiness",
    "sadness":    "sadness",
    # Common model paraphrases
    "angry":      "anger",
    "bored":      "boredom",
    "disgusted":  "disgust",
    "fearful":    "fear",
    "scared":     "fear",
    "happy":      "happiness",
    "joy":        "happiness",
    "sad":        "sadness",
}


# ---------------------------------------------------------------------------
# RAVDESS
# ---------------------------------------------------------------------------

RAVDESS_LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

RAVDESS_DE_LABEL_MAP = {
    "neutral":   "Neutral",
    "calm":      "Ruhig",
    "happy":     "Freude",
    "sad":       "Traurigkeit",
    "angry":     "Wut",
    "fearful":   "Furcht",
    "disgust":   "Ekel",
    "surprised": "Überraschung",
}

RAVDESS_PROMPT_TEMPLATES = {
    # --- English ---
    "normal": (
        "Classify the emotion in this speech recording. "
        "Reply with ONLY one of the following words: "
        "neutral, calm, happy, sad, angry, fearful, disgust, surprised."
    ),
    "prosodic": (
        "Analyse pitch, tempo, energy and voice quality in this speech recording. "
        "Classify the emotion. "
        "Reply with ONLY one of the following words: "
        "neutral, calm, happy, sad, angry, fearful, disgust, surprised."
    ),
    "expert": (
        "You are an expert in speech emotion recognition. "
        "Analyse this recording for prosodic and paralinguistic cues "
        "and decide which of the following emotions fits best: "
        "neutral, calm, happy, sad, angry, fearful, disgust, surprised. "
        "Reply with that single word only."
    ),
    # --- German ---
    "normal_de": (
        "Klassifiziere die Emotion in dieser Sprachaufnahme. "
        "Antworte NUR mit einem der folgenden Wörter: "
        "Neutral, Ruhig, Freude, Traurigkeit, Wut, Furcht, Ekel, Überraschung."
    ),
    "prosodic_de": (
        "Analysiere Tonhöhe, Tempo, Energie und Stimmqualität dieser Sprachaufnahme. "
        "Klassifiziere die Emotion. "
        "Antworte NUR mit einem der folgenden Wörter: "
        "Neutral, Ruhig, Freude, Traurigkeit, Wut, Furcht, Ekel, Überraschung."
    ),
    "expert_de": (
        "Du bist ein Experte für Sprachemotion. "
        "Analysiere diese Aufnahme auf prosodische und paralinguistische Merkmale "
        "und entscheide, welche der folgenden Emotionen am besten passt: "
        "Neutral, Ruhig, Freude, Traurigkeit, Wut, Furcht, Ekel, Überraschung. "
        "Antworte ausschließlich mit dem passenden Wort."
    ),
}

RAVDESS_ALIAS_TABLE = {
    # English
    "neutral":    "neutral",
    "calm":       "calm",
    "happy":      "happy",
    "happiness":  "happy",
    "joy":        "happy",
    "sad":        "sad",
    "sadness":    "sad",
    "angry":      "angry",
    "anger":      "angry",
    "fearful":    "fearful",
    "fear":       "fearful",
    "scared":     "fearful",
    "disgust":    "disgust",
    "disgusted":  "disgust",
    "surprised":  "surprised",
    "surprise":   "surprised",
    # German
    "neutral":    "neutral",
    "ruhig":      "calm",
    "freude":     "happy",
    "traurigkeit":"sad",
    "wut":        "angry",
    "furcht":     "fearful",
    "ekel":       "disgust",
    "überraschung": "surprised",
    "uberraschung": "surprised",
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_prompt_templates(dataset: str) -> dict:
    """Return the prompt template dict for *dataset* ('emodb' or 'ravdess')."""
    if dataset == "emodb":
        return EMODB_PROMPT_TEMPLATES
    elif dataset == "ravdess":
        return RAVDESS_PROMPT_TEMPLATES
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_labels(dataset: str) -> list:
    if dataset == "emodb":
        return EMODB_LABELS
    elif dataset == "ravdess":
        return RAVDESS_LABELS
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_alias_table(dataset: str) -> dict:
    if dataset == "emodb":
        return EMODB_ALIAS_TABLE
    elif dataset == "ravdess":
        return RAVDESS_ALIAS_TABLE
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def normalize_label(raw: str, dataset: str) -> str:
    """
    Map raw model output to a canonical label.
    Returns the canonical label string, or 'unknown' if no match found.
    """
    alias_table = get_alias_table(dataset)
    cleaned = raw.strip().lower().rstrip(".,!?;:")
    # Try exact match first
    if cleaned in alias_table:
        return alias_table[cleaned]
    # Try substring match (model sometimes adds extra words)
    for key, label in alias_table.items():
        if key in cleaned:
            return label
    return "unknown"


def get_train_target(emotion: str, prompt_name: str, dataset: str) -> str:
    """
    Return the training target string for a given emotion and prompt name.
    EmoDB German prompts use DE_LABEL_MAP; English prompts use the raw label.
    RAVDESS German prompts use RAVDESS_DE_LABEL_MAP; English prompts use the raw label.
    """
    if dataset == "emodb":
        if not prompt_name.endswith("_en"):
            return DE_LABEL_MAP[emotion]
        return emotion
    elif dataset == "ravdess":
        if prompt_name.endswith("_de"):
            return RAVDESS_DE_LABEL_MAP[emotion]
        return emotion
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
