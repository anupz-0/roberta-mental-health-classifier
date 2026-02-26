import re
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════
#  DEVICE
# ══════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ══════════════════════════════════════════════════════════════
#  LABELS  — 6 classes (Stress merged into Anxiety)
# ══════════════════════════════════════════════════════════════
SENTIMENT_LABELS = [
    "Anxiety", "Bipolar", "Depression",
    "Normal", "Personality Disorder", "Suicidal"
]

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

# ══════════════════════════════════════════════════════════════
#  LOAD MODELS
# ══════════════════════════════════════════════════════════════
sentiment_tokenizer = AutoTokenizer.from_pretrained("Sentimental-analysis")
sentiment_model     = AutoModelForSequenceClassification.from_pretrained("Sentimental-analysis").to(device).eval()
print("✅ Sentiment model loaded!")

emotion_tokenizer = AutoTokenizer.from_pretrained("Goemotion-detection")
emotion_model     = AutoModelForSequenceClassification.from_pretrained("Goemotion-detection").to(device).eval()
print("✅ Emotion model loaded!")

SUICIDAL_IDX    = SENTIMENT_LABELS.index("Suicidal")
print(f"   Labels        : {SENTIMENT_LABELS}")
print(f"   Suicidal idx  : {SUICIDAL_IDX}")

# ══════════════════════════════════════════════════════════════
#  TEXT CLEANER
# ══════════════════════════════════════════════════════════════
_FILLERS = re.compile(
    r"\b(you know|i mean|basically|literally|obviously|kind of|sort of"
    r"|to be honest|anyway|anyways|just saying|so basically|like i said"
    r"|i am the|the thing is|does that make sense|if that makes sense)\b"
    r"|https?://\S+|www\.\S+|@\w+"
    r"|([.!?])\2{2,}",
    re.IGNORECASE,
)

def clean(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = text.lower().strip()
    text = _FILLERS.sub(" ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════
#  CHUNKING  (reads every word)
# ══════════════════════════════════════════════════════════════
MAX_LEN = 256
STRIDE  = 64

def get_chunks(text: str, tokenizer) -> list:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    step   = MAX_LEN - STRIDE - 2
    if len(tokens) <= step:
        return [text]
    chunks, start = [], 0
    while start < len(tokens):
        end = min(start + step, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True))
        if end == len(tokens):
            break
        start += step
    return chunks


def infer_best_chunk(text: str, tokenizer, model) -> np.ndarray:
    """Reads every chunk — returns chunk with HIGHEST confidence."""
    chunks          = get_chunks(text, tokenizer)
    best_probs      = None
    best_confidence = 0.0

    for chunk in chunks:
        enc = tokenizer(
            chunk, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1)[0].cpu().numpy()
        top_conf = float(probs.max())
        if top_conf > best_confidence:
            best_confidence = top_conf
            best_probs      = probs

    return best_probs


def infer_emotion(text: str) -> np.ndarray:
    """Last 2 sentences — avoids early positive words dominating."""
    sentences    = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    focused_text = '. '.join(sentences[-2:]) if len(sentences) > 2 else text

    enc = emotion_tokenizer(
        focused_text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = F.softmax(emotion_model(**enc).logits, dim=-1)[0].cpu().numpy()
    return probs


# ══════════════════════════════════════════════════════════════
#  DEPRESSION / NORMAL RESOLVER
#
#  If model says Normal with low confidence AND
#  Depression is second highest → return Depression instead
# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
#  RESPONSE BUILDERS
# ══════════════════════════════════════════════════════════════
def risk_level(conf: float) -> str:
    if conf >= 0.85: return "Critical"
    if conf >= 0.70: return "High"
    if conf >= 0.50: return "Moderate"
    return "Low"

def build_mental_state(label: str, conf: float, probs: np.ndarray) -> dict:
    return {
        "label"     : label,
        "confidence": round(conf, 4),
        "risk_level": risk_level(conf),
        "all_scores": {
            SENTIMENT_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }

def build_emotion(e_probs: np.ndarray) -> dict:
    e_top_idx = int(e_probs.argmax())
    return {
        "label"     : EMOTION_LABELS[e_top_idx],
        "confidence": round(float(e_probs[e_top_idx]), 4),
        "emotions"  : [
            {"emotion": EMOTION_LABELS[i], "confidence": round(c, 4)}
            for i, c in sorted(enumerate(e_probs.tolist()), key=lambda x: x[1], reverse=True)
        ],
    }


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(title="Mental Health Detection API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextIn(BaseModel):
    text: str

# ══════════════════════════════════════════════════════════════
@app.post("/analyze")
def analyze(body: TextIn):
    if not body.text or not body.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    text = clean(body.text)
    if not text:
        raise HTTPException(status_code=400, detail="Text empty after cleaning.")

    # Step 1: Sentiment model
    probs         = infer_best_chunk(text, sentiment_tokenizer, sentiment_model)
    top_idx       = int(np.argmax(probs))
    label         = SENTIMENT_LABELS[top_idx]
    conf          = float(probs[top_idx])
    suicidal_conf = float(probs[SUICIDAL_IDX])


    # Step 3: High risk flag — pure model decision only
    high_risk = label == "Suicidal"

    mental_state = build_mental_state(label, conf, probs)

    # Step 4: Emotion (skip if Suicidal)
    emotion = None
    if not high_risk:
        emotion = build_emotion(infer_emotion(text))

    return {
        "alert"          : high_risk,
        "stage"          : "crisis" if high_risk else "completed",
        "cleaned_text"   : text,
        "mental_state"   : mental_state,
        "suicidal_signal": {
            "detected"  : high_risk,
            "confidence": round(suicidal_conf, 4),
            "risk_level": risk_level(suicidal_conf),
        },
        "emotion"        : emotion,
        "high_risk"      : high_risk,
    }


@app.post("/debug")
def debug(body: TextIn):
    text  = clean(body.text)
    probs = infer_best_chunk(text, sentiment_tokenizer, sentiment_model)
    top_idx = int(np.argmax(probs))
    label   = SENTIMENT_LABELS[top_idx]
    conf    = float(probs[top_idx])
    return {
        "cleaned_text"    : text,
        "sentiment_scores": {SENTIMENT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "raw_prediction"  : label,
        "raw_confidence"  : round(conf, 4),
        "suicidal_conf"   : round(float(probs[SUICIDAL_IDX]), 4),
    }


@app.get("/health")
def health():
    return {
        "status" : "ok",
        "version": "5.0.0",
        "models" : {"sentiment": "loaded", "emotion": "loaded"},
        "labels" : SENTIMENT_LABELS,
    }