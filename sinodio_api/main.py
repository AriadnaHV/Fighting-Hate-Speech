"""
SinOdio/SansHaine - Hate Speech Detection API
FastAPI inference endpoint for XLM-RoBERTa binary classifier (Phase 1)
"""

import os
import io
import re
import logging
from contextlib import asynccontextmanager

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------
# Logging
# -----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------
# Constants  (match training notebook)
# -----------------------------------------
MODEL_NAME       = "xlm-roberta-base"
MAX_LENGTH       = 128
GCS_BUCKET       = "sinodio-models"
GCS_MODEL_PREFIX = "phase1/xlm_roberta/xlm_roberta_phase1"
LABELS           = {0: "no_hate_speech", 1: "hate_speech"}

# -------------------------------------------------------
# Model architecture  (identical to 05_xlm_roberta.ipynb)
# -------------------------------------------------------
class HateSpeechClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super(HateSpeechClassifier, self).__init__()
        self.roberta    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.hidden     = nn.Linear(768, 256)
        self.relu       = nn.ReLU()
        self.classifier = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x          = self.dropout(cls_output)
        x          = self.hidden(x)
        x          = self.relu(x)
        x          = self.dropout(x)
        logits     = self.classifier(x)
        return logits

# --------------------------------------------------------------
# Text cleaning  (matches clean_text_minimal from preprocessing)
# --------------------------------------------------------------
def clean_text_minimal(text: str) -> str:
    """Minimal cleaning consistent with XLM-RoBERTa preprocessing pipeline."""
    text = re.sub(r"@USER", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\brt\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bamp\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --------------------------------------------
# Global model state  (loaded once at startup)
# --------------------------------------------
model_state: dict = {}

def load_model_from_gcs():
    """Download model weights from GCS and load into HateSpeechClassifier."""
    logger.info("Downloading model weights from GCS...")
    device = torch.device("cpu")   # Cloud Run has no GPU; CPU is fine for inference

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)

    weights_blob  = bucket.blob(f"{GCS_MODEL_PREFIX}/model_weights.pt")
    weights_bytes = weights_blob.download_as_bytes()
    state_dict    = torch.load(
        io.BytesIO(weights_bytes),
        map_location=device,
        weights_only=True
    )
    logger.info("Weights downloaded.")

    logger.info("Loading tokenizer from GCS...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        for filename in ["tokenizer.json", "tokenizer_config.json"]:
            blob = bucket.blob(f"{GCS_MODEL_PREFIX}/{filename}")
            blob.download_to_filename(os.path.join(tmpdir, filename))
        tokenizer = AutoTokenizer.from_pretrained(tmpdir)
    logger.info("Tokenizer loaded.")

    logger.info("Instantiating model architecture...")
    model = HateSpeechClassifier(MODEL_NAME).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model ready.")

    return model, tokenizer, device

# -----------------------------------------
# Helper: run inference on a single text
# -----------------------------------------
def run_inference(text: str) -> dict:
    """Clean, tokenize and classify a single text. Returns a dict of results."""
    model     = model_state["model"]
    tokenizer = model_state["tokenizer"]
    device    = model_state["device"]

    text_clean = clean_text_minimal(text)

    encoding = tokenizer(
        text_clean,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs  = torch.softmax(logits, dim=1).squeeze().tolist()

    label_id   = int(torch.argmax(logits, dim=1).item())
    label      = LABELS[label_id]
    confidence = round(probs[label_id], 4)

    return {
        "text":         text,
        "text_clean":   text_clean,
        "label":        label,
        "label_id":     label_id,
        "confidence":   confidence,
        "prob_hate":    round(probs[1], 4),
        "prob_no_hate": round(probs[0], 4),
    }

# -----------------------------------------
# Lifespan: load model once at startup
# -----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_state["model"], model_state["tokenizer"], model_state["device"] = load_model_from_gcs()
    yield
    model_state.clear()

# -----------------------------------------
# FastAPI app
# -----------------------------------------
app = FastAPI(
    title="SinOdio / SansHaine",
    description="Binary hate speech classifier for Spanish (Phase 1). "
                "Fine-tuned XLM-RoBERTa on ~57k examples from Twitter and news comments.",
    version="1.0.0",
    lifespan=lifespan,
)

# -----------------------------------------
# Pydantic schemas
# -----------------------------------------
class TextInput(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Ojalá que todos los inmigrantes se vayan de aquí"},
                {"text": "Hoy ha sido un día tranquilo en el parque"},
            ]
        }
    }

class PredictionOutput(BaseModel):
    text:         str
    text_clean:   str
    label:        str
    label_id:     int
    confidence:   float
    prob_hate:    float
    prob_no_hate: float

class BatchTextInput(BaseModel):
    texts: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": [
                    "Ojalá que todos los inmigrantes se vayan de aquí",
                    "Hoy ha sido un día tranquilo en el parque",
                    "Las mujeres no deberían tener derecho a votar",
                ]}
            ]
        }
    }

class BatchPredictionOutput(BaseModel):
    results:       list[PredictionOutput]
    total:         int
    hate_count:    int
    no_hate_count: int
    hate_percent:  float

# -----------------------------------------
# Endpoints
# -----------------------------------------
@app.get("/")
def root():
    return {
        "project":       "SinOdio/SansHaine",
        "description":   "Hate speech detection API — Phase 1 (Spanish, binary)",
        "docs":          "/docs",
        "health":        "/health",
        "predict":       "/predict",
        "predict_batch": "/predict-batch",
    }

@app.get("/health")
def health():
    """Liveness check — returns OK if model is loaded."""
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: TextInput):
    """
    Classify a single text as hate speech (1) or not (0).

    - **text**: raw text in Spanish (tweets, news comments, social media posts)
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    return run_inference(payload.text)

@app.post("/predict-batch", response_model=BatchPredictionOutput)
def predict_batch(payload: BatchTextInput):
    """
    Classify a list of texts in one request.

    Returns individual predictions for each text plus a summary:
    total count, hate speech count, non-hate count and hate speech percentage.

    - **texts**: list of raw texts in Spanish (up to 50 texts per request)
    """
    if "model" not in model_state:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if len(payload.texts) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 texts per batch request."
        )

    results    = [run_inference(text) for text in payload.texts]
    total      = len(results)
    hate_count = sum(1 for r in results if r["label_id"] == 1)

    return BatchPredictionOutput(
        results       = results,
        total         = total,
        hate_count    = hate_count,
        no_hate_count = total - hate_count,
        hate_percent  = round(hate_count / total * 100, 1) if total > 0 else 0.0,
    )
