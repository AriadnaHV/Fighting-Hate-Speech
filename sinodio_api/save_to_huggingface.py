"""
save_to_huggingface.py
----------------------
One-time script to re-save the model in HuggingFace format and push to HF Hub.
Run this locally (not on the VM) after downloading the weights from GCS.

This future-proofs the model: once on HuggingFace Hub, it's free forever
and independent of GCS credits.

Steps:
  1. pip install huggingface_hub
  2. Create a free account at huggingface.co
  3. Create a new model repository (e.g. your-username/sinodio-xlm-roberta)
  4. Run: python save_to_huggingface.py
"""

import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
import io
import tempfile

# -- Config --------------------------------------------------------
HF_REPO_ID       = "your-username/sinodio-xlm-roberta"   # <-- edit this
MODEL_NAME       = "xlm-roberta-base"
GCS_BUCKET       = "sinodio-models"
GCS_MODEL_PREFIX = "phase1/xlm_roberta/xlm_roberta_phase1"

# -- Model class (identical to training notebook) ------------------
class HateSpeechClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.3):
        super().__init__()
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

# -- 1. Download weights from GCS ----------------------------------
print("Downloading from GCS...")
client = storage.Client()
bucket = client.bucket(GCS_BUCKET)

weights_blob  = bucket.blob(f"{GCS_MODEL_PREFIX}/model_weights.pt")
weights_bytes = weights_blob.download_as_bytes()
state_dict    = torch.load(io.BytesIO(weights_bytes), map_location="cpu", weights_only=True)

with tempfile.TemporaryDirectory() as tmpdir:
    for filename in ["tokenizer.json", "tokenizer_config.json"]:
        bucket.blob(f"{GCS_MODEL_PREFIX}/{filename}").download_to_filename(
            os.path.join(tmpdir, filename)
        )
    tokenizer = AutoTokenizer.from_pretrained(tmpdir)

# -- 2. Load into model and convert to HF format ------------------
print("Loading model...")
model = HateSpeechClassifier(MODEL_NAME)
model.load_state_dict(state_dict)
model.eval()

# -- 3. Save locally in HuggingFace format ------------------------
save_dir = "sinodio_hf_model"
os.makedirs(save_dir, exist_ok=True)

# Save the backbone (XLM-RoBERTa part) with HF save_pretrained
# We wrap state dict keys to match HF expectations
model.roberta.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# Save the custom head weights separately
torch.save({
    "hidden.weight":     state_dict["hidden.weight"],
    "hidden.bias":       state_dict["hidden.bias"],
    "classifier.weight": state_dict["classifier.weight"],
    "classifier.bias":   state_dict["classifier.bias"],
}, f"{save_dir}/classification_head.pt")

print(f"Model saved locally to {save_dir}/")

# -- 4. Push to HuggingFace Hub ----------------------------------
from huggingface_hub import HfApi, login

login()   # will prompt for your HF token
api = HfApi()

api.upload_folder(
    folder_path=save_dir,
    repo_id=HF_REPO_ID,
    repo_type="model",
)
print(f"Model uploaded to https://huggingface.co/{HF_REPO_ID}")
