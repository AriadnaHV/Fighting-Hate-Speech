# SinOdio / SansHaine
### Automatic Hate Speech Detection in Spanish (architecture ready for extension to French)

[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-green)]()
[![Model](https://img.shields.io/badge/Model-XLM--RoBERTa-blue)]()
[![API](https://img.shields.io/badge/API-Live%20on%20Cloud%20Run-brightgreen)]()
[![Bootcamp](https://img.shields.io/badge/KeepCoding-XVI%20Edición-orange)]()

## Description

Automatic hate speech classifier for Spanish, with a planned extension to French
in the next phase, developed as the final project of the *KeepCoding Big Data, 
Artificial Intelligence & Machine Learning Bootcamp, 16th Edition*. Built using XLM-RoBERTa
with transfer learning, designed as a support tool for NGOs and human rights
organizations working with Spanish social media content.

The project is structured in four incremental phases. Phase 1 (binary hate
speech classifier for Spanish) is complete and deployed as a live REST API.
The multilingual architecture (XLM-RoBERTa, pretrained on 100 languages) was
chosen deliberately to enable extension to French in Phase 2 without architectural changes.

---

## 🚀 Live API

The Phase 1 model is deployed on Google Cloud Run and available at:

**https://sinodio-api-1064831214069.europe-west4.run.app**

Interactive documentation (Swagger UI):
**https://sinodio-api-1064831214069.europe-west4.run.app/docs**

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Project info and endpoint index |
| GET | `/health` | Liveness check |
| POST | `/predict` | Classify a single text |
| POST | `/predict-batch` | Classify up to 50 texts with summary statistics |

### Quick example

```python
import requests

response = requests.post(
    "https://sinodio-api-1064831214069.europe-west4.run.app/predict",
    json={"text": "Ojalá que todos los inmigrantes se vayan de aquí"}
)
print(response.json())
# {"label": "hate_speech", "confidence": 0.9045, "prob_hate": 0.9045, ...}
```

---

## 📊 Results (Phase 1)

| Model | F1 Macro | Accuracy | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Logistic Regression (baseline) | 0.745 | 0.775 | 0.739 | 0.754 |
| XLM-RoBERTa (fine-tuned) | **0.810** | **0.840** | **0.814** | **0.807** |

XLM-RoBERTa improves F1 macro by 6.5 percentage points over the TF-IDF baseline,
with particularly strong gains on the hate speech class (F1: 0.65 → 0.74).

Training corpus: ~57,000 examples from three Spanish datasets (Twitter, news
comments, humor) covering diverse registers and target groups.

---

## 🗂️ Project Structure

```
Fighting-Hate-Speech/
│
├── data/
│   ├── raw/             # original datasets (stored in GCS)
│   ├── processed/       # preprocessed splits in Parquet format
│   └── annotated/       # manually annotated corpus (Phase 3)
│
├── notebooks/
│   ├── phase1/          # binary classifier in Spanish
│   ├── phase2/          # extension to French
│   ├── phase3/          # multiclass classifier
│   └── phase4/          # counter-narrative retrieval
│
├── sinodio_api/         # REST API deployment
│   ├── main.py          # FastAPI application
│   ├── Dockerfile
│   └── requirements_api.txt
│
├── src/
│   └── utils.py         # shared utility functions (seed, preprocessing)
│
├── models/              # trained models (stored in GCS)
├── reports/             # technical report ("memoria")│
├── requirements.txt     # local development environment
└── requirements_vm.txt  # GCP VM training environment
```

---

## ⚙️ Setup

### Local development environment

```bash
git clone https://github.com/AriadnaHV/Fighting-Hate-Speech.git
cd Fighting-Hate-Speech
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### GCP VM environment (for XLM-RoBERTa training)

PyTorch must be installed separately before the remaining dependencies,
as its CUDA builds are not available on the standard PyPI index:

```bash
pip3 install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements_vm.txt
```

### API deployment (Google Cloud Run)

See `sinodio_api/deploy.sh` for the full deployment script. Requirements:
- Google Cloud SDK (`gcloud`) installed and authenticated
- Access to the `sinodio-models` GCS bucket

```bash
cd sinodio_api
gcloud builds submit \
  --tag europe-west4-docker.pkg.dev/project-5c89dcac-34cb-453d-bd7/sinodio-api/sinodio-api .
gcloud run deploy sinodio-api \
  --image europe-west4-docker.pkg.dev/project-5c89dcac-34cb-453d-bd7/sinodio-api/sinodio-api \
  --region europe-west4 --memory 8Gi --cpu 4 --allow-unauthenticated
```

---

## 🗺️ Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Binary hate speech classifier in Spanish | ✅ Complete |
| 2 | Extension to French | 🔄 Planned |
| 3 | Multiclass classifier by target group | 🔄 Planned |
| 4 | Counter-narrative retrieval | 🔄 Planned |

---

## 🏗️ Infrastructure

- **Cloud Storage**: raw data, processed splits, trained models (`europe-west4`)
- **BigQuery**: experiment results and model metrics (`sinodio_results` dataset)
- **Cloud Run**: REST API serving (8Gi RAM, 4 vCPU, `europe-west4`)
- **Artifact Registry**: Docker images
- **GitHub Projects**: SCRUM board with 6 milestones and issue tracking

---

## 📚 Datasets

| Status | Dataset | Language | Examples | Source |
|--------|---------|----------|----------|--------|
| ✅ | Spanish Hate Speech Superset | ES | 29,855 | Twitter |
| ✅ | DETOXIS | ES | 3,463 | News comments |
| ✅ | HAHA | ES | 24,000 | Humor / social media |
| 🔄 | CONAN | FR | TBD | NGO-validated (Phases 2 & 4) |
| 🔄 | OLID-FR | FR | TBD | Twitter (Phase 2) |
| 🔄 | Corpus propio | ES | 200-400 | News scraping (Phase 3) |

---

## 🎓 Context

Final project for the **KeepCoding Bootcamp | Big Data, Inteligencia Artificial & Machine Learning | Edición XVI**
at [KeepCoding](https://keepcoding.io). Developed individually by Ariane Heinz Vallribera.

