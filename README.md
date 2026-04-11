# SinOdio/SansHaine
### Automatic Hate Speech Detection in Spanish and French

## Description
Multilingual hate speech classifier targeting vulnerable groups 
including women, immigrants, regional minorities and LGBTQ+ 
communities. Built using XLM-RoBERTa with transfer learning, 
designed as a support tool for NGOs and human rights organizations.

## Project Structure
- `data/` — raw, processed and annotated datasets
- `notebooks/` — Jupyter notebooks organized by phase
- `src/` — shared utility functions
- `models/` — trained models
- `reports/` — PowerBI dashboards and visualizations

## Phases
- Phase 1: Binary classifier in Spanish (core)
- Phase 2: Extension to French
- Phase 3: Multiclass classifier
- Phase 4: Counter-narrative retrieval

## Setup
*(to be completed)*
### GCP VM Setup (for XLM-RoBERTa training)
PyTorch must be installed separately before the VM requirements:
```bash
pip3 install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip3 install -r requirements_vm.txt
```


## Results
*(to be completed)*

## License
*(to be completed)*