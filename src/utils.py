# utils.py
# Shared utility functions for the SinOdio/SansHaine project

import random
import numpy as np
import torch

# Global seed for reproducibility
SEED = 42

def set_seed(seed=SEED):
    """Set global seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(filepath):
    """Load dataset from given filepath.
    To be implemented in Phase 1."""
    pass

def preprocess_text(text):
    """Clean and normalize input text.
    To be implemented in Phase 1."""
    pass

def evaluate_model(y_true, y_pred):
    """Return precision, recall and F1 score.
    To be implemented in Phase 1."""
    pass