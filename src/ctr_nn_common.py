"""Reuse the existing tabular implementation without modifying its saved recipe."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('frozen_tabular_nn', ROOT / 'src/16_train_tabular_nn.py')
tabular = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tabular)
BASE_MODEL = ROOT / 'artifacts/models/deepfm_f1_f2_f3/best_model.pt'
BASE_SCALER = ROOT / 'results/deepfm_f1_f2_f3/numeric_scaler.json'


def load_frozen(device):
    checkpoint = torch.load(BASE_MODEL, map_location='cpu', weights_only=False)
    cat, binary, numeric = [checkpoint[key] for key in ('categorical_features', 'binary_features', 'numeric_features')]
    model = tabular.TabularCTRModel(checkpoint['model_type'],
        [checkpoint['vocabulary_sizes'][k] for k in cat], len(binary)+len(numeric),
        checkpoint['embedding_dim'], checkpoint['hidden_dims'], checkpoint['dropout'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.requires_grad_(False)
    model.eval().to(device)
    scaler = json.loads(BASE_SCALER.read_text())
    assert scaler['features'] == numeric and scaler['train_only']
    return model, checkpoint, np.array(scaler['mean'],dtype=np.float32), np.array(scaler['scale'],dtype=np.float32)
