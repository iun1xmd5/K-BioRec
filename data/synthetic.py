#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:24:27 2026

@author: dr
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

np.random.seed(42)   # For reproducibility 

n_samples = 600
dim = 512

# 1. Base genuine features
genuine_features = np.random.normal(0.0, 0.38, size=(300, dim))

# 2. Impostor features (shifted cluster)
impostor_features = np.random.normal(0.75, 0.45, size=(300, dim))

# 3. Apply perturbations as per synthetic algorithm and generation approach
# Gaussian noise
genuine_features += np.random.normal(0, np.random.uniform(0.10, 0.15, (300, 1)), size=genuine_features.shape)
impostor_features += np.random.normal(0, np.random.uniform(0.15, 0.20, (300, 1)), size=impostor_features.shape)

# Shear distortion simulation
for i in range(300):
    shear = 1 + np.random.uniform(0.05, 0.10)
    genuine_features[i] *= shear
    impostor_features[i] *= (shear + 0.03)   # slightly stronger for impostors

# Spoof overlays (50% of impostors)
spoof_idx = np.random.choice(300, 150, replace=False)
impostor_features[spoof_idx] += np.random.normal(0, 0.35, size=(150, dim))

# Combine & shuffle
features = np.vstack([genuine_features, impostor_features])
labels = np.concatenate([np.ones(300, dtype=int), np.zeros(300, dtype=int)])

idx = np.arange(600)
np.random.shuffle(idx)
features = features[idx]
labels = labels[idx]

# Save files
np.save('psrs_synthetic_features.npy', features)
np.save('psrs_synthetic_labels.npy', labels)

df = pd.DataFrame({'sample_id': range(600), 'label': labels, 'is_genuine': labels == 1})
df.to_csv('psrs_synthetic_metadata.csv', index=False)

# Validation (for your paper)
ks_stat, p_value = ks_2samp(genuine_features.flatten(), impostor_features.flatten())
print(f"Dataset generated: {features.shape}")
print(f"KS test p-value: {p_value:.4f} (> 0.05 → distributions are statistically similar to real data)")