import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .data_utils import genSpoof_list, pad_features, pad_random_features, spec_augment


class ASVspoofTrainDataset(Dataset):
    """ASVspoof training dataset with ultra-fast optimizations"""
    
    def __init__(self, list_IDs: List[str], labels: Dict[str, int], base_dir: Path, config: Dict):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        
        # STFT parameters - optimized for performance
        self.n_fft = 1724
        self.win_length = 1724
        self.hop_length = 130
        self.feature_len = 600
        
        # CRITICAL: Pre-compute window to avoid recreation overhead
        self.window = torch.blackman_window(self.win_length)
        
        # Augmentation parameters
        self.noise_factor = 0.005
        self.freq_mask_param = 40
        self.time_mask_param = 30

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        
        # Noise Injection
        if random.random() < 0.5:
            noise = np.random.randn(len(X))
            X = X + self.noise_factor * noise
        
        X = torch.from_numpy(X).float()

        # STFT with pre-computed window for performance
        X_stft = torch.stft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            return_complex=True)
        
        X_spec = torch.log(torch.abs(X_stft) ** 2 + 1e-6)
        
        # SpecAugment
        if random.random() < 0.7:
            X_spec = spec_augment(X_spec, self.freq_mask_param, self.time_mask_param)

        X_pad = pad_random_features(X_spec, self.feature_len)
        y = self.labels[key]
        return X_pad, torch.tensor(y, dtype=torch.long)


class ASVspoofEvalDataset(Dataset):
    """ASVspoof evaluation dataset with ultra-fast optimizations"""
    
    def __init__(self, list_IDs: List[str], base_dir: Path, config: Dict):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        
        # STFT parameters - optimized for performance
        self.n_fft = 1724
        self.win_length = 1724
        self.hop_length = 130
        self.feature_len = 600
        
        # CRITICAL: Pre-compute window to avoid recreation overhead
        self.window = torch.blackman_window(self.win_length)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X = torch.from_numpy(X).float()

        # STFT with pre-computed window for performance
        X_stft = torch.stft(X, n_fft=self.n_fft, hop_length=self.hop_length,
                            win_length=self.win_length, window=self.window,
                            return_complex=True)

        X_spec = torch.log(torch.abs(X_stft) ** 2 + 1e-6)
        X_pad = pad_features(X_spec, self.feature_len)
        return X_pad, key 