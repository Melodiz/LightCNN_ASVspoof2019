import torch
import numpy as np
import random
from typing import Dict, List, Tuple


def genSpoof_list(dir_meta):
    """
    Parses a protocol file and returns a dictionary of labels, a list of
    file keys, and a raw list of the labels for weighting.
    Taken from the original ASVspoof2019 guidelines.
    """
    d_meta = {}
    file_list = []
    label_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    for line in l_meta:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        key = parts[1]
        label = parts[4]
        
        file_list.append(key)
        is_bonafide = 1 if label == "bonafide" else 0
        d_meta[key] = is_bonafide
        label_list.append(is_bonafide)
    
    return d_meta, file_list, label_list


def pad_features(x: torch.Tensor, max_len: int = 600):
    """Pads or truncates features to a specified length."""
    x_len = x.shape[1]
    if x_len >= max_len:
        return x[:, :max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(1, num_repeats)[:, :max_len]
    return padded_x


def pad_random_features(x: torch.Tensor, max_len: int = 600):
    """Pads or randomly crops features to a specified length."""
    x_len = x.shape[1]
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len + 1)
        return x[:, stt:stt + max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = x.repeat(1, num_repeats)[:, :max_len]
    return padded_x


def spec_augment(spec: torch.Tensor, freq_mask_param: int, time_mask_param: int):
    """
    Applies frequency and time masking to a spectrogram.
    """
    freq_len, time_len = spec.shape
    
    # Frequency Masking
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, freq_len - f)
    spec[f0:f0+f, :] = 0

    # Time Masking
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, time_len - t)
    spec[:, t0:t0+t] = 0
    
    return spec
