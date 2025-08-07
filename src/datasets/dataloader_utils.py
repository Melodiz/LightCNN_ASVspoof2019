import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from pathlib import Path

from .asvspoof_dataset import ASVspoofTrainDataset, ASVspoofEvalDataset
from .data_utils import genSpoof_list
from ..model.lightcnn_original import LightCNNOriginal


def setup_dataloaders(config):
    """
    Setup dataloaders with class balancing
    
    Args:
        config: Configuration object containing dataloader settings
        
    Returns:
        tuple: (train_loader, eval_loader) - training and evaluation dataloaders
    """
    # Use absolute path to avoid Hydra working directory issues
    database_path = Path.cwd() / "LA"
    protocol_dir = database_path / "ASVspoof2019_LA_cm_protocols"
    
    # Training dataset
    train_data_dir = database_path / "ASVspoof2019_LA_train"
    train_protocol_path = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    train_labels_dict, train_files, train_labels_list = genSpoof_list(train_protocol_path)
    train_dataset = ASVspoofTrainDataset(train_files, train_labels_dict, train_data_dir, config)
    
    # Evaluation dataset - use 'eval' split like original
    eval_data_dir = database_path / "ASVspoof2019_LA_eval"
    eval_protocol_path = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    _, eval_files, _ = genSpoof_list(eval_protocol_path)
    eval_dataset = ASVspoofEvalDataset(eval_files, eval_data_dir, config)
    
    # Create dataloaders with WeightedRandomSampler for class balancing
    class_counts = Counter(train_labels_list)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels_list]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.dataloader.batch_size,
        sampler=sampler,
        num_workers=config.dataloader.num_workers, 
        pin_memory=config.dataloader.pin_memory,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.dataloader.batch_size,
        shuffle=False, 
        num_workers=config.dataloader.num_workers, 
        pin_memory=config.dataloader.pin_memory,
        drop_last=False
    )
    
    return train_loader, eval_loader


def setup_model(config, device):
    """
    Setup model with multi-GPU support
    
    Args:
        config: Configuration object containing model settings
        device: Device to place model on
        
    Returns:
        tuple: (model, is_data_parallel) - model and whether it's using DataParallel
    """
    model = LightCNNOriginal(
        input_shape=tuple(config.model.input_shape),
        num_classes=config.model.num_classes,
        dropout_prob=config.model.dropout_prob
    ).to(device)
    
    # Multi-GPU setup
    is_data_parallel = torch.cuda.device_count() > 1
    if is_data_parallel: 
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    return model, is_data_parallel
