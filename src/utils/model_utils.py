import torch
from collections import OrderedDict
from pathlib import Path

from ..model.lightcnn_original import LightCNNOriginal


def load_model_from_checkpoint(model_path, config, device):
    """
    Load model from checkpoint with proper state dict cleaning
    
    Args:
        model_path: Path to the checkpoint file
        config: Configuration dictionary containing model parameters
        device: Device to load model on
        
    Returns:
        model: Loaded model
        
    Raises:
        ValueError: If model path doesn't exist
    """
    if not Path(model_path).exists():
        raise ValueError(f"Model path not found: {model_path}")
    
    print(f"Loading model from checkpoint: {model_path}")
    model = LightCNNOriginal(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        dropout_prob=config['model']['dropout_prob']
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Clean state_dict 'cause it was saved from DataParallel (in my case)
    # Should be ok for single GPU (non-parallel) checkpoints
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully.")
    
    return model


def create_model_from_config(config, device):
    """
    Create a new model instance from configuration
    
    Args:
        config: Configuration dictionary containing model parameters
        device: Device to place model on
        
    Returns:
        model: Created model instance
    """
    model = LightCNNOriginal(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        dropout_prob=config['model']['dropout_prob']
    ).to(device)
    
    return model


def save_model_checkpoint(model, optimizer, scheduler, epoch, best_eer, save_path):
    """
    Save model checkpoint with all necessary state
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        best_eer: Best EER achieved
        save_path: Path to save checkpoint
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save like working implementation
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_eer': best_eer,
    }, save_path)
    
    print(f"Checkpoint saved to {save_path}")
