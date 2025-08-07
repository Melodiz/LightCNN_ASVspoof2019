from pathlib import Path
from torch.utils.data import DataLoader

from .asvspoof_dataset import ASVspoofEvalDataset
from .data_utils import genSpoof_list


def get_eval_dataloader(config):
    """
    Create evaluation dataloader like the original implementation
    
    Args:
        config: Configuration dictionary containing dataloader settings
        
    Returns:
        eval_loader: DataLoader for evaluation data
    """
    database_path = Path.cwd() / "LA"
    protocol_dir = database_path / "ASVspoof2019_LA_cm_protocols"
    
    # Evaluation dataset - use 'eval' split like original
    eval_data_dir = database_path / "ASVspoof2019_LA_eval"
    eval_protocol_path = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    _, eval_files, _ = genSpoof_list(eval_protocol_path)
    eval_dataset = ASVspoofEvalDataset(eval_files, eval_data_dir, config)
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False, 
        num_workers=config['training']['num_workers'], 
        pin_memory=True,
        drop_last=False
    )
    
    return eval_loader


def get_eval_dataloader_from_config(config):
    """
    Create evaluation dataloader from Hydra config
    
    Args:
        config: Hydra configuration object
        
    Returns:
        eval_loader: DataLoader for evaluation data
    """
    database_path = Path.cwd() / "LA"
    protocol_dir = database_path / "ASVspoof2019_LA_cm_protocols"
    
    # Evaluation dataset - use 'eval' split like original
    eval_data_dir = database_path / "ASVspoof2019_LA_eval"
    eval_protocol_path = protocol_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    _, eval_files, _ = genSpoof_list(eval_protocol_path)
    eval_dataset = ASVspoofEvalDataset(eval_files, eval_data_dir, config)
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.dataloader.batch_size,
        shuffle=False, 
        num_workers=config.dataloader.num_workers, 
        pin_memory=config.dataloader.pin_memory,
        drop_last=False
    )
    
    return eval_loader
