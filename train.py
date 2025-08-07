#!/usr/bin/env python3
"""
Main training script for ASVspoof with modular architecture
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.utils import instantiate

from src.datasets import setup_dataloaders, setup_model
from src.trainer import ASVspoofTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


@hydra.main(config_path="src/configs", config_name="asvspoof_baseline", version_base=None)
def main(config: DictConfig):
    """Main training function with modular architecture"""
    # Setup logging and saving
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)
    
    # Setup
    set_random_seed(config.get('seed', 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Disable cudnn benchmark for better performance
    torch.backends.cudnn.benchmark = False
    
    # Setup dataloaders
    train_loader, eval_loader = setup_dataloaders(config)
    
    # Setup model
    model, is_data_parallel = setup_model(config, device)
    
    print(model)
    print(f"All parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Setup training components
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler.gamma)
    
    # Initialize trainer
    trainer = ASVspoofTrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        writer=writer,
        config=config
    )
    
    # Start training
    trainer.train(num_epochs=config.epochs)


if __name__ == "__main__":
    main()
