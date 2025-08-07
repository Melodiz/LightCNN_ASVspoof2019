import torch
import torch.nn as nn
import time
from pathlib import Path
from tqdm import tqdm

from .evaluator import ASVspoofEvaluator


class ASVspoofTrainer:
    """
    Trainer class for ASVspoof LightCNN model
    """
    
    def __init__(self, model, train_loader, eval_loader, criterion, optimizer, 
                 scheduler, device, writer, config):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            eval_loader: DataLoader for evaluation data
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            writer: Experiment writer (CometML)
            config: Configuration object
        """
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.writer = writer
        self.config = config
        
        # Initialize evaluator
        self.evaluator = ASVspoofEvaluator(device)
        
        # Training state
        self.best_eer = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch, num_epochs):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            num_epochs: Total number of epochs
            
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_correct = 0.0
        num_total = 0.0
        
        progress_bar = tqdm(self.train_loader, desc="Training", unit="batch")
        epoch_start_time = time.time()
        
        for batch_idx, (batch_x, batch_y) in enumerate(progress_bar):
            batch_start_time = time.time()
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Direct model call for maximum performance
            output = self.model(batch_x, batch_y)
            
            loss = self.criterion(output, batch_y)
            loss.backward()
            
            self.optimizer.step()
            
            # Calculate batch time
            batch_time = time.time() - batch_start_time
            steps_per_sec = 1.0 / batch_time if batch_time > 0 else 0
            
            running_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(output.data, 1)
            num_total += batch_y.size(0)
            num_correct += (preds == batch_y).sum().item()
            
            # Log batch metrics to CometML (EVERY batch)
            self.writer.set_step(self.global_step, mode="train")
            self.writer.add_scalar("batch_loss", loss.item())
            self.writer.add_scalar("batch_accuracy", (num_correct/num_total)*100)
            self.writer.add_scalar("steps_per_sec", steps_per_sec)
            self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0])
            
            self.global_step += 1
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(num_correct/num_total)*100:.2f}%",
                speed=f"{steps_per_sec:.1f} steps/s"
            )
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        train_loss = running_loss / len(self.train_loader.dataset)
        train_acc = (num_correct / num_total) * 100
        
        return {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'epoch_time': epoch_time
        }
    
    def evaluate_epoch(self, epoch):
        """
        Evaluate the model for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (eer, scores_map) - Equal Error Rate and scores mapping
        """
        self.config.current_epoch = epoch  # Pass epoch info for score saving
        eval_eer, scores_map = self.evaluator.run_evaluation(
            self.model, self.eval_loader, self.config
        )
        return eval_eer, scores_map
    
    def save_checkpoint(self, epoch, is_data_parallel=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch number
            is_data_parallel: Whether model is using DataParallel
        """
        checkpoint_path = Path("saved/best_model.pth")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"New best model found! Saving to {checkpoint_path}")
        
        # Save like working implementation
        model_state = self.model.module.state_dict() if is_data_parallel else self.model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eer': self.best_eer,
        }, checkpoint_path)
        
        # Log checkpoint to CometML
        if self.config.writer.log_checkpoints:
            self.writer.add_checkpoint(str(checkpoint_path), str(checkpoint_path.parent))
    
    def train(self, num_epochs):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train for
        """
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Training
            train_metrics = self.train_epoch(epoch, num_epochs)
            print(f"Epoch {epoch} | Train Loss: {train_metrics['train_loss']:.4f} | Train Acc: {train_metrics['train_acc']:.2f}%")
            
            # Evaluation
            eval_eer, scores_map = self.evaluate_epoch(epoch)
            print(f"Epoch {epoch} | Eval EER: {eval_eer:.2f}%")
            
            # Log epoch metrics to CometML
            self.writer.set_step(epoch, mode="train")
            self.writer.add_scalar("epoch_loss", train_metrics['train_loss'])
            self.writer.add_scalar("epoch_accuracy", train_metrics['train_acc'])
            self.writer.add_scalar("epoch_time", train_metrics['epoch_time'])
            self.writer.set_step(epoch, mode="eval")
            self.writer.add_scalar("epoch_eer", eval_eer)
            
            # Save best model
            if eval_eer < self.best_eer:
                self.best_eer = eval_eer
                self.save_checkpoint(epoch, hasattr(self.model, 'module'))
            
            self.scheduler.step()
        
        print(f"\n--- Training Finished ---")
        print(f"Best Eval EER: {self.best_eer:.2f}%")
        
        # Log final metrics
        self.writer.set_step(num_epochs, mode="final")
        self.writer.add_scalar("best_eval_eer", self.best_eer)
        
        # End the experiment
        if hasattr(self.writer, 'exp') and self.writer.exp is not None:
            self.writer.exp.end()
