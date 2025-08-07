#!/usr/bin/env python3
"""
Standalone evaluation script for ASVspoof models with modular architecture.
Matches the original evaluate.py from LightCNN_AVSpoof.
"""

import torch

from src.datasets import get_eval_dataloader
from src.trainer import ASVspoofEvaluator
from src.utils import load_model_from_checkpoint, save_predictions_to_csv, print_evaluation_summary


def main():
    """Main evaluation function with modular architecture"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Configuration (matching original config.yaml)
    config = {
        'model': {
            'num_classes': 2,
            'input_shape': [1, 863, 600],
            'dropout_prob': 0.75
        },
        'training': {
            'batch_size': 16,
            'num_workers': 8
        },
        'paths': {
            'dataset_path': 'LA'
        }
    }

    # Model path - you can change this
    model_path = "saved/best_model.pth"
    
    # --- Model Loading ---
    model = load_model_from_checkpoint(model_path, config, device)

    # --- Data Loading ---
    eval_loader = get_eval_dataloader(config)
    
    # --- Run Evaluation using modular evaluator ---
    evaluator = ASVspoofEvaluator(device)
    eval_eer, scores_map = evaluator.run_evaluation(model, eval_loader, config)
    
    # --- Print and save results ---
    print_evaluation_summary(eval_eer, scores_map)
    save_predictions_to_csv(scores_map)


if __name__ == "__main__":
    main()
