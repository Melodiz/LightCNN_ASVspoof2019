import torch
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..datasets.data_utils import genSpoof_list
from ..metrics.eer_utils import compute_eer


class ASVspoofEvaluator:
    """
    Evaluator class for ASVspoof models
    """
    
    def __init__(self, device):
        """
        Initialize the evaluator
        
        Args:
            device: Device to run evaluation on
        """
        self.device = device
    
    def run_evaluation(self, model, eval_loader, config):
        """
        Run evaluation and calculate EER like the original implementation
        
        Args:
            model: The model to evaluate
            eval_loader: DataLoader for evaluation data
            config: Configuration object
            
        Returns:
            tuple: (eer, scores_map) - Equal Error Rate and scores mapping
        """
        model.eval()
        scores_map = {}
        
        # 1. Generate scores for all files
        with torch.no_grad():
            for batch_x, batch_keys in tqdm(eval_loader, desc="Evaluating"):
                batch_x = batch_x.to(self.device)
                # Create dummy labels for AngleLinear compatibility during evaluation
                batch_size = batch_x.size(0)
                dummy_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                output = model(batch_x, dummy_labels)  # Pass dummy labels for AngleLinear compatibility
                scores = output[:, 1]  # Use second class score as spoof score
                
                for key, score in zip(batch_keys, scores.cpu().numpy()):
                    scores_map[key] = score
        
        # 2. Get ground truth labels from the protocol file
        database_path = Path.cwd() / "LA"
        protocol_path = database_path / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        
        truth_labels_dict, _, _ = genSpoof_list(protocol_path)

        # 3. Align scores with labels and calculate EER
        scores_for_eer, labels_for_eer = [], []
        for key, label in truth_labels_dict.items():
            if key in scores_map:
                scores_for_eer.append(scores_map[key])
                labels_for_eer.append(label)
            else:
                print(f"WARNING: score for key {key} not found in model output.")

        scores_for_eer = np.array(scores_for_eer)
        labels_for_eer = np.array(labels_for_eer)
        
        # Separate scores for bonafide (label 1) and spoof (label 0)
        bonafide_scores = scores_for_eer[labels_for_eer == 1]
        spoof_scores = scores_for_eer[labels_for_eer == 0]

        eer = compute_eer(bonafide_scores, spoof_scores)
        
        # Save raw scores to CSV like the original evaluate.py
        output_filename = f"raw_model_scores_epoch_{config.get('current_epoch', 'unknown')}.csv"
        with open(output_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'score'])  # Header
            for key, score in scores_map.items():
                writer.writerow([key, score])
        print(f"Raw scores saved to {output_filename}")
        
        return eer * 100, scores_map  # Convert to percentage
