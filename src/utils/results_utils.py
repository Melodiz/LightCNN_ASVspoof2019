import csv
from pathlib import Path
from typing import Dict, Any


def save_predictions_to_csv(scores_map: Dict[str, float], output_filename: str = "raw_model_scores.csv"):
    """
    Save model predictions to CSV file
    
    Args:
        scores_map: Dictionary mapping keys to scores
        output_filename: Output CSV filename
    """
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, score in scores_map.items():
            writer.writerow([key, score])
    print(f"Predictions saved to {output_filename}")


def save_evaluation_results(eer: float, scores_map: Dict[str, float], output_dir: str = "results"):
    """
    Save comprehensive evaluation results
    
    Args:
        eer: Equal Error Rate
        scores_map: Dictionary mapping keys to scores
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save scores
    scores_file = output_path / "scores.csv"
    save_predictions_to_csv(scores_map, str(scores_file))
    
    # Save summary
    summary_file = output_path / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Equal Error Rate (EER): {eer:.4f}%\n")
        f.write(f"Number of samples: {len(scores_map)}\n")
    
    print(f"Evaluation results saved to {output_path}")
    print(f"EER: {eer:.4f}%")


def load_scores_from_csv(filename: str) -> Dict[str, float]:
    """
    Load scores from CSV file
    
    Args:
        filename: CSV file to load
        
    Returns:
        Dictionary mapping keys to scores
    """
    scores_map = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                key, score = row[0], float(row[1])
                scores_map[key] = score
    return scores_map


def print_evaluation_summary(eer: float, scores_map: Dict[str, float]):
    """
    Print a formatted evaluation summary
    
    Args:
        eer: Equal Error Rate
        scores_map: Dictionary mapping keys to scores
    """
    print("\n" + "="*50)
    print("  EVALUATION SUMMARY")
    print("="*50)
    print(f"  Equal Error Rate (EER): {eer:.4f}%")
    print(f"  Number of samples: {len(scores_map)}")
    print(f"  Score range: {min(scores_map.values()):.4f} to {max(scores_map.values()):.4f}")
    print("="*50 + "\n")
