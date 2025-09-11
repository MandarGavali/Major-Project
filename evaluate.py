import os
import torch
import numpy as np
from transformers import SiglipForImageClassification, AutoImageProcessor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import argparse
import json
import logging
from train import DeepfakeDataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, data_dir: str, split: str = "test"):
    """
    Evaluate trained model on test dataset
    
    Args:
        model_path: Path to trained model directory
        data_dir: Path to dataset directory
        split: Dataset split to evaluate on ('test', 'val', or 'train')
    """
    
    # Load model and processor
    logger.info(f"Loading model from {model_path}")
    model = SiglipForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    # Set model to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load test dataset
    logger.info(f"Loading {split} dataset from {data_dir}")
    dataset = DeepfakeDataset(data_dir, processor, split=split)
    
    # Create data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Perform evaluation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            
            # Get probabilities and predictions
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Generate detailed classification report
    class_names = ["fake", "real"]
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Prepare results dictionary
    results = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": {
            "fake": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0])
            },
            "real": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1])
            }
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    return results, all_predictions, all_labels, all_probabilities

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_probability_distribution(probabilities, labels, save_path=None):
    """Plot probability distribution for predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fake class probabilities
    fake_probs = probabilities[labels == 0, 0]  # P(fake) for actual fake images
    real_probs = probabilities[labels == 1, 0]  # P(fake) for actual real images
    
    ax1.hist(fake_probs, bins=50, alpha=0.7, label='Actual Fake', color='red')
    ax1.hist(real_probs, bins=50, alpha=0.7, label='Actual Real', color='blue')
    ax1.set_xlabel('P(Fake)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Fake Class Probabilities')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Real class probabilities
    fake_real_probs = probabilities[labels == 0, 1]  # P(real) for actual fake images
    real_real_probs = probabilities[labels == 1, 1]  # P(real) for actual real images
    
    ax2.hist(fake_real_probs, bins=50, alpha=0.7, label='Actual Fake', color='red')
    ax2.hist(real_real_probs, bins=50, alpha=0.7, label='Actual Real', color='blue')
    ax2.set_xlabel('P(Real)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Real Class Probabilities')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Probability distribution plot saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained SigLIP model for deepfake detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], 
                       help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", 
                       help="Output directory for results")
    parser.add_argument("--save_plots", action="store_true", help="Save evaluation plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    logger.info(f"Evaluating model on {args.split} split")
    results, predictions, labels, probabilities = evaluate_model(
        args.model_path, args.data_dir, args.split
    )
    
    # Print results
    logger.info("=== Evaluation Results ===")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Macro F1: {results['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {results['weighted_f1']:.4f}")
    logger.info("\nPer-class metrics:")
    for class_name, metrics in results['per_class_metrics'].items():
        logger.info(f"{class_name.capitalize()}:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  Support: {metrics['support']}")
    
    # Save results
    results_path = os.path.join(args.output_dir, f"{args.split}_evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Generate and save plots if requested
    if args.save_plots:
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, f"{args.split}_confusion_matrix.png")
        plot_confusion_matrix(
            np.array(results['confusion_matrix']), 
            ["fake", "real"], 
            save_path=cm_path
        )
        
        # Probability distribution
        prob_path = os.path.join(args.output_dir, f"{args.split}_probability_distribution.png")
        plot_probability_distribution(probabilities, labels, save_path=prob_path)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
