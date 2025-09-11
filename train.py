import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    SiglipForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from datasets import load_dataset
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDataset(Dataset):
    """Custom dataset for deepfake detection training"""
    
    def __init__(self, data_dir: str, processor, split: str = "train", transform=None):
        """
        Args:
            data_dir: Path to dataset directory
            processor: Image processor for the model
            split: 'train', 'val', or 'test'
            transform: Additional transforms to apply
        """
        self.data_dir = data_dir
        self.processor = processor
        self.split = split
        self.transform = transform
        
        # Expected directory structure:
        # data_dir/
        #   ├── train/
        #   │   ├── fake/
        #   │   └── real/
        #   ├── val/
        #   │   ├── fake/
        #   │   └── real/
        #   └── test/
        #       ├── fake/
        #       └── real/
        
        self.images = []
        self.labels = []
        
        # Map split names to match dataset structure
        split_mapping = {
            "train": "Train",
            "val": "Validation", 
            "test": "Test"
        }
        
        actual_split = split_mapping.get(split, split)
        split_dir = os.path.join(data_dir, actual_split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Load fake images (label 0) - check both "fake" and "Fake"
        fake_dirs = [os.path.join(split_dir, "fake"), os.path.join(split_dir, "Fake")]
        for fake_dir in fake_dirs:
            if os.path.exists(fake_dir):
                for img_name in os.listdir(fake_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.images.append(os.path.join(fake_dir, img_name))
                        self.labels.append(0)
        
        # Load real images (label 1) - check both "real" and "Real"
        real_dirs = [os.path.join(split_dir, "real"), os.path.join(split_dir, "Real")]
        for real_dir in real_dirs:
            if os.path.exists(real_dir):
                for img_name in os.listdir(real_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        self.images.append(os.path.join(real_dir, img_name))
                        self.labels.append(1)
        
        logger.info(f"Loaded {len(self.images)} images for {split} split")
        logger.info(f"Fake images: {self.labels.count(0)}, Real images: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            # Process image using the processor
            inputs = self.processor(images=image, return_tensors="pt")
            
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a dummy sample in case of error
            dummy_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(images=dummy_image, return_tensors="pt")
            return {
                "pixel_values": inputs["pixel_values"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)
            }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_config(config_path: str) -> Dict:
    """Load training configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        return {
            "model_name": "google/siglip-base-patch16-512",
            "num_epochs": 10,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 100,
            "save_total_limit": 3,
            "load_best_model_at_end": True,
            "metric_for_best_model": "f1",
            "greater_is_better": True,
            "early_stopping_patience": 3
        }

def main():
    parser = argparse.ArgumentParser(description="Train SigLIP model for deepfake detection")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Training configuration: {json.dumps(config, indent=2)}")
    
    # Initialize processor and model
    model_name = config["model_name"]
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Load model with custom labels
    model = SiglipForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "fake", 1: "real"},
        label2id={"fake": 0, "real": 1}
    )
    
    # Create datasets
    train_dataset = DeepfakeDataset(args.data_dir, processor, split="train")
    val_dataset = DeepfakeDataset(args.data_dir, processor, split="val")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        push_to_hub=False,
        report_to=None,  # Disable wandb/tensorboard for now
        learning_rate=config["learning_rate"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config["fp16"],
        dataloader_num_workers=config["dataloader_num_workers"],
        seed=config["seed"],
        max_grad_norm=config["max_grad_norm"],
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])]
    )
    
    # Start training
    logger.info("Starting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    # Save the final model
    logger.info("Saving model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Evaluate on test set if available
    test_dir = os.path.join(args.data_dir, "test")
    if os.path.exists(test_dir):
        logger.info("Evaluating on test set...")
        test_dataset = DeepfakeDataset(args.data_dir, processor, split="test")
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        logger.info(f"Test results: {test_results}")
        
        # Save test results
        with open(f"{args.output_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
