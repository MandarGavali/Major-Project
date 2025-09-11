import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import logging
import argparse
import json
from tqdm import tqdm
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightDeepfakeDataset(Dataset):
    """Optimized dataset loader for your specific dataset structure"""
    
    def __init__(self, data_dir: str, split: str = "Train", transform=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Your dataset structure: Dataset/Train/Fake, Dataset/Train/Real, etc.
        split_dir = os.path.join(data_dir, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Load fake images (label 0)
        fake_dir = os.path.join(split_dir, "Fake")
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            # Limit samples if specified (for faster training on your PC)
            if max_samples_per_class:
                fake_files = fake_files[:max_samples_per_class]
                
            for img_name in fake_files:
                self.images.append(os.path.join(fake_dir, img_name))
                self.labels.append(0)
        
        # Load real images (label 1)
        real_dir = os.path.join(split_dir, "Real")
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            # Limit samples if specified
            if max_samples_per_class:
                real_files = real_files[:max_samples_per_class]
                
            for img_name in real_files:
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
            # Load and convert image
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return a dummy sample
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, torch.tensor(label, dtype=torch.long)

class OptimizedDeepfakeModel(nn.Module):
    """Lightweight but effective model for deepfake detection"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(OptimizedDeepfakeModel, self).__init__()
        
        # Use EfficientNet-B0 - good balance of speed and accuracy
        # If EfficientNet not available, fall back to MobileNet
        try:
            from torchvision.models import efficientnet_b0
            self.backbone = efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            # Fallback to MobileNetV2
            logger.info("Using MobileNetV2 as backbone")
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with progress tracking"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_targets

def main():
    parser = argparse.ArgumentParser(description="Train optimized model for your deepfake dataset")
    parser.add_argument("--data_dir", type=str, default="./Dataset", help="Path to your dataset directory")
    parser.add_argument("--output_dir", type=str, default="./my_trained_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples per class for faster training")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Training on max {args.max_samples} samples per class for efficiency")
    
    # Data transforms - optimized for deepfake detection
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with your data
    logger.info("Loading your dataset...")
    train_dataset = LightweightDeepfakeDataset(
        args.data_dir, split="Train", transform=train_transform, 
        max_samples_per_class=args.max_samples
    )
    val_dataset = LightweightDeepfakeDataset(
        args.data_dir, split="Validation", transform=val_transform, 
        max_samples_per_class=args.max_samples//2
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    logger.info("Creating optimized model...")
    model = OptimizedDeepfakeModel(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_accuracies = []
    
    logger.info("Starting training on YOUR dataset...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch+1)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log results
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'model_config': {
                    'num_classes': 2,
                    'architecture': 'OptimizedDeepfakeModel'
                }
            }, os.path.join(args.output_dir, 'best_deepfake_model.pth'))
            
            logger.info(f"✅ New best validation accuracy: {best_val_acc:.2f}%")
    
    # Training summary
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 TRAINING COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"⏱️  Total training time: {total_time/60:.1f} minutes")
    logger.info(f"🎯 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"📁 Model saved to: {args.output_dir}/best_deepfake_model.pth")
    
    # Generate classification report
    logger.info(f"\n📊 Final Classification Report:")
    print(classification_report(val_targets, val_preds, target_names=['Fake', 'Real']))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'total_training_time': total_time,
        'dataset_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        }
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"📈 Training history saved to: {args.output_dir}/training_history.json")
    logger.info(f"\n🚀 Next step: Use app_with_custom_model.py to test your trained model!")

if __name__ == "__main__":
    main()
