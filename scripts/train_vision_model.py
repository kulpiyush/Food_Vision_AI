"""
Vision Model Training Script
Step 2.2: Fine-tune EfficientNet-B0 on Indian cuisine dataset

This script fine-tunes a pretrained EfficientNet-B0 model on Indian food images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
from pathlib import Path
from tqdm import tqdm
import argparse
from models.vision_model import INDIAN_FOOD_CLASSES

# Training hyperparameters
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data_transforms():
    """Get data augmentation transforms for training and validation"""
    
    # Training: augmentation for better generalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation: no augmentation, just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir, batch_size=32):
    """
    Create data loaders for training and validation
    
    Args:
        data_dir (str): Path to dataset directory (should have train/ and val/ subdirectories)
        batch_size (int): Batch size for training
    
    Returns:
        train_loader, val_loader, num_classes, class_names
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(f"Dataset structure not found. Expected {train_dir} and {val_dir}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(root=str(val_dir), transform=val_transform)
    
    # Get class names and count
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"✅ Found {num_classes} classes: {class_names}")
    print(f"✅ Train images: {len(train_dataset)}")
    print(f"✅ Val images: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, num_classes, class_names


def create_model(num_classes, device):
    """
    Create EfficientNet-B0 model with custom classifier
    
    Args:
        num_classes (int): Number of food classes
        device: torch device
    
    Returns:
        model, criterion, optimizer
    """
    import torchvision.models as models
    
    # Load pretrained EfficientNet-B0
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except (AttributeError, TypeError):
        model = models.efficientnet_b0(pretrained=True)
    
    # Modify classifier head
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    # Move to device
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    return model, criterion, optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/(total//labels.size(0)):.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/(total//labels.size(0)):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    data_dir="data/training_data",
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    output_path="models/weights/food_classifier_indian.pth",
    device=None
):
    """
    Main training function
    
    Args:
        data_dir (str): Path to dataset
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        output_path (str): Path to save model
        device: torch device (auto-detect if None)
    """
    if device is None:
        device = torch.device(DEFAULT_DEVICE)
    
    print("=" * 60)
    print("Fine-tuning EfficientNet-B0 on Indian Cuisine")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    
    # Create data loaders
    print("\n📦 Loading dataset...")
    train_loader, val_loader, num_classes, class_names = create_data_loaders(
        data_dir, batch_size
    )
    
    # Create model
    print("\n🤖 Creating model...")
    model, criterion, optimizer, scheduler = create_model(num_classes, device)
    print(f"✅ Model created with {num_classes} classes")
    
    # Training loop
    print("\n🚀 Starting training...")
    best_val_acc = 0.0
    train_history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate step
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'num_classes': num_classes
            }, output_path)
            print(f"✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        # History
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_path}")
    print("=" * 60)
    
    return model, train_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Indian cuisine")
    parser.add_argument("--data_dir", type=str, default="data/training_data",
                       help="Path to training data directory")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                       help="Learning rate")
    parser.add_argument("--output", type=str, default="models/weights/food_classifier_indian.pth",
                       help="Path to save model")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.data_dir).exists():
        print(f"❌ Dataset not found at {args.data_dir}")
        print("\nPlease:")
        print("1. Collect images for each food class")
        print("2. Organize them in train/val folders")
        print("3. Run: python scripts/prepare_dataset.py verify")
        exit(1)
    
    # Train
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_path=args.output
    )

