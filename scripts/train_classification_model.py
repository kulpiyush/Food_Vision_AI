#!/usr/bin/env python3
"""
Train Classification Model on Khana Dataset
Trains EfficientNet/ResNet on Indian food classification dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import os

def is_image_file(filename):
    """Check if a file is an image, even without extension"""
    # Standard image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    filename_lower = filename.lower()
    
    # Check if it has a known image extension
    if any(filename_lower.endswith(ext) for ext in image_extensions):
        return True
    
    # Check if it's a non-image file (skip these)
    non_image_extensions = ('.txt', '.csv', '.json', '.yaml', '.yml', '.xml')
    if any(filename_lower.endswith(ext) for ext in non_image_extensions):
        return False
    
    # If no extension, try to verify it's an image by opening it
    try:
        with Image.open(filename) as img:
            img.verify()
        return True
    except:
        return False


def get_data_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_model(model_name="efficientnet_b0", num_classes=None):
    """Create model architecture"""
    if model_name.startswith("efficientnet"):
        variant = model_name.replace("efficientnet_", "").upper()
        try:
            weights = getattr(models, f"EfficientNet_{variant}_Weights").DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes)
        )
        
    elif model_name.startswith("resnet"):
        if "50" in model_name:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, num_classes)
            
    elif model_name.startswith("mobilenet"):
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def train_classification_model(
    data_dir="data/training_data",
    model_name="efficientnet_b0",
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    output_dir="models/weights"
):
    """Train classification model on Khana dataset"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_path}")
    
    # Check for train/val structure
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(f"Expected train/ and val/ directories in {data_path}")
    
    print(f"Training {model_name} on Khana dataset")
    print(f"Data directory: {data_path}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get transforms
    train_transform, val_transform = get_data_transforms()
    
    # Load datasets
    print("Loading datasets...")
    # Use ImageFolder with custom is_valid_file to handle files without extensions
    train_dataset = ImageFolder(str(train_dir), transform=train_transform, is_valid_file=is_image_file)
    val_dataset = ImageFolder(str(val_dir), transform=val_transform, is_valid_file=is_image_file)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    print(f"Found {num_classes} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Save class names
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    class_names_file = output_path / "class_names.txt"
    with open(class_names_file, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Saved class names to {class_names_file}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"Creating {model_name} model...")
    model = create_model(model_name, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    train_history = []
    
    print("\nStarting training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        train_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = output_path / "food_classifier.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes,
                'class_names': class_names,
                'model_name': model_name,
                'epoch': epoch + 1,
                'val_acc': val_acc
            }, model_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.2f}%) to {model_path}")
    
    # Save training history
    history_path = output_path / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"\n✅ Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {output_path / 'food_classifier.pt'}")
    print(f"Class names saved to: {class_names_file}")
    print(f"Training history saved to: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification model on Khana dataset")
    parser.add_argument("--data", type=str, default="data/training_data",
                       help="Path to training data directory (should have train/ and val/ subdirectories)")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                       choices=["efficientnet_b0", "resnet50", "mobilenet_v2"],
                       help="Model architecture")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--output", type=str, default="models/weights",
                       help="Output directory for saved model")
    
    args = parser.parse_args()
    
    train_classification_model(
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output
    )

