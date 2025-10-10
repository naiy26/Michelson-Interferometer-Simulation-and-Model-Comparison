"""
Michelson Interferometer d-value Prediction
Train multiple neural network models to predict d values from interference patterns
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for faster training


class MichelsonDataset(Dataset):
    """Custom Dataset for Michelson Interferometer images"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def load_dataset(data_dir, start_um=1.0, end_um=5.0, step_um=0.01, use_preprocessed=True):
    """
    Load and prepare the dataset by sorting images and assigning labels
    
    Args:
        data_dir: Directory containing images
        start_um: Starting d value in micrometers
        end_um: Ending d value in micrometers
        step_um: Step size in micrometers
        use_preprocessed: If True, use preprocessed labels from CSV if available
    
    Returns:
        image_paths: List of image file paths
        labels: Corresponding d values in micrometers
    """
    # Check for preprocessed labels
    preprocessed_csv = 'preprocessed/labels.csv'
    
    if use_preprocessed and os.path.exists(preprocessed_csv):
        print(f"Loading preprocessed labels from: {preprocessed_csv}")
        df = pd.read_csv(preprocessed_csv)
        image_paths = df['filepath'].tolist()
        labels = df['d_value_um'].values
        
        print(f"Loaded {len(image_paths)} images with preprocessed labels")
        print(f"Label range: {labels.min():.4f} - {labels.max():.4f} μm")
        
        return image_paths, labels
    
    # Fallback to original method if no preprocessed data
    print("Preprocessed labels not found. Using automatic labeling...")
    
    # Get all jpg files
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    # Sort by the last 6 digits of the filename (time portion)
    # Extract HHMMSS from IMG_20250922_HHMMSS.jpg format
    def extract_time(filename):
        # For files like IMG_20250922_180555_1.jpg, extract the base time
        base_name = filename.replace('.jpg', '')
        parts = base_name.split('_')
        if len(parts) >= 3:
            return parts[2]  # This is the HHMMSS part
        return ''
    
    image_files.sort(key=extract_time)
    
    # Generate labels based on range
    num_images = len(image_files)
    expected_num = int((end_um - start_um) / step_um) + 1
    
    print(f"Found {num_images} images")
    print(f"Expected ~{expected_num} images based on range {start_um}μm to {end_um}μm with step {step_um}μm")
    
    # Generate labels linearly from start to end
    labels = np.linspace(start_um, end_um, num_images)
    
    # Create full paths
    image_paths = [os.path.join(data_dir, f) for f in image_files]
    
    return image_paths, labels


class SimpleCNN(nn.Module):
    """Simple CNN architecture for regression"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
            
            # Conv Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.regressor(x)
        return x.squeeze()


class ResNetRegressor(nn.Module):
    """ResNet-18 based regression model"""
    
    def __init__(self, pretrained=True):
        super(ResNetRegressor, self).__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.resnet(x).squeeze()


class EfficientNetRegressor(nn.Module):
    """EfficientNet-B0 based regression model"""
    
    def __init__(self, pretrained=True):
        super(EfficientNetRegressor, self).__init__()
        
        # Load EfficientNet
        if pretrained:
            try:
                self.efficientnet = models.efficientnet_b0(pretrained=True)
            except:
                # If pretrained weights not available, use untrained
                self.efficientnet = models.efficientnet_b0(pretrained=False)
        else:
            self.efficientnet = models.efficientnet_b0(pretrained=False)
        
        # Replace the classifier
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.efficientnet(x).squeeze()


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with optional mixed precision"""
    model.train()
    running_loss = 0.0
    predictions = []
    targets = []
    
    # Progress bar for training batches
    pbar = tqdm(train_loader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training if scaler is provided
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        predictions.extend(outputs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_mae = mean_absolute_error(targets, predictions)
    
    return epoch_loss, epoch_mae


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    predictions = []
    targets = []
    
    # Progress bar for validation batches
    pbar = tqdm(val_loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Use mixed precision for inference too if available
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_mae = mean_absolute_error(targets, predictions)
    epoch_rmse = np.sqrt(mean_squared_error(targets, predictions))
    epoch_r2 = r2_score(targets, predictions)
    
    return epoch_loss, epoch_mae, epoch_rmse, epoch_r2, predictions, targets


def train_model(model, model_name, train_loader, val_loader, num_epochs, device, save_dir='models'):
    """
    Train a model and save the best version
    
    Returns:
        history: Dictionary with training history
        best_model_path: Path to the saved best model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                       patience=5, verbose=True)
    
    # Mixed precision training for GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print(f"  Using mixed precision training (FP16) for faster GPU computation")
    
    history = {
        'train_loss': [], 'train_mae': [],
        'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_r2': []
    }
    
    best_val_mae = float('inf')
    best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc=f'{model_name}', unit='epoch')
    
    for epoch in epoch_pbar:
        # Training
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validation
        val_loss, val_mae, val_rmse, val_r2, _, _ = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_mae)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['val_r2'].append(val_r2)
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            }, best_model_path)
        
        # Update progress bar with current metrics
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_mae': f'{val_mae:.4f}',
            'best_mae': f'{best_val_mae:.4f}'
        })
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}]")
            tqdm.write(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            tqdm.write(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
    
    print(f"\nBest Val MAE: {best_val_mae:.4f}")
    print(f"Model saved to: {best_model_path}")
    
    return history, best_model_path


def plot_training_history(histories, model_names, save_path='training_comparison.png'):
    """Plot training history for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = [
        ('train_loss', 'val_loss', 'Loss', axes[0, 0]),
        ('train_mae', 'val_mae', 'MAE (μm)', axes[0, 1]),
        ('val_rmse', None, 'Validation RMSE (μm)', axes[1, 0]),
        ('val_r2', None, 'Validation R²', axes[1, 1])
    ]
    
    for train_key, val_key, ylabel, ax in metrics:
        for history, model_name in zip(histories, model_names):
            epochs = range(1, len(history[train_key]) + 1)
            if 'train' in train_key:
                ax.plot(epochs, history[train_key], label=f'{model_name} (Train)', linestyle='--')
                if val_key:
                    ax.plot(epochs, history[val_key], label=f'{model_name} (Val)')
            else:
                ax.plot(epochs, history[train_key], label=model_name)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs. Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    plt.close()


def evaluate_model(model, model_name, test_loader, device, save_dir='results'):
    """Evaluate model on test set and create visualizations"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    predictions = []
    targets = []
    
    # Progress bar for test set evaluation
    pbar = tqdm(test_loader, desc=f'Evaluating {model_name}', leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            
            # Use mixed precision for faster inference
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    print(f"\n{model_name} - Test Set Results:")
    print(f"  MAE: {mae:.4f} μm")
    print(f"  RMSE: {rmse:.4f} μm")
    print(f"  R² Score: {r2:.4f}")
    
    # Create prediction plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(targets, predictions, alpha=0.6, s=50)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True d value (μm)', fontsize=12)
    axes[0].set_ylabel('Predicted d value (μm)', fontsize=12)
    axes[0].set_title(f'{model_name} - Predictions vs True Values', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add metrics text
    textstr = f'MAE: {mae:.4f} μm\nRMSE: {rmse:.4f} μm\nR²: {r2:.4f}'
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Error distribution
    errors = predictions - targets
    axes[1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].set_xlabel('Prediction Error (μm)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'{model_name} - Error Distribution', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_evaluation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved to: {save_path}")
    plt.close()
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': predictions, 'targets': targets}


def main():
    # Configuration
    DATA_DIR = 'dataset/experiment'
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    IMAGE_SIZE = 224
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    
    # Check device and print detailed GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"DEVICE INFORMATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"Mixed Precision: Enabled (FP16)")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("GPU: Not available - using CPU")
        print("Note: Training will be significantly slower on CPU")
    print(f"{'='*60}")
    
    # Load dataset
    print("\nLoading dataset...")
    image_paths, labels = load_dataset(DATA_DIR)
    
    # Split data: Train / Val / Test
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=TEST_SIZE, random_state=42
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    print(f"  Testing: {len(test_paths)} images")
    print(f"  Label range: {labels.min():.2f} - {labels.max():.2f} μm")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MichelsonDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = MichelsonDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = MichelsonDataset(test_paths, test_labels, transform=val_transform)
    
    # Create dataloaders with GPU optimizations
    # num_workers: Use 4 workers for parallel data loading (adjust based on CPU cores)
    # pin_memory: Faster data transfer to GPU
    # persistent_workers: Keep workers alive between epochs
    num_workers = 4 if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Define models
    models_dict = {
        'SimpleCNN': SimpleCNN(),
        'ResNet18': ResNetRegressor(pretrained=True),
        'EfficientNetB0': EfficientNetRegressor(pretrained=True)
    }
    
    # Train all models
    histories = []
    model_paths = {}
    
    for model_name, model in models_dict.items():
        model = model.to(device)
        history, best_path = train_model(
            model, model_name, train_loader, val_loader, NUM_EPOCHS, device
        )
        histories.append(history)
        model_paths[model_name] = best_path
        
        # Clear GPU cache between models to prevent OOM
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        del model  # Delete model to free memory
    
    # Plot training comparison
    plot_training_history(histories, list(models_dict.keys()))
    
    # Evaluate all models on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_results = {}
    
    for model_name, model in models_dict.items():
        # Load best model
        checkpoint = torch.load(model_paths[model_name])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Evaluate
        results = evaluate_model(model, model_name, test_loader, device)
        test_results[model_name] = results
    
    # Create comparison summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': list(test_results.keys()),
        'MAE (μm)': [test_results[m]['mae'] for m in test_results],
        'RMSE (μm)': [test_results[m]['rmse'] for m in test_results],
        'R² Score': [test_results[m]['r2'] for m in test_results]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison to CSV
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("\nComparison saved to: results/model_comparison.csv")
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['MAE (μm)'].idxmin(), 'Model']
    print(f"\n🏆 Best Model: {best_model}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

