"""
Prediction script for Michelson Interferometer d-value
Load a trained model and predict d value from new images
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


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
    
    def __init__(self, pretrained=False):
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
    
    def __init__(self, pretrained=False):
        super(EfficientNetRegressor, self).__init__()
        
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        
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


def load_model(model_path, model_type='SimpleCNN', device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of model ('SimpleCNN', 'ResNet18', 'EfficientNetB0')
        device: Device to load model on
    
    Returns:
        Loaded model in evaluation mode
    """
    # Initialize model
    if model_type == 'SimpleCNN':
        model = SimpleCNN()
    elif model_type == 'ResNet18':
        model = ResNetRegressor(pretrained=False)
    elif model_type == 'EfficientNetB0':
        model = EfficientNetRegressor(pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} model from {model_path}")
    print(f"  Validation MAE: {checkpoint.get('val_mae', 'N/A')}")
    print(f"  Validation RMSE: {checkpoint.get('val_rmse', 'N/A')}")
    print(f"  Validation R²: {checkpoint.get('val_r2', 'N/A')}")
    
    return model


def predict_image(model, image_path, device='cpu', image_size=224, true_value=None):
    """
    Predict d value for a single image
    
    Args:
        model: Trained model
        image_path: Path to image
        device: Device to run prediction on
        image_size: Size to resize image to
        true_value: True d value for comparison (optional)
    
    Returns:
        Predicted d value in micrometers
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Move to device (use non_blocking for GPU)
    if isinstance(device, torch.device) and device.type == 'cuda':
        image_tensor = image_tensor.to(device, non_blocking=True)
    else:
        image_tensor = image_tensor.to(device)
    
    # Predict with mixed precision on GPU
    with torch.no_grad():
        if isinstance(device, torch.device) and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                prediction = model(image_tensor)
        else:
            prediction = model(image_tensor)
    
    predicted_value = prediction.item()
    
    # If true value is provided, calculate and display error
    if true_value is not None:
        absolute_error = abs(predicted_value - true_value)
        print(f"  Real d: {true_value:.4f} μm")
        print(f"  Predicted d: {predicted_value:.4f} μm")
        print(f"  绝对误差 (Absolute Error): {absolute_error:.4f} μm")
    
    return predicted_value


def predict_batch(model, image_dir, device='cpu', image_size=224, labels_csv=None):
    """
    Predict d values for all images in a directory
    
    Args:
        model: Trained model
        image_dir: Directory containing images
        device: Device to run prediction on
        image_size: Size to resize images to
        labels_csv: Path to CSV file with true labels (optional)
    
    Returns:
        Dictionary mapping image filename to predicted d value
    """
    # Get all jpg files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files.sort()
    
    predictions = {}
    errors = []
    
    # Load true labels if provided
    true_labels = {}
    if labels_csv and os.path.exists(labels_csv):
        df = pd.read_csv(labels_csv)
        true_labels = dict(zip(df['filename'], df['d_value_um']))
        print(f"Loaded true labels from {labels_csv}")
    
    print(f"\nPredicting d values for {len(image_files)} images...")
    
    # Progress bar for batch prediction
    for img_file in tqdm(image_files, desc='Predicting'):
        img_path = os.path.join(image_dir, img_file)
        
        # Get true value if available
        true_value = true_labels.get(img_file) if true_labels else None
        
        pred = predict_image(model, img_path, device, image_size, true_value)
        predictions[img_file] = pred
        
        # Calculate error if true value available
        if true_value is not None:
            error = abs(pred - true_value)
            errors.append(error)
    
    # Print summary
    print("\nPredictions complete!")
    
    if errors:
        print(f"\nError Statistics:")
        print(f"  Mean Absolute Error: {np.mean(errors):.4f} μm")
        print(f"  Max Absolute Error: {np.max(errors):.4f} μm")
        print(f"  Min Absolute Error: {np.min(errors):.4f} μm")
    
    print(f"\nSample results:")
    for i, (img_file, pred) in enumerate(list(predictions.items())[:5]):
        true_val = true_labels.get(img_file, "N/A")
        if true_val != "N/A":
            error = abs(pred - true_val)
            print(f"  {img_file}: Pred={pred:.4f} μm, True={true_val:.4f} μm, Error={error:.4f} μm")
        else:
            print(f"  {img_file}: {pred:.4f} μm")
    if len(predictions) > 5:
        print(f"  ... and {len(predictions) - 5} more images")
    
    return predictions, true_labels


def main():
    parser = argparse.ArgumentParser(description='Predict d values from Michelson interferometer images')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='SimpleCNN',
                        choices=['SimpleCNN', 'ResNet18', 'EfficientNetB0'],
                        help='Type of model architecture')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for prediction')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory containing images for batch prediction')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file to save predictions (CSV format)')
    parser.add_argument('--labels_csv', type=str, default='preprocessed/labels.csv',
                        help='Path to CSV file with true labels for comparison')
    
    args = parser.parse_args()
    
    # Check device and print GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using mixed precision (FP16) for faster inference")
    print()
    
    # Load model
    model = load_model(args.model_path, args.model_type, device)
    
    # Predict
    if args.image:
        # Single image prediction
        prediction = predict_image(model, args.image, device)
        print(f"\nPrediction for {args.image}:")
        print(f"  d = {prediction:.4f} μm")
        
    elif args.image_dir:
        # Batch prediction
        predictions, true_labels = predict_batch(model, args.image_dir, device, labels_csv=args.labels_csv)
        
        # Save predictions if output specified
        if args.output:
            # Create detailed DataFrame with all information
            data = []
            for img_file, pred in predictions.items():
                true_val = true_labels.get(img_file, None)
                if true_val is not None:
                    error = abs(pred - true_val)
                    data.append({
                        'Image': img_file,
                        'Real_d_value_um': true_val,
                        'Predicted_d_value_um': pred,
                        'Absolute_Error_um': error
                    })
                else:
                    data.append({
                        'Image': img_file,
                        'Real_d_value_um': 'N/A',
                        'Predicted_d_value_um': pred,
                        'Absolute_Error_um': 'N/A'
                    })
            
            df = pd.DataFrame(data)
            
            # Add error statistics as a summary at the end
            if true_labels:
                errors = [row['Absolute_Error_um'] for row in data if row['Absolute_Error_um'] != 'N/A']
                if errors:
                    summary_row = {
                        'Image': '=== SUMMARY STATISTICS ===',
                        'Real_d_value_um': f'Mean Absolute Error: {np.mean(errors):.4f}',
                        'Predicted_d_value_um': f'Max Absolute Error: {np.max(errors):.4f}',
                        'Absolute_Error_um': f'Min Absolute Error: {np.min(errors):.4f}'
                    }
                    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            
            df.to_csv(args.output, index=False)
            print(f"\nDetailed predictions saved to: {args.output}")
            
            # Show CSV preview
            print(f"\nCSV Preview (first 5 rows):")
            print(df.head().to_string(index=False))
            
            if true_labels:
                print(f"\nCSV includes:")
                print(f"  - Image filename")
                print(f"  - Real d-value (μm)")
                print(f"  - Predicted d-value (μm)")
                print(f"  - Absolute Error (μm)")
                print(f"  - Summary statistics at the end")
    else:
        print("Please specify either --image or --image_dir for prediction")


if __name__ == '__main__':
    main()

