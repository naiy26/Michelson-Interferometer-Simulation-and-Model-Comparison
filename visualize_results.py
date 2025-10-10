"""
Visualization script for Michelson Interferometer results
Run this after training to see comprehensive visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from tqdm import tqdm
from predict import SimpleCNN, ResNetRegressor, EfficientNetRegressor, load_model, predict_image

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def visualize_dataset(data_dir='dataset/experiment', num_samples=10):
    """Visualize sample images from the dataset"""
    
    # Load dataset
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    # Sort by timestamp
    def extract_time(filename):
        base_name = filename.replace('.jpg', '')
        parts = base_name.split('_')
        if len(parts) >= 3:
            return parts[2]
        return ''
    
    image_files.sort(key=extract_time)
    
    # Generate labels
    start_um, end_um = 1.0, 5.0
    labels = np.linspace(start_um, end_um, len(image_files))
    
    print(f"Dataset Information:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Label range: {labels.min():.2f} - {labels.max():.2f} μm")
    print(f"  Label step (avg): {np.mean(np.diff(labels)):.4f} μm\n")
    
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    axes = axes.flatten()
    
    indices = np.linspace(0, len(image_files)-1, num_samples, dtype=int)
    
    for idx, ax in zip(indices, axes):
        img_path = os.path.join(data_dir, image_files[idx])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f'd = {labels[idx]:.2f} μm', fontsize=12)
        ax.axis('off')
    
    plt.suptitle('Sample Michelson Interferometer Images', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('results/dataset_samples.png', dpi=300, bbox_inches='tight')
    print("Dataset samples saved to: results/dataset_samples.png\n")
    plt.show()


def visualize_label_distribution(data_dir='dataset/experiment'):
    """Visualize label distribution"""
    
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    def extract_time(filename):
        base_name = filename.replace('.jpg', '')
        parts = base_name.split('_')
        if len(parts) >= 3:
            return parts[2]
        return ''
    
    image_files.sort(key=extract_time)
    labels = np.linspace(1.0, 5.0, len(image_files))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(labels, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('d value (μm)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of d values', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Line plot
    axes[1].plot(range(len(labels)), labels, linewidth=2, color='coral')
    axes[1].set_xlabel('Image Index', fontsize=12)
    axes[1].set_ylabel('d value (μm)', fontsize=12)
    axes[1].set_title('d value vs. Image Index', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/label_distribution.png', dpi=300, bbox_inches='tight')
    print("Label distribution saved to: results/label_distribution.png\n")
    plt.show()


def visualize_model_comparison():
    """Visualize model comparison results"""
    
    if not os.path.exists('results/model_comparison.csv'):
        print("Model comparison results not found.")
        print("Please run training first: python train_michelson.py\n")
        return
    
    comparison_df = pd.read_csv('results/model_comparison.csv')
    print("Model Performance Comparison:\n")
    print(comparison_df.to_string(index=False))
    print("\n" + "="*60 + "\n")
    
    # Find best model
    best_model = comparison_df.loc[comparison_df['MAE (μm)'].idxmin(), 'Model']
    best_mae = comparison_df.loc[comparison_df['MAE (μm)'].idxmin(), 'MAE (μm)']
    print(f"🏆 Best Model: {best_model} with MAE = {best_mae:.4f} μm\n")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['MAE (μm)', 'RMSE (μm)', 'R² Score']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                     color=color, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} Comparison', fontsize=14)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_bars.png', dpi=300, bbox_inches='tight')
    print("Model comparison plot saved to: results/model_comparison_bars.png\n")
    plt.show()


def test_predictions(model_path='models/SimpleCNN_best.pth', 
                    model_type='SimpleCNN',
                    data_dir='dataset/experiment',
                    num_samples=20):
    """Test model predictions on sample images"""
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please run training first: python train_michelson.py\n")
        return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, model_type, device)
    print()
    
    # Load dataset
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    def extract_time(filename):
        base_name = filename.replace('.jpg', '')
        parts = base_name.split('_')
        if len(parts) >= 3:
            return parts[2]
        return ''
    
    image_files.sort(key=extract_time)
    labels = np.linspace(1.0, 5.0, len(image_files))
    
    # Make predictions on evenly spaced samples
    sample_indices = np.linspace(0, len(image_files)-1, num_samples, dtype=int)
    sample_true = []
    sample_pred = []
    
    print(f"Making predictions on {num_samples} sample images...")
    
    # Progress bar for predictions
    for idx in tqdm(sample_indices, desc='Predicting'):
        img_path = os.path.join(data_dir, image_files[idx])
        pred = predict_image(model, img_path, device)
        sample_true.append(labels[idx])
        sample_pred.append(pred)
    
    print()
    
    sample_true = np.array(sample_true)
    sample_pred = np.array(sample_pred)
    errors = sample_pred - sample_true
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(sample_true, sample_pred, alpha=0.7, s=100, color='royalblue', edgecolors='black')
    axes[0].plot([sample_true.min(), sample_true.max()], 
                 [sample_true.min(), sample_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('True d value (μm)', fontsize=12)
    axes[0].set_ylabel('Predicted d value (μm)', fontsize=12)
    axes[0].set_title(f'{model_type} - Predictions vs True Values', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Add metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    r2 = 1 - (np.sum(errors**2) / np.sum((sample_true - sample_true.mean())**2))
    
    textstr = f'MAE: {mae:.4f} μm\nRMSE: {rmse:.4f} μm\nR²: {r2:.4f}'
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, 
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Error plot
    axes[1].plot(sample_true, errors, 'o-', alpha=0.7, color='coral', 
                markersize=8, markeredgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[1].fill_between(sample_true, errors, alpha=0.3, color='coral')
    axes[1].set_xlabel('True d value (μm)', fontsize=12)
    axes[1].set_ylabel('Prediction Error (μm)', fontsize=12)
    axes[1].set_title(f'{model_type} - Error vs True Value', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_type}_sample_predictions.png', dpi=300, bbox_inches='tight')
    print(f"Prediction plot saved to: results/{model_type}_sample_predictions.png\n")
    plt.show()
    
    print(f"Sample Statistics:")
    print(f"  MAE: {mae:.4f} μm")
    print(f"  RMSE: {rmse:.4f} μm")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Max Error: {np.max(np.abs(errors)):.4f} μm\n")


def main():
    """Run all visualizations"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print("Michelson Interferometer Results Visualization")
    print("="*60)
    print()
    
    # 1. Dataset visualization
    print("1. Visualizing Dataset Samples...")
    print("-"*60)
    visualize_dataset()
    
    # 2. Label distribution
    print("2. Visualizing Label Distribution...")
    print("-"*60)
    visualize_label_distribution()
    
    # 3. Model comparison
    print("3. Visualizing Model Comparison...")
    print("-"*60)
    visualize_model_comparison()
    
    # 4. Test predictions for each model
    print("4. Testing Model Predictions...")
    print("-"*60)
    
    models_to_test = [
        ('models/SimpleCNN_best.pth', 'SimpleCNN'),
        ('models/ResNet18_best.pth', 'ResNet18'),
        ('models/EfficientNetB0_best.pth', 'EfficientNetB0')
    ]
    
    for model_path, model_type in models_to_test:
        if os.path.exists(model_path):
            print(f"\nTesting {model_type}...")
            print("-"*40)
            test_predictions(model_path, model_type, num_samples=30)
        else:
            print(f"\n{model_type} not found. Skipping...\n")
    
    print("="*60)
    print("Visualization Complete!")
    print("All plots saved in the 'results/' directory")
    print("="*60)


if __name__ == '__main__':
    main()

