"""
Simple script to create labels for exactly 400 images with 0.01 um steps
"""

import os
import pandas as pd
import numpy as np

# Check dataset
data_dir = 'dataset/experiment'
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
print(f"Found {len(image_files)} images")

# Sort by timestamp (last 6 digits)
def extract_timestamp(filename):
    base_name = filename.replace('.jpg', '')
    parts = base_name.split('_')
    if len(parts) >= 3:
        return parts[2]
    return ''

image_files.sort(key=extract_timestamp)
print(f"Sorted {len(image_files)} images by timestamp")

# Create labels with 0.01 um steps
start_um = 1.0
step_um = 0.01
num_images = len(image_files)

# Generate labels: 1.00, 1.01, 1.02, ..., 4.99
# Use integer arithmetic to avoid floating-point precision errors
labels = np.array([start_um + i * step_um for i in range(num_images)])
labels = np.round(labels, 2)  # Round to 2 decimal places

print(f"Generated {len(labels)} labels")
print(f"Range: {labels.min():.2f} - {labels.max():.2f} um")
print(f"Step size: {step_um} um")

# Create DataFrame
data = {
    'filename': image_files,
    'filepath': [os.path.join(data_dir, f) for f in image_files],
    'd_value_um': labels,
    'index': range(num_images),
    'timestamp': [extract_timestamp(f) for f in image_files]
}

df = pd.DataFrame(data)

# Create preprocessed directory
os.makedirs('preprocessed', exist_ok=True)

# Save labels
csv_path = 'preprocessed/labels.csv'
df.to_csv(csv_path, index=False)
print(f"Labels saved to: {csv_path}")

# Show sample
print("\nFirst 10 entries:")
print(df.head(10)[['filename', 'd_value_um']].to_string(index=False))

print("\nLast 10 entries:")
print(df.tail(10)[['filename', 'd_value_um']].to_string(index=False))

# Verify step sizes
step_sizes = np.diff(df['d_value_um'])
print(f"\nStep size verification:")
print(f"  Mean step: {step_sizes.mean():.6f} um")
print(f"  All steps are 0.01 um: {np.allclose(step_sizes, 0.01)}")
print(f"  Step sizes: {step_sizes[:10]}")  # Show first 10 step sizes

print(f"\n✓ Dataset ready! {num_images} images with 0.01 um steps")
print("✓ Training script will automatically use these labels")
