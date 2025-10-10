# Michelson Interferometer d-value Prediction

This project uses deep learning to predict the d-value (path difference) from Michelson interferometer images. Multiple neural network architectures are trained and compared to find the best model for this regression task.

## Dataset

The dataset consists of 400 Michelson interferometer images with corresponding d-values ranging from 1 μm to 5 μm with a step size of 0.01 μm. Images are sorted by the last six digits of their filenames (timestamp) and labels are assigned linearly across this range.

### Creating Labels

Before training, generate the label file:
```bash
python create_labels.py
```

This will create `preprocessed/labels.csv` which maps each image to its d-value.

### Dataset Structure
```
dataset/
└── experiment/
    ├── IMG_20250922_173804.jpg
    ├── IMG_20250922_174041.jpg
    ├── ...
    └── README.txt
```

## Models

Three different neural network architectures are implemented and compared:

1. **SimpleCNN**: Custom CNN with 4 convolutional blocks and fully connected layers
2. **ResNet18**: Transfer learning using ResNet-18 pretrained on ImageNet
3. **EfficientNetB0**: Transfer learning using EfficientNet-B0 pretrained on ImageNet

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Generate label mapping
python create_labels.py

# Train models
python train_michelson.py

# Visualize results
python visualize_results.py
```

## Usage

### Step 1: Generate Labels

First, create the label mapping file:

```bash
python create_labels.py
```

This creates `preprocessed/labels.csv` with the mapping of images to d-values.

### Step 2: Train Models

Train all three models and compare their performance:

```bash
python train_michelson.py
```

The script will:
- Load and preprocess the dataset
- Split data into train (70%), validation (15%), and test (15%) sets
- Train all three models for 50 epochs each
- Save the best model for each architecture
- Generate training history plots
- Evaluate all models on the test set
- Create comparison visualizations and metrics

### Training Configuration

You can modify these parameters in `train_michelson.py`:

```python
BATCH_SIZE = 16          # Batch size for training
NUM_EPOCHS = 50          # Number of training epochs
IMAGE_SIZE = 224         # Input image size
TEST_SIZE = 0.15         # Test set proportion
VAL_SIZE = 0.15          # Validation set proportion
```

### Step 3: Make Predictions

Use a trained model to predict d-values for new images:

#### Single Image Prediction
```bash
python predict.py --model_path models/SimpleCNN_best.pth --model_type SimpleCNN --image path/to/image.jpg
```

#### Batch Prediction
```bash
python predict.py --model_path models/ResNet18_best.pth --model_type ResNet18 --image_dir dataset/experiment --output predictions.csv
```

#### Arguments:
- `--model_path`: Path to the trained model checkpoint (required)
- `--model_type`: Model architecture (`SimpleCNN`, `ResNet18`, or `EfficientNetB0`)
- `--image`: Path to a single image for prediction
- `--image_dir`: Directory containing images for batch prediction
- `--output`: Output CSV file to save batch predictions

## Output Files

After training, the following files will be generated:

### Models
```
models/
├── SimpleCNN_best.pth
├── ResNet18_best.pth
└── EfficientNetB0_best.pth
```

### Results
```
results/
├── SimpleCNN_evaluation.png      # Prediction vs True plot and error distribution
├── ResNet18_evaluation.png
├── EfficientNetB0_evaluation.png
└── model_comparison.csv           # Comparison metrics (MAE, RMSE, R²)
```

### Training History
```
training_comparison.png            # Training curves for all models
```

## Model Performance

After training, the script will display a comparison table:

```
Model Comparison Summary
------------------------
         Model    MAE (μm)   RMSE (μm)   R² Score
    SimpleCNN       0.XXXX      0.XXXX      0.XXXX
     ResNet18       0.XXXX      0.XXXX      0.XXXX
EfficientNetB0      0.XXXX      0.XXXX      0.XXXX
```

### Metrics Explained:
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and true values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences
- **R² Score**: Coefficient of determination (1.0 = perfect predictions)

## Data Augmentation

Training data is augmented with:
- Random horizontal and vertical flips
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)
- Standard ImageNet normalization

## Project Structure

```
.
├── dataset/
│   └── experiment/         # Training images (400 .jpg files)
├── preprocessed/           # Generated label files
│   └── labels.csv          # Image to d-value mapping
├── models/                 # Saved model checkpoints (generated)
│   ├── SimpleCNN_best.pth
│   ├── ResNet18_best.pth
│   └── EfficientNetB0_best.pth
├── results/                # Evaluation results (generated)
├── create_labels.py        # Generate label mapping
├── train_michelson.py      # Main training script
├── predict.py              # Prediction script
├── visualize_results.py    # Visualization script
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── PROJECT_SUMMARY.md     # Project overview
```

## Technical Details

### SimpleCNN Architecture
- 4 convolutional blocks (32→64→128→256 filters)
- Batch normalization and dropout for regularization
- Adaptive average pooling
- 3-layer fully connected head

### ResNet18 
- Pretrained ResNet-18 backbone
- Custom regression head (512→256→64→1)
- Dropout for regularization

### EfficientNetB0
- Pretrained EfficientNet-B0 backbone
- Custom regression head (1280→256→64→1)
- Compound scaling for efficiency

### Training Strategy
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with initial learning rate 0.001
- Learning rate scheduling: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping based on validation MAE

## GPU Acceleration

The training script automatically detects and uses CUDA if available:
- **With GPU**: Training takes approximately 10-15 minutes
- **Without GPU**: Training takes approximately 1-2 hours

## Tips for Best Results

1. **Dataset Quality**: Ensure images are properly sorted and labels are accurate
2. **Hyperparameter Tuning**: Adjust learning rate, batch size, or epochs as needed
3. **Data Augmentation**: Modify augmentation strategies based on your specific data
4. **Model Selection**: Use the model comparison to select the best architecture
5. **GPU Usage**: Use GPU acceleration for faster training

## Troubleshooting

### Out of Memory (OOM) Error
- Reduce `BATCH_SIZE` in `train_michelson.py`
- Use a smaller image size (e.g., 128 instead of 224)

### Poor Model Performance
- Increase `NUM_EPOCHS` for longer training
- Check if labels are correctly assigned
- Verify image quality and dataset consistency

### Slow Training
- Enable GPU acceleration if available
- Reduce image size or model complexity
- Use fewer data augmentation operations

## Citation

If you use this code in your research, please cite:

```
Michelson Interferometer d-value Prediction using Deep Learning
[Your Name/Institution]
2025
```

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please open an issue in the repository or contact [your email].

