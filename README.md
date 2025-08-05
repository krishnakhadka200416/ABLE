# Adversarially Bracketed Local Explanation (ABLE)

## Overview of ABLE

Modern machine learning models—especially deep neural networks—are frequently used in critical applications but are often viewed as "black boxes" due to their limited interpretability. Local explanation methods like LIME approximate a model's behavior near a test instance with a simple surrogate model. However, such approaches can suffer from instability and poor local fidelity.

**Adversarially Bracketed Local Explanation (ABLE)** is a novel approach designed to overcome these limitations by using adversarial attacks to bracket the local decision boundary. By generating adversarial pairs that enclose the decision boundary near a test instance, ABLE trains a linear surrogate model that yields stable and high-fidelity local explanations.

## Approach

ABLE comprises three main steps:

### 1. Neighborhood Generation
A set of neighborhood points is created around a test instance $x_{\text{test}}$ by adding bounded Gaussian noise. This step captures the local variations in the input space.

### 2. Adversarial Pair Generation
For each generated neighborhood point $D$:
- An adversarial attack finds a minimally perturbed example $A$ that flips the label.
- A second adversarial attack then produces $A'$, which flips back to the original label of $D$.
- The pair $(A, A')$ forms an adversarial bracket that closely encloses the relevant decision boundary for $x_{\text{test}}$.

### 3. Surrogate Model Training
A simple linear model (e.g., logistic regression) is trained on the adversarial pairs. The coefficients of this surrogate model are then used to identify the top‑k important features that influence the prediction of $x_{\text{test}}$.

## Key Features

- **Multiple Attack Methods**: Supports PGD, FGSM, etDeepFool, and HopSkipJump attacks
- **Multi-Class Support**: Works with both binary and multi-class classification problems
- **Multiple Model Architectures**: Compatible with MLP, TabNet, and TabTransformer models
- **Comprehensive Datasets**: Supports 6 benchmark datasets including both binary and multi-class
- **High Fidelity**: Generates explanations with superior local fidelity compared to baseline methods
- **Stability**: Produces more stable explanations under input perturbations
- **Modular Architecture**: Clean separation of datasets, models, and core ABLE functionality

## Architecture & File Structure

ABLE features a modular design for better maintainability and extensibility:

```
ABLE/
├── able.py          # Main ABLE implementation and CLI
├── datasets.py      # Dataset loading and configuration
├── models.py        # Model definitions and training
├── example.py       # Example usage
├── README.md        # This documentation
└── requirements.txt # Dependencies
```

### Core Modules

- **`able.py`**: Main ABLE implementation containing adversarial generation and explanation logic
- **`datasets.py`**: Handles dataset loading, preprocessing, and configuration for all supported datasets
- **`models.py`**: Contains model definitions, training procedures, and ART wrapper classes

## Supported Components

### Attack Methods
- **PGD (Projected Gradient Descent)**: Iterative gradient-based attack with projection
- **FGSM (Fast Gradient Sign Method)**: Single-step gradient-based attack
- **etDeepFool**: Enhanced Targeted DeepFool specifically designed for tabular data
- **HopSkipJump**: Black-box boundary-based attack method

### Model Architectures
- **MLP**: Multi-Layer Perceptron with batch normalization and dropout
- **TabNet**: Attention-based tabular neural network
- **TabTransformer**: Transformer architecture for tabular data

### Datasets
- **Binary Classification**: credit, adult, breast_cancer, mushroom
- **Multi-Class Classification**: car (4 classes), covertype (7 classes)

## Installation

1. Clone or download the ABLE module
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Generate an explanation for a test instance using the default PGD attack:

```bash
python able.py --model MLP --dataset adult --test-index 20 --top-k 5
```

### Advanced Usage

Use different attack methods and parameters:

```bash
# Using FGSM attack
python able.py --model TabNet --dataset car --test-index 5 --attack FGSM --top-k 3

# Using etDeepFool with custom parameters
python able.py --model TabTransformer --dataset covertype --test-index 10 --attack ETDEEPFOOL --adversarial-pairs 30 --radius 0.8

# Using HopSkipJump attack
python able.py --model MLP --dataset mushroom --test-index 2 --attack HOPSKIPJUMP --neighbors 150
```

### Command Line Arguments

| Argument | Type | Required | Choices | Default | Description |
|----------|------|----------|---------|---------|-------------|
| `--model` | str | Yes | MLP, TabNet, TabTransformer | - | Black-box model to use |
| `--dataset` | str | Yes | credit, adult, breast_cancer, mushroom, car, covertype | - | Dataset to use |
| `--test-index` | int | Yes | - | - | Index of test instance to explain |
| `--attack` | str | No | PGD, FGSM, ETDEEPFOOL, HOPSKIPJUMP | PGD | Adversarial attack method |
| `--top-k` | int | No | - | 5 | Number of top features to return |
| `--adversarial-pairs` | int | No | - | 50 | Number of adversarial pairs to generate |
| `--radius` | float | No | - | 0.5 | Neighborhood radius for local sampling |
| `--neighbors` | int | No | - | 100 | Number of neighbors to generate |
| `--device` | str | No | auto, cuda, cpu | auto | Computing device to use |

## Examples

### Example 1: Basic ABLE Explanation
```bash
python able.py --model MLP --dataset adult --test-index 20 --top-k 5
```

**Output:**
```
Dataset adult: 2 classes, 2 unique labels
Training samples: 26048, Test samples: 6513
Input dimensions: 108

Dataset: adult
Model: MLP
Attack Method: ABLE_PGD
Test Instance: 0
Top-5 Features:
  1. capital-gain: 0.0000
  2. marital-status_Married-civ-spouse: Married-civ-spouse
  3. education-num: 13.0000
  4. age: 25.0000
  5. hours-per-week: 40.0000
```

### Example 2: Multi-Class Explanation
```bash
python able.py --model TabNet --dataset car --test-index 5 --attack FGSM --top-k 3
```

**Output:**
```
Dataset car: 4 classes, 4 unique labels
Training samples: 1382, Test samples: 346
Input dimensions: 15

Dataset: car
Model: TabNet
Attack Method: ABLE_FGSM
Test Instance: 5
Top-3 Features:
  1. safety: high
  2. buying: low
  3. persons: 4
```

### Example 3: Different Attack Methods
```bash
# Compare different attack methods on the same instance
python able.py --model MLP --dataset adult --test-index 20 --attack PGD
python able.py --model MLP --dataset adult --test-index 20 --attack FGSM
python able.py --model MLP --dataset adult --test-index 20 --attack ETDEEPFOOL
python able.py --model MLP --dataset adult --test-index 20 --attack HOPSKIPJUMP
```

## Implementation Details

### Modular Architecture
- **Dataset Module**: Centralizes dataset configuration and preprocessing logic
- **Model Module**: Contains all model architectures and training procedures
- **Core ABLE**: Focuses purely on adversarial generation and explanation logic
- **Clean Interfaces**: Well-defined APIs between modules for easy extension

### Data Processing
- **Feature Transformation**: Uses RDT (Reversible Data Transforms) for automatic feature preprocessing
- **Standardization**: Features are standardized using sklearn's StandardScaler
- **Reverse Transformation**: Applies inverse transformations to display original meaningful feature values
- **Type Conversion**: All data is converted to float32 for optimal performance

### Model Training
- **Automatic Training**: Models are trained automatically when running explanations
- **Optimized Hyperparameters**: Each model uses optimized hyperparameters for good performance
- **Multi-Class Support**: All models support both binary and multi-class classification

### Adversarial Generation
- **Dual Adversarial Pairs**: Generates pairs that bracket the decision boundary
- **Robust Implementation**: Handles failed attacks gracefully with adaptive epsilon adjustment
- **FGSM Retry Mechanism**: Automatically increases epsilon (0.6→2.5) when FGSM attacks fail
- **Multiple Algorithms**: Supports gradient-based and geometric attack methods

### Explanation Quality
- **Feature Ranking**: Ranks features by absolute coefficient magnitude from surrogate model
- **Multi-Class Handling**: Properly handles multi-class logistic regression coefficients


## Technical Requirements

- **Python**: 3.7 or higher
- **PyTorch**: 1.9.0 or higher
- **CUDA**: Optional, for GPU acceleration
- **Memory**: At least 4GB RAM (8GB recommended for larger datasets)
- **Storage**: Minimal storage requirements (datasets downloaded automatically)

## Performance Characteristics

### Typical Performance
- **Binary Classification**: High-quality explanations with stable feature rankings
- **Multi-Class Classification**: Effective explanations across multiple classes


### Scalability
- **Instance Size**: Handles up to 1000+ features efficiently
- **Dataset Size**: Tested on datasets with 50K+ samples
- **Memory Usage**: Optimized for memory efficiency

## Troubleshooting


### Performance Tips

1. **Use GPU**: Enable CUDA for faster model training and adversarial generation
2. **Adjust Parameters**: Tune adversarial pairs and neighborhood size for quality/speed trade-off
3. **Choose Attack Method**: PGD or eTDeepFool generally provides best results, FGSM is fastest

## Extending ABLE

### Adding New Datasets
1. Add dataset configuration to `datasets.py` in `DATASET_CONFIG`
2. Implement preprocessing function if needed

### Adding New Models
1. Define model class in `models.py`
2. Add training logic to `create_and_train_model()` function
3. Create ART wrapper if needed

### Adding New Attack Methods
1. Add attack creation logic to `create_adversarial_attack()` in `able.py`
2. Implement dual adversarial generation logic in `generate_dual_adversary()`



