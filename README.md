# ASVspoof 2019 LightCNN Implementation

This repository contains a PyTorch implementation of LightCNN for voice anti-spoofing detection on the ASVspoof 2019 Logical Access (LA) dataset. The implementation follows the specifications from the original LightCNN paper and is designed for the HSE Voice Anti-spoofing homework assignment.

## Task Overview

This project implements a **Countermeasure (CM) system** for voice anti-spoofing detection using the **LightCNN (LCNN) architecture** on the Logical Access partition of the ASVspoof 2019 Dataset. The goal is to distinguish between bonafide (genuine) and spoofed audio samples.

### Key Requirements
- **LightCNN Architecture**: Implemented according to the original paper specifications
- **ASVspoof 2019 LA Dataset**: Logical Access partition
- **STFT Frontend**: Short-Time Fourier Transform for feature extraction
- **Cross-Entropy Loss**: Standard classification loss function
- **Dropout Regularization**: As specified in the training recipe
- **EER Evaluation**: Equal Error Rate as the primary metric

## Architecture

### LightCNN Model
The implementation follows the LightCNN architecture from the original paper with the following key components:

- **MFM Layers**: Max-Feature-Map operations for feature selection
- **Batch Normalization**: For training stability
- **Dropout**: 0.75 dropout probability for regularization
- **STFT Frontend**: Short-Time Fourier Transform with optimized parameters

### Model Specifications
```yaml
model:
  input_shape: [1, 863, 600]  # (channels, height, width)
  num_classes: 2               # bonafide vs spoof
  dropout_prob: 0.75          # dropout for regularization
```

## Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Melodiz/LightCNN_ASVspoof2019.git
cd cd LightCNN_ASVspoof2019/

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Comet.ml Setup
This project uses Comet.ml for experiment tracking and logging. Follow these steps to set up Comet.ml:

#### Step 1: Create Comet.ml Account
1. Go to [Comet.ml](https://www.comet.com) and create a free account
2. Verify your email address

#### Step 2: Get Your API Key
1. Log in to your Comet.ml account
2. Go to your profile settings (click on your avatar in the top right)
3. Navigate to "API Keys" section
4. Copy your API key

#### Step 3: Set Environment Variables
Set your Comet.ml API key as an environment variable:

**On macOS/Linux:**
```bash
export COMET_API_KEY="your_api_key_here"
export COMET_WORKSPACE="your_workspace_name"
export COMET_PROJECT_NAME="asvspoof-baseline"
```

**On Windows (Command Prompt):**
```cmd
set COMET_API_KEY=your_api_key_here
set COMET_WORKSPACE=your_workspace_name
set COMET_PROJECT_NAME=asvspoof-baseline
```

**On Windows (PowerShell):**
```powershell
$env:COMET_API_KEY="your_api_key_here"
$env:COMET_WORKSPACE="your_workspace_name"
$env:COMET_PROJECT_NAME="asvspoof-baseline"
```

**Alternative: Create a .env file**
Create a `.env` file in your project root:
```bash
# .env file
COMET_API_KEY=your_api_key_here
COMET_WORKSPACE=your_workspace_name
COMET_PROJECT_NAME=asvspoof-baseline
```

Then load it in your shell:
```bash
source .env
```

#### Step 4: Verify Setup
Test your Comet.ml setup by running a quick training session:
```bash
python train.py trainer.n_epochs=1  # Quick test with 1 epoch
```

You should see Comet.ml initialization messages in the console output.

### 3. Dataset Setup
This project uses the **ASVspoof 2019 Logical Access (LA) dataset**. The ASVspoof 2019 dataset was created for the third Automatic Speaker Verification Spoofing and Countermeasures Challenge. This repository focuses on the Logical Access (LA) partition, which, for the first time in the challenge's history, includes all three major attack types: text-to-speech (TTS), voice conversion (VC), and replay attacks. The data is derived from the VCTK corpus and is split into training, development, and evaluation sets with no speaker overlap between them. A key feature of the evaluation set is the inclusion of "unknown attacks"—spoofing techniques not present in the training or development data—to rigorously test a model's ability to generalize. 

Download the ASVspoof 2019 LA dataset and organize it as follows:

**Preferred method (direct download):**
```bash
curl -o ./LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip\?sequence\=3\&isAllowed\=y
unzip LA.zip
```

**Alternative sources:**
- **Original Source:** [ASVspoof 2019 dataset page](https://datashare.ed.ac.uk/handle/10283/3336)
- **Kaggle Mirror:** [ASVpoof 2019 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset/data)

**Expected directory structure:**
```
LA/
├── ASVspoof2019_LA_train/
│   ├── flac/
│   └── LICENSE.txt
├── ASVspoof2019_LA_dev/
│   ├── flac/
│   └── LICENSE.txt
├── ASVspoof2019_LA_eval/
│   ├── flac/
│   └── LICENSE.txt
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

### 4. Training
```bash
# Basic training with default settings
python train.py

# Custom training parameters
python train.py trainer.n_epochs=50 optimizer.lr=0.001

# CPU training
python train.py trainer.device=cpu
```

### 5. Evaluation
```bash
# Evaluate trained model
python evaluate.py
```

### 6. Download Pre-trained Checkpoint
The trained model checkpoint (122MB) is not included in this repository to keep it lightweight. You can download it from Comet.ml:

#### Method 1: Download from Comet.ml Web Interface
1. Go to [Comet.ml Experiment](https://www.comet.com/ivan-novosad/asvspoof-baseline/view/new/panels)
2. Navigate to the "Assets" tab
3. Find the `best_model.pth` file in the `checkpoints/` folder
4. Click the download button to save the file
5. Place the downloaded file in the `saved/` directory:
    ```bash
    mkdir -p saved
    mv best_model.pth saved/
    ```

#### Method 2: Download via Comet.ml API
```bash
# Install comet-ml if not already installed
pip install comet-ml

# Download the checkpoint using Python
python -c "
import comet_ml
api = comet_ml.API()
experiment = api.get_experiment('ivan-novosad', 'asvspoof-baseline', '4rhkga45el9k6gs00b0qs36qvscu475t')
experiment.download_asset('checkpoints/best_model.pth', 'saved/best_model.pth')
"
```

#### Method 3: Direct Download Link
If available, you can download directly from the experiment URL:
1. Visit the experiment page
2. Go to Assets → checkpoints
3. Right-click on `best_model.pth` and "Save link as..."
4. Save to `saved/best_model.pth`

**Note**: The checkpoint file is ~122MB. Make sure you have sufficient disk space and a stable internet connection.

## Project Structure

```
template/
├── train.py                    # Modular training script
├── evaluate.py                 # Modular evaluation script
├── requirements.txt            # Dependencies
├── README.md                  # This file
├── src/
│   ├── configs/
│   │   ├── asvspoof_baseline.yaml  # Main configuration
│   │   ├── model/lightcnn.yaml     # Model configuration
│   │   ├── optimizer/adam.yaml      # Optimizer configuration
│   │   ├── scheduler/exponential.yaml # Scheduler configuration
│   │   └── writer/cometml.yaml      # Writer configuration
│   ├── datasets/
│   │   ├── asvspoof_dataset.py     # Dataset implementations
│   │   ├── data_utils.py           # Data utilities
│   │   ├── dataloader_utils.py     # Training data loading
│   │   └── eval_utils.py           # Evaluation data loading
│   ├── model/
│   │   └── lightcnn_original.py    # LightCNN model implementation
│   ├── trainer/
│   │   ├── asvspoof_trainer.py     # Main trainer class
│   │   └── evaluator.py            # Evaluation logic
│   ├── metrics/
│   │   └── eer_utils.py            # EER calculation functions
│   ├── utils/
│   │   ├── init_utils.py           # Initialization utilities
│   │   ├── model_utils.py          # Model loading utilities
│   │   └── results_utils.py        # Results management
│   └── logger/
│       └── cometml.py              # CometML writer
└── LA/                          # Dataset directory (not in repo)
```

## Configuration

### Main Configuration (`src/configs/asvspoof_baseline.yaml`)
```yaml
# Training parameters
epochs: 16
batch_size: 16
learning_rate: 0.0005

# Model parameters
model:
  input_shape: [1, 863, 600]
  num_classes: 2
  dropout_prob: 0.75

# Optimizer
optimizer:
  lr: 0.0005

# Scheduler
scheduler:
  gamma: 0.98
```

### Key Hyperparameters
- **Learning Rate**: 0.0005 (Adam optimizer)
- **Batch Size**: 16
- **Epochs**: 16 (configurable)
- **Dropout**: 0.75
- **Scheduler**: ExponentialLR with gamma=0.98

## Implementation Details

### Data Processing
- **STFT Parameters**: n_fft=1724, hop_length=130, win_length=1724
- **Feature Shape**: [1, 863, 600] (channels, height, width)
- **Augmentation**: SpecAugment with frequency and time masking
- **Class Balancing**: WeightedRandomSampler for balanced training

### Training Features
- **Multi-GPU Support**: Current training/evaluation pipeline is adapted for multi-GPU computations (parallel)
- **Class Balancing**: WeightedRandomSampler for balanced training
- **SpecAugment**: Data augmentation for spectrograms
  - Applying uniform noise to 50% of objects
  - Frequency Masking
  - Time Masking
- **Exponential LR Scheduler**: Learning rate scheduling
- **Comet.ml Integration**: Experiment tracking and logging

## Training and Evaluation

### Training Process
1. **Data Loading**: ASVspoof 2019 LA dataset with class balancing
2. **Feature Extraction**: STFT with optimized parameters
3. **Model Training**: LightCNN with Cross-Entropy loss
4. **Evaluation**: EER calculation on evaluation set
5. **Logging**: Comet.ml integration for experiment tracking

### Evaluation Process
1. **Model Loading**: Load trained checkpoint
2. **Score Generation**: Generate scores for evaluation set
3. **EER Calculation**: Compute Equal Error Rate
4. **Results Saving**: Save predictions to CSV

## Results and Logging

### Comet.ml Integration
The project uses Comet.ml for experiment tracking:
- **Real-time Metrics**: Live training and evaluation metrics
- **Experiment Management**: Organized experiment tracking
- **Checkpoint Logging**: Automatic model checkpoint logging
- **Hyperparameter Tracking**: Configuration parameter logging

**Live Training Logs**: [View on Comet.ml](https://www.comet.com/ivan-novosad/asvspoof-baseline/view/new/panels)

### Training Results
The model achieved excellent performance with **EER of 2.13%** after only 4 epochs of training. This result is close to the authors' reported EER value of 1.86% and significantly exceeds the HSE homework goal of 5.3% EER. The training was stopped early due to satisfactory performance.

#### Training Dashboards
Below are the training metrics and loss curves from the 4-epoch training run:


**Training Loss vs Steps:**

<img src="assets/batch_loss_train%20VS%20step.jpeg" alt="Training Loss" width="830">

**Training Metrics (Epoch Loss and EER):**

<table>
<tr>
<td><img src="assets/epoch_loss_train%20VS%20step.jpeg" alt="Epoch Loss" width="400"></td>
<td><img src="assets/epoch_eer_eval%20VS%20step.jpeg" alt="Evaluation EER" width="400"></td>
</tr>
</table>


### Core Dependencies
```txt
torch==2.8.0
torchaudio==2.8.0
torchvision==0.23.0
soundfile==0.13.1
hydra-core==1.3.2
omegaconf==2.3.0
comet_ml==3.50.0
numpy==1.26.4
pandas==2.3.1
tqdm==4.67.1
```

## References and Citations

### Academic Papers
1. **ASVspoof 2019**: [Evaluation Plan](https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf)
2. **LightCNN Paper**: [Speech Technology Center](https://arxiv.org/abs/1904.05576)

### Code and Templates
3. **PyTorch Project Template**: [GitHub Repository](https://github.com/Blinorot/pytorch_project_template) by Petr Grinberg

### Citation
If you use this implementation, please cite:

```bibtex
@software{asvspoof_lightcnn_2024,
  title={ASVspoof 2019 LightCNN Implementation},
  author={Novosad, Ivan},
  year={2025},
  url={https://github.com/Melodiz/LightCNN_ASVspoof2019},
  note={Voice anti-spoofing detection using LightCNN on ASVspoof 2019 LA dataset}
}
```
