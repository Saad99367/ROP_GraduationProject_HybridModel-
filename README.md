# ROP-HybridVision 👁️🏥

![Project Status](https://img.shields.io/badge/Status-Completed-success) 
![Grade](https://img.shields.io/badge/Grade-A%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

## 🎓 Graduation Project Context
This repository contains the source code for my Computer Science graduation project, which received an **A+ grade**. The project explores an advanced hybrid deep learning architecture for classifying Retinopathy of Prematurity (ROP) from retinal images. 

**Note:** This project is intended for research and educational purposes. It is not approved for clinical diagnosis.

---

## 📖 Project Overview and Objectives
Retinopathy of Prematurity (ROP) is an eye disease that can happen in premature babies. Early and accurate staging is critical to prevent blindness. The objective of this project is to build an automated, deep-learning-based classification pipeline capable of staging ROP into five categories (Normal, Stage 1, Stage 2, Stage 3, and Stage 4).

The core contribution is a **Hybrid Vision Architecture** that combines the strong local feature extraction capabilities of Convolutional Neural Networks (CNNs) with the global context modeling of Transformers, capped by a Learnable Gaussian Fuzzy Classification Layer to handle uncertainty in medical grading.

---

## ✨ Key Features
- **Hybrid Architecture:** Integration of a CNN backbone, a Transformer encoder, and Fuzzy Logic.
- **Learnable Fuzzy Logic:** Models classification uncertainty explicitly for the 5 ordinal stages.
- **Modular Codebase:** Cleanly structured PyTorch code with separate components for data, modeling, and training.
- **Gradio Interface:** A web-based UI for testing the model via image uploads.
- **CLI Scripts:** Streamlined training and inference scripts.

---

## 🧠 Model Architecture

The ROP classification is performed using a custom hybrid pipeline composed of the following components:

1. **EfficientNet-B3 Pretrained Backbone:**
   Acts as the primary visual feature extractor. It leverages ImageNet-pretrained weights to extract rich spatial feature maps from the retinal images efficiently.
2. **Transformer Encoder:**
   Takes the flattened spatial features from the backbone and models global relationships and contextual dependencies between different regions of the retina.
3. **Classification / Projection Head:**
   A fully-connected sub-network that projects the high-dimensional transformer embeddings into an intermediate feature space.
4. **Learnable Gaussian Fuzzy Classification Layer:**
   Replaces the traditional Softmax layer. It uses learnable Gaussian membership functions to predict the probability of the image belonging to the 5 ROP stages, naturally accommodating the overlapping characteristics of progressive medical conditions.

---

## 📊 Dataset Overview

The dataset used in this project consists of **3,866 retinal images** categorized into five stages. 

| Class | ROP Stage | Number of Images |
|:---:|:---|---:|
| 0 | No ROP (Normal) | 875 |
| 1 | Mild ROP (Stage 1) | 713 |
| 2 | Moderate ROP (Stage 2) | 810 |
| 3 | Severe ROP (Stage 3) | 868 |
| 4 | Critical ROP (Stage 4) | 600 |
| **Total** | | **3,866** |

### Sample Retinal Images

*(Note: Data samples shown here are representative of the 5 classes. Actual patient-level data splits and specific preprocessing depend on the configuration used.)*

| Normal | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|:---:|:---:|:---:|:---:|:---:|
| <img src="assets/sample_normal.jpg" width="150" height="150"> | <img src="assets/sample_stage1.jpg" width="150" height="150"> | <img src="assets/sample_stage2.jpg" width="150" height="150"> | <img src="assets/sample_stage3.jpg" width="150" height="150"> | <img src="assets/sample_stage4.jpg" width="150" height="150"> |

---

## 📈 Recorded Training and Validation Results

The following metrics were recorded during the model's training phase (using a learning rate of `0.000005` and an average epoch duration of `93.9 seconds`):

| Metric | Training Phase | Validation Phase |
|:---|:---:|:---:|
| **Loss** | 0.4779 | 0.6097 |
| **Accuracy** | 97.60% | 93.45% |
| **F1 Score** | 97.60% | **93.51%** |

*The best model checkpoint was saved based on the highest validation F1 score of 0.9351.*

### Evaluation Methodology & Scientific Integrity
- **Validation vs. Test:** The above metrics reflect **Training** and **Validation** performance on the splits used during model development. 
- **Independent Test Set:** Currently, independent hold-out test set metrics are not reported here. Do not interpret the 93.51% validation F1 score as the final generalized performance on unseen clinical test data unless explicitly verified via `scripts/predict.py` or the test evaluation block in `scripts/train.py`.

---

## 💻 Technology Stack
- **Python 3.x**
- **PyTorch** (Deep Learning framework)
- **Gradio** (Web UI for demonstration)
- **PyYAML** (Configuration management)

---

## 📂 Repository Structure

```text
ROP/
├── App.py                  # Gradio web application for model inference
├── assets/                 # Images for README presentation
├── configs/                
│   └── default.yaml        # YAML configuration for dataset, model, and training
├── Dataset/                # Image dataset directory (Normal, Stage_1 to Stage_4)
├── model/                  
│   └── best_rop_model.pth  # Saved weights of the best validation checkpoint
├── scripts/
│   ├── predict.py          # CLI script for single-image inference
│   └── train.py            # CLI script to train and evaluate the model
├── src/rop/                # Core Python package
│   ├── data/               # Datasets, transforms, and PyTorch Lightning-style DataModule
│   ├── evaluation/         # Metrics evaluation logic
│   ├── inference/          # Predictor class for deployment
│   ├── models/             # Architecture implementations (Backbone, Transformer, Fuzzy Layer)
│   ├── training/           # Trainer loop, optimizer, and checkpointing logic
│   └── config.py           # Configuration parsing utilities
└── tests/                  # Unit tests for models and training components
```

---

## 🛠️ Installation and Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ROP-HybridVision.git
   cd ROP-HybridVision
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you install the correct PyTorch version for your CUDA environment).*

---

## ⚙️ Dataset Configuration

The data pipeline expects the dataset to be structured under the path defined in `configs/default.yaml` (default: `Dataset/`), with subfolders representing each class. 

Update `configs/default.yaml` to adjust image size, batch size, or other training hyperparameters.

---

## 🚀 Training Instructions

To train the model from scratch, run the training script and pass the configuration file:

```bash
python scripts/train.py --config configs/default.yaml
```

The `Trainer` class will automatically log progress, perform validation at the end of each epoch, and save the best checkpoint (based on validation loss) into the `artifacts/checkpoints/` directory. The script also includes a final evaluation block on the test split.

---

## 🔬 Example Inference Workflow

To predict the ROP stage of a single new image from the command line, use the `predict.py` script:

```bash
python scripts/predict.py --image path/to/retina_image.jpg --checkpoint model/best_rop_model.pth
```
The script will output the predicted class index, class name, and the probability distribution across all 5 stages.

---

## 🌐 Gradio Web Demo

The repository includes a web interface built with Gradio for easy interaction.

To launch the demo:
```bash
python App.py
```
This will start a local web server (usually at `http://127.0.0.1:7860`). Open this URL in your browser to upload an image and view the model's fuzzy probabilities.

---

## ⚠️ Limitations and Responsible Medical AI Use

- **Research Only:** This software is an academic project. It is **not** an FDA-approved medical device and must not be used for clinical decision-making or diagnosis.
- **Dataset Bias:** The model's performance is strictly tied to the demographic and hardware distribution of the training dataset. It may not generalize to different hospital populations or camera types.
- **Data Privacy:** Do not upload patient-identifiable data to public instances of the Gradio application. Verify dataset licensing and ethics board (IRB) approvals before distributing or utilizing the dataset for further research.

---

## 🔮 Future Improvements
- **Cross-Validation:** Implement k-fold cross-validation to rigorously assess stability.
- **Explainability:** Integrate Grad-CAM or Attention Rollout to visualize what regions the Transformer is focusing on.
- **Hyperparameter Tuning:** Systematically optimize the number of fuzzy terms and transformer heads.

