# 🦴 KneeVision | Knee Osteoarthritis Severity Analyzer

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.1-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN%20%7C%20Transfer%20Learning-blue?style=for-the-badge)](https://en.wikipedia.org/wiki/Convolutional_neural_network)
[![Interpretability](https://img.shields.io/badge/Explainable%20AI-Grad--CAM-purple?style=for-the-badge)](https://arxiv.org/abs/1610.02391)

> **KneeVision** is an end-to-end Deep Learning and Explainable AI (XAI) platform designed for the automated grading of **Knee Osteoarthritis (KOA)** severity from radiographic (X-ray) images based on the clinical **Kellgren-Lawrence (KL)** classification system. It combines state-of-the-art Convolutional Neural Networks (CNNs), fine-tuning, ensemble strategies, and **Grad-CAM** interpretability into a sleek, interactive Streamlit web application.

---

## 📌 Table of Contents

- [Overview & Medical Background](#-overview--medical-background)
- [Kellgren-Lawrence (KL) Grading Scale](#-kellgren-lawrence-kl-grading-scale)
- [System Architecture](#-system-architecture)
- [Dataset & Data Preparation](#-dataset--data-preparation)
- [Model Exploration & Training](#-model-exploration--training)
- [Evaluation & Benchmark Results](#-evaluation--benchmark-results)
- [Explainable AI (Grad-CAM)](#-explainable-ai-grad-cam)
- [Interactive Web Application](#-interactive-web-application)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Dependencies](#-dependencies)
- [Medical Disclaimer](#-medical-disclaimer)

---

## 🩺 Overview & Medical Background

Knee Osteoarthritis (KOA) is one of the leading causes of chronic pain and mobility impairment worldwide, affecting hundreds of millions of individuals. Accurate early detection and objective severity grading are essential for planning clinical interventions, physical therapy, and surgical decisions (such as total knee arthroplasty).

In clinical practice, radiologists assess bilateral weight-bearing knee radiographs using the **Kellgren-Lawrence (KL)** grading system. However, manual grading is time-consuming, suffers from inter-observer variability, and depends heavily on specialist experience.

**KneeVision** addresses these challenges by:
1. **Automating multi-class classification** across all 5 Kellgren-Lawrence severity grades (0 to 4).
2. **Mitigating heavy class imbalance** using balanced loss weighting and targeted image augmentation.
3. **Benchmarking transfer learning backbones** (Xception, ResNet50, Inception-ResNet-v2) alongside metric-weighted soft ensemble models.
4. **Providing visual interpretability** via Gradient-weighted Class Activation Mapping (Grad-CAM) with automated peak region-of-interest detection to reveal the radiological basis behind every prediction.
5. **Offering an accessible clinical interface** built on Streamlit with real-time inference and diagnostic summaries.

---

## 🔬 Kellgren-Lawrence (KL) Grading Scale

The model classifies knee radiographs into five standard clinical severity grades:

| Grade | Clinical Label | Color Code | Radiographic Hallmarks |
| :---: | :---: | :---: | :--- |
| **0** | **Healthy (Normal)** | 🟢 Green | No radiological features of osteoarthritis; normal joint space and healthy articular margins. |
| **1** | **Doubtful** | 🟡 Yellow | Doubtful joint space narrowing (JSN) and possible minor osteophytic lipping / small bone spurs. |
| **2** | **Minimal (Mild)** | 🟠 Orange | Definite presence of osteophytes; possible mild joint space narrowing. |
| **3** | **Moderate** | 🔴 Red | Multiple moderate osteophytes, definite joint space narrowing, mild sclerosis, possible bone contour deformity. |
| **4** | **Severe** | 🟣 Purple | Marked and severe joint space narrowing, large bone spurs, significant subchondral sclerosis, and marked bone deformity. |

---

## 🏗 System Architecture

```text
               +-------------------------------------------------+
               |            Knee Radiograph (X-Ray)              |
               +-------------------------------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |           Preprocessing & Augmentation          |
               |      (Resize 224x224, Normalized Inputs,        |
               |       Brightness, Shift, Zoom, Flip)            |
               +-------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                   |                   |
                   v                   v                   v
           +---------------+   +---------------+   +---------------+
           |    Xception   |   |   ResNet50    |   | Inception-    |
           |  Fine-Tuned   |   |  Fine-Tuned   |   | ResNet-v2     |
           +---------------+   +---------------+   +---------------+
                   |                   |                   |
                   +-------------------+-------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |      F1-Score / Accuracy Weighted Ensemble      |
               |       or Fine-Tuned Single Backbone Model       |
               +-------------------------------------------------+
                                       |
                         +-------------+-------------+
                         |                           |
                         v                           v
         +-------------------------------+   +-------------------------------+
         |     Softmax Probabilities     |   |      Grad-CAM Activation      |
         |  (KL Grade 0 to 4 Prediction) |   |  (Anatomical Peak Detection)  |
         +-------------------------------+   +-------------------------------+
                         |                           |
                         +-------------+-------------+
                                       |
                                       v
               +-------------------------------------------------+
               |             Streamlit Dashboard UI              |
               |  (Diagnosis Card, Heatmap Overlay, Confidence)  |
               +-------------------------------------------------+
```

---

## 📊 Dataset & Data Preparation

The dataset comprises **9,786 knee radiographs** divided into standard partitions:

### Split Distribution

| Grade / Split | Grade 0 (Healthy) | Grade 1 (Doubtful) | Grade 2 (Minimal) | Grade 3 (Moderate) | Grade 4 (Severe) | Total Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 2,286 | 1,046 | 1,516 | 757 | 173 | **5,778** |
| **Validation** | 328 | 153 | 212 | 106 | 27 | **826** |
| **Test** | 639 | 296 | 447 | 223 | 51 | **1,656** |
| **Auto Test** | 604 | 275 | 403 | 200 | 44 | **1,526** |
| **Overall** | **3,857** | **1,770** | **2,578** | **1,286** | **295** | **9,786** |

### Addressing Class Imbalance & Data Augmentation
- **Class Weights**: Severe imbalance exists between Grade 0 (2,286 training images) and Grade 4 (173 training images). Class weights were dynamically computed using Scikit-Learn's balanced heuristic to penalize minority-class misclassifications proportionally:
  $$\text{weight}_c = \frac{N_{\text{samples}}}{N_{\text{classes}} \times N_c}$$
- **Data Augmentation**:
  - `horizontal_flip = True`
  - `brightness_range = [0.3, 0.8]`
  - `width_shift_range = [-50, 0, 50, 30, -30]`
  - `zoom_range = 0.1`
  - `fill_mode = "nearest"`
- **Target Resolution**: $224 \times 224 \times 3$ with model-specific normalization.

---

## 🧠 Model Exploration & Training

Three deep convolutional architectures pre-trained on ImageNet were fine-tuned:

1. **Xception**: Deep convolutional network featuring depthwise separable convolutions that decouple spatial correlations from cross-channel correlations.
2. **ResNet50**: 50-layer residual architecture using identity skip connections to mitigate vanishing gradient problems in deep networks.
3. **Inception-ResNet-v2**: Hybrid architecture merging multi-scale Inception factorization blocks with residual connections for superior representation capacity.

### Fine-Tuning Setup
- **Classification Head**: `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(5, activation='softmax')`.
- **Optimizer**: Adam with learning rate adaptation.
- **Loss**: Categorical Crossentropy with balanced class weighting.
- **Regularization & Callbacks**:
  - `EarlyStopping(monitor='val_loss', min_delta=0.01, patience=8, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='loss', factor=0.1, min_delta=0.01, min_lr=1e-10, patience=4)`

### Ensemble Strategies
In `02_ensemble_models.ipynb`, three soft voting ensemble paradigms combine model predictions:
1. **Uniform Mean (`ensemble_mean`)**: Equal weighting across all models.
2. **Accuracy-Weighted (`ensemble_acc`)**: Weights models according to class-level training accuracy.
3. **F1-Score-Weighted (`ensemble_f1`)**: Weights predictions dynamically based on per-class F1-scores, maximizing minority class sensitivity.

---

## 📈 Evaluation & Benchmark Results

### Model Comparison Summary

| Model Configuration | Balanced Accuracy | Overall Accuracy | Training Execution Time |
| :--- | :---: | :---: | :---: |
| **Ensemble (F1-Weighted)** 🏆 | **71.43%** | **68.72%** | ~3 min (Inference) |
| **Xception (Fine-Tuned)** | **67.00%** | **61.65%** | ~68 min |
| **ResNet50 (Fine-Tuned)** | **65.00%** | **62.00%** | ~80 min |
| **Inception-ResNet-v2 (Fine-Tuned)** | **64.00%** | **61.00%** | ~56 min |

### Detailed Classification Report (Test Set — Xception Backbone)

```text
              precision    recall  f1-score   support

    Grade 0        0.82      0.60      0.69       639
    Grade 1        0.31      0.46      0.37       296
    Grade 2        0.65      0.66      0.66       447
    Grade 3        0.73      0.71      0.72       223
    Grade 4        0.59      0.92      0.72        51

   accuracy                            0.62      1656
  macro avg        0.62      0.67      0.63      1656
weighted avg       0.66      0.62      0.63      1656

Test Loss: 0.8618  |  Accuracy: 61.65%  |  Balanced Accuracy: 67.00%
```

### Detailed Classification Report (Test Set — F1-Weighted Ensemble)

```text
              precision    recall  f1-score   support

    Grade 0        0.82      0.73      0.77       639
    Grade 1        0.36      0.43      0.39       296
    Grade 2        0.69      0.73      0.71       447
    Grade 3        0.86      0.79      0.82       223
    Grade 4        0.73      0.90      0.81        51

   accuracy                            0.69      1656
  macro avg        0.69      0.71      0.70      1656
weighted avg       0.71      0.69      0.69      1656

Accuracy: 68.72%  |  Balanced Accuracy: 71.43%
```

*Key Insight:* The F1-weighted ensemble demonstrates remarkable sensitivity on Grade 4 severe osteoarthritis (90% recall) and Grade 3 moderate osteoarthritis (79% recall), validating its clinical utility in flagging advanced disease progression.

---

## 🔍 Explainable AI (Grad-CAM)

Deep neural networks in healthcare must never be opaque black boxes. KneeVision integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**:

1. **Feature Map Extraction**: Computes the gradient of the predicted class score with respect to the output feature map of the final convolutional layer (before Global Average Pooling).
2. **Channel Weighting**: Averages the gradients over spatial dimensions to capture the importance of each feature map activation channel.
3. **Heatmap Generation**: Generates a 2D weighted linear combination of feature maps, passes it through ReLU (to retain positive influences), and normalizes it.
4. **Peak Localization**: Implements peak-detection algorithms to locate local activation maxima separated by a minimum spatial distance (identifying discrete joint compartments).
5. **Clinical Superimposition**: Blends the jet heatmap with the original radiograph and renders bounding markers around key focal regions (osteophytes, tibial spines, joint space narrowing).

---

## 💻 Interactive Web Application

The application (`app/app.py`) provides an intuitive dashboard for practitioners and researchers:

- 📤 **Drag & Drop Upload**: Accepts standard medical image formats (`.jpg`, `.jpeg`, `.png`).
- 🏷️ **Diagnosis Summary**: Displays the predicted KL grade, clinical severity title, and color-coded status badge.
- 📊 **Raw Probability Distribution**: Details individual model output probabilities alongside a comparative horizontal bar chart.
- 🔬 **Grad-CAM Visualizer**: Displays an interpretable heatmap overlay highlighting anatomical areas guiding model decision-making.
- 📖 **Sidebar Reference Guide**: Includes the full Kellgren-Lawrence criteria for rapid clinical cross-referencing.

---

## 📁 Repository Structure

```text
.
├── .gitignore                          # Git exclusions (datasets, weights, virtual environments, caches)
├── requirements.txt                    # Project dependencies
├── command                             # Quick execution script reference
├── 9007827R.png                        # Sample knee radiograph for testing
├── README.md                           # Comprehensive project documentation
├── app/
│   └── app.py                          # Streamlit web application & Grad-CAM pipeline
├── dataset/                            # Radiograph dataset (local / ignored from git)
│   ├── train/                          # Training split (5,778 images across grades 0-4)
│   ├── val/                            # Validation split (826 images across grades 0-4)
│   ├── test/                           # Test split (1,656 images across grades 0-4)
│   └── auto_test/                      # Automated validation set (1,526 images)
└── src/
    ├── 01_data_preparation.ipynb       # Exploratory analysis, augmentation, class weight calculation
    ├── 02_model_xception.ipynb         # Xception model training and fine-tuning
    ├── 02_model_resnet50.ipynb         # ResNet50 model training and validation
    ├── 02_model_inception_resnet_v2.ipynb # Inception-ResNet-v2 model training
    ├── 02_ensemble_models.ipynb        # Mean, accuracy, and F1-weighted ensemble pipelines
    ├── 03_best_model_on_test_xception.ipynb # Final evaluation, test set metrics, and Grad-CAM
    └── models/
        └── model_Xception_ft.hdf5      # Trained weights for fine-tuned Xception model
```

---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python **3.10** or higher
- Git
- NVIDIA GPU with CUDA support (recommended for model training, optional for Streamlit inference)

### 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/KneeVision.git
cd KneeVision
```

### 3. Set Up a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional: specific tested pinned versions)*
```bash
pip install streamlit==1.37.1 numpy==1.23.5 matplotlib==3.7.5 pillow==10.4.0 tensorflow==2.13.1
```

### 5. Launch the Streamlit Web Application
Run the Streamlit application from the root directory:
```bash
streamlit run app/app.py
```
Or navigate into the `app` directory:
```bash
cd app
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the analyzer.

### 6. Explore Jupyter Notebooks
To inspect the exploratory data analysis, model training pipelines, and ensemble experimentation:
```bash
jupyter notebook
```
Navigate to the `src/` folder and open any notebook (`01_data_preparation.ipynb`, `02_ensemble_models.ipynb`, etc.).

---

## 📦 Dependencies

The core libraries utilized in this project are:

- **TensorFlow** (`>= 2.13.0`): Deep learning model construction, fine-tuning, and Grad-CAM gradient calculation.
- **Streamlit** (`>= 1.37.0`): Reactive web dashboard interface.
- **Scikit-Learn** (`>= 1.3.0`): Metrics computation, balanced class weights, and classification reports.
- **NumPy** (`>= 1.23.0`): Numerical computation and array manipulations.
- **Pandas** (`>= 2.0.0`): Tabular data structures and file indexing.
- **Matplotlib & Seaborn**: Statistical data visualizations, Grad-CAM overlays, and bar charts.
- **Pillow**: Image loading, formatting, and bounding box drawing.
- **Jupyter**: Interactive experimentation and reproducible notebooks.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: This software is intended strictly for **educational, academic, and research purposes**. It has not undergone clinical certification (such as FDA 510(k) clearance or CE marking as software as a medical device). It is not designed to replace clinical diagnoses, assessments, or treatment plans formulated by certified radiologists, orthopedic surgeons, or licensed medical professionals.
