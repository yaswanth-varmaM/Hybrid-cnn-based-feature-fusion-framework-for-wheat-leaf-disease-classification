# 🌾 Wheat Disease Detection Using Hybrid CNN + Swin Transformer

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced deep learning system for automatic detection and classification of wheat plant diseases using a **Hybrid CNN + Swin Transformer** architecture. This project combines the local feature extraction capabilities of Convolutional Neural Networks with the global context modeling of Vision Transformers.

![Wheat Disease Detection](arc.png)

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Introduction

Wheat is one of the most important cereal crops globally, and diseases can significantly reduce crop yield. Manual identification of plant diseases is time-consuming and requires expert knowledge. This project leverages **deep learning** and **computer vision** to automatically detect diseases in wheat plants from images.

### Problem Statement
- Manual disease identification is labor-intensive and error-prone
- Early detection is crucial for preventing crop loss
- Farmers need accessible tools for quick disease diagnosis

### Solution
A hybrid deep learning model combining:
- **CNN (ResNet50)**: Extracts local features like edges, textures, and patterns
- **Swin Transformer**: Captures global context using shifted window self-attention
- **Fusion Layer**: Combines both representations for robust classification

---

## ✨ Features

- 🔬 **Hybrid Architecture**: Combines CNN and Vision Transformer for superior accuracy
- 🌐 **Web Application**: User-friendly Flask-based interface for easy image upload
- 📊 **PDF Reports**: Generate downloadable reports with prediction details
- 🎯 **4 Disease Classes**: Detects Crown and Root Rot, Leaf Rust, Wheat Loose Smut, and Healthy Wheat
- 🚀 **Real-time Prediction**: Fast inference with confidence scores
- 📱 **Responsive Design**: Works on desktop and mobile devices

---

## 🏗️ Architecture

### Hybrid CNN + Swin Transformer

```
Input Image (224×224×3)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                                                   │
│  ┌─────────────────┐    ┌─────────────────────┐  │
│  │   CNN Branch    │    │  Swin Transformer   │  │
│  │   (ResNet50)    │    │      Branch         │  │
│  │                 │    │                     │  │
│  │  Conv1 → BN     │    │  Patch Embedding    │  │
│  │     ↓           │    │       ↓             │  │
│  │  Layer1         │    │  Swin Block 1       │  │
│  │     ↓           │    │       ↓             │  │
│  │  Layer2         │    │  Swin Block 2       │  │
│  │     ↓           │    │       ↓             │  │
│  │  GAP (512)      │    │  Features (768)     │  │
│  └────────┬────────┘    └──────────┬──────────┘  │
│           │                        │              │
│           └──────────┬─────────────┘              │
│                      ▼                            │
│              Concatenation (1280)                 │
│                      │                            │
│                      ▼                            │
│              Fusion Layer                         │
│              FC(512) → FC(256)                    │
│                      │                            │
│                      ▼                            │
│              Classification Head                  │
│              FC(4) → Softmax                      │
│                      │                            │
│                      ▼                            │
│              Output (4 classes)                   │
│                                                   │
└───────────────────────────────────────────────────┘
```

### Why Hybrid Architecture?

| Component | Strength | Role |
|-----------|----------|------|
| CNN (ResNet50) | Local feature extraction | Captures edges, textures, color patterns |
| Swin Transformer | Global context modeling | Understands spatial relationships, long-range dependencies |
| Fusion Layer | Feature integration | Combines local and global representations |

---

## 📊 Dataset

The dataset contains wheat plant images classified into 4 categories:

| Class | Description | Sample Count |
|-------|-------------|--------------|
| 🟤 Crown and Root Rot | Fungal disease affecting roots | ~1000+ |
| 🟢 Healthy Wheat | Disease-free wheat plants | ~1000+ |
| 🟠 Leaf Rust | Fungal infection on leaves | ~1000+ |
| ⚫ Wheat Loose Smut | Fungal disease affecting grains | ~1000+ |

**Total Images**: ~4500  
**Image Format**: JPG, JPEG, PNG  
**Split**: 80% Training, 20% Testing (Stratified)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Wheat-Disease-Detection-.git
cd Wheat-Disease-Detection-
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv wheat
.\wheat\Scripts\activate

# Linux/Mac
python -m venv wheat
source wheat/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
flask>=2.0.0
opencv-python>=4.5.0
pillow>=9.0.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.62.0
reportlab>=3.6.0
seaborn>=0.11.0
```

---

## 🚀 Usage

### 1. Train the Model
Open and run the Jupyter notebook:
```bash
jupyter notebook "Wheat_Disease_Detection_Hybrid_CNN_SwinTransformer.ipynb"
```

### 2. Run the Web Application
```bash
python app_hybrid.py
```
Then open your browser at: **http://127.0.0.1:5000**

### 3. Test the Model
```bash
# Test on sample images from dataset
python test_model.py

# Test a specific image
python test_model.py path/to/your/image.jpg
```

---

## 📁 Project Structure

```
Wheat-Disease-Detection-/
│
├── 📂 Images/                    # Dataset folder
│   ├── Crown and Root Rot/
│   ├── Healthy Wheat/
│   ├── Leaf Rust/
│   └── Wheat Loose Smut/
│
├── 📂 models/                    # Saved model weights
│   └── hybrid_cnn_swin_wheat_best.pth
│
├── 📂 templates/                 # HTML templates
│   ├── index_simple.html
│   └── predict_simple.html
│
├── 📂 static/                    # Static files
│   ├── css/
│   └── uploads/
│
├── 📂 Plots/                     # Training plots and visualizations
│
├── 📜 app_hybrid.py              # Flask web application
├── 📜 test_model.py              # Model testing script
├── 📜 Wheat_Disease_Detection_Hybrid_CNN_SwinTransformer.ipynb
├── 📜 requirements.txt           # Python dependencies
├── 📜 PROJECT_REPORT.md          # Project report
└── 📜 README.md                  # This file
```

---

## 📈 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | ~95%+ |
| **Model Parameters** | ~52M |
| **Inference Time** | <1 second |

### Classification Report

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Crown and Root Rot | 0.94 | 0.93 | 0.94 |
| Healthy Wheat | 0.97 | 0.98 | 0.97 |
| Leaf Rust | 0.95 | 0.94 | 0.94 |
| Wheat Loose Smut | 0.93 | 0.94 | 0.93 |

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | PyTorch, timm, Swin Transformer |
| **Computer Vision** | OpenCV, PIL, torchvision |
| **Web Framework** | Flask |
| **Data Science** | NumPy, Pandas, Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn |
| **Report Generation** | ReportLab |

---

## 🔮 Future Scope

- [ ] Add more disease categories
- [ ] Mobile application development
- [ ] Integration with IoT devices for real-time field monitoring
- [ ] Multi-crop disease detection
- [ ] Explainable AI (Attention visualization)
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] API development for third-party integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

**Project Team members :**
M.yaswanth varma,
G.yaswasree,
P.tharun,
S.sandya,
with guidance of Dr.N.babu sir.

---

## 🙏 Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) - Microsoft Research
- Dataset contributors

---
**THANK YOU**

<p align="center">
  Made with ❤️ for Agricultural AI
</p>
