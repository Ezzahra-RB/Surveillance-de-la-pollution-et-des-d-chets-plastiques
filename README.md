# 🌍 Automatic Plastic Waste Detection using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

> **Master's Project in Artificial Intelligence** | Computer Vision Module  
> Supervised by **Dr. Issam QAFFOU**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Team](#team)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Challenges](#challenges)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project develops an **intelligent computer vision system** for automatic detection of plastic waste (bottles, bags, packaging, etc.) in real-world environments. By leveraging state-of-the-art deep learning techniques, particularly **YOLOv8**, our system enables environmental protection organizations, municipalities, and NGOs to:

- **Monitor plastic pollution** at scale
- **Automate visual data collection** from images and videos
- **Identify and quantify** waste in diverse settings (beaches, rivers, urban areas)

The system provides real-time detection with bounding boxes and automatic counting capabilities, making environmental monitoring more efficient and data-driven.

---

## ✨ Key Features

- ✅ **Real-time Detection**: Fast and accurate plastic waste identification using YOLOv8
- ✅ **Multi-Environment Support**: Trained on diverse contexts (beaches, urban areas, waterways)
- ✅ **Interactive Interface**: User-friendly Streamlit web application for testing and visualization
- ✅ **Comprehensive Pipeline**: End-to-end workflow from data preparation to deployment
- ✅ **Multiple Waste Categories**: Detection of bottles, bags, packaging, and other plastic debris
- ✅ **Performance Metrics**: Detailed evaluation with mAP, precision, and recall metrics

---

## 👥 Team

- **AIT EL ARBI Ezzahra** 
- **EL AAMRANI Zahira**
- **ERRAMI Hafssa**

---

## 🛠 Technology Stack

### Core Technologies
- **Python** 3.10+
- **PyTorch** - Deep learning framework
- **Ultralytics YOLOv8** - Object detection model
- **OpenCV** - Computer vision operations

### Visualization & Interface
- **Streamlit** - Interactive web application
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical graphics

### Dataset
- **TACO** (Trash Annotations in Context) - Annotated waste images in real-world settings

---

## 📊 Dataset

### TACO Dataset Overview

The **TACO (Trash Annotations in Context)** dataset provides high-quality annotated images of waste in natural environments.

#### Categories Detected:
- 🧴 Plastic bottles
- 🛍️ Plastic bags
- 📦 Packaging materials
- 🥫 Cans and mixed waste

#### Data Preparation:
1. **Annotation Validation**: Quality control and verification
2. **Format Conversion**: Transformed to YOLO-compatible format
3. **Dataset Split**: 
   - Training set
   - Validation set
   - Test set

#### Dataset Statistics:
- **Images**: 299 (test set)
- **Instances**: 1,014 annotated objects
- **Classes**: Multiple plastic waste categories

---

## 📁 Project Architecture

```
SURVEILLANCE-DE-LA-POLLUTION-ET-DES-DECHETS/
│
├── 📂 BaseLine/
│   ├── 📂 runs/                          # Training runs and results
│   ├── 📓 YOLOv8_Small.ipynb            # YOLOv8 Small model notebook
│   └── 📄 data.yaml                      # Dataset configuration
│
├── 📂 YOLOv8m_augmentation_best/
│   ├── 📂 runs/                          # Training runs with augmentation
│   ├── 🐍 app.py                         # Streamlit web application
│   ├── 📓 YOLOv8_medium_Aug.ipynb       # YOLOv8 Medium with augmentation
│   └── 📄 data.yaml                      # Dataset configuration
│
├── 📂 YOLOv11m_augmentation_cntxExtraction/
│   ├── 📂 Coco_Annotation_vesion/        # COCO format annotations
│   ├── 📂 recalib/                       # Model recalibration
│   ├── 📂 runs/                          # Training runs and results
│   ├── 📓 projet_cv_YOLOv11_medium_aug_featerExtraction.ipynb
│   └── 📄 data.yaml                      # Dataset configuration
│
├── 📄 requirements.txt                    # Project dependencies
└── 📄 README.md                          # Project documentation
```

### Directory Structure Explanation:

- **BaseLine/**: Initial baseline experiments with YOLOv8 Small model
- **YOLOv8m_augmentation_best/**: Optimized YOLOv8 Medium with data augmentation (best performing model)
- **YOLOv11m_augmentation_cntxExtraction/**: Advanced experiments with YOLOv11 Medium, including context extraction and feature analysis
- **runs/**: Contains training logs, validation results, and model checkpoints
- **data.yaml**: YOLO dataset configuration files with class definitions and paths

---

## 🔬 Methodology

### 1. **Data Analysis**
   - Comprehensive exploration of TACO dataset
   - Statistical analysis of waste distribution
   - Identification of class imbalances

### 2. **Data Preparation**
   - Annotation cleaning and validation
   - Conversion to YOLO format
   - Train/validation/test split (70/20/10)

### 3. **Model Training**
   - YOLOv8 architecture variants (small, medium, large)
   - Transfer learning from pre-trained weights
   - Custom hyperparameter tuning

### 4. **Performance Evaluation**
   - **mAP@50**: Mean Average Precision at IoU 0.5
   - **mAP@50-95**: Mean Average Precision across IoU thresholds
   - **Precision**: Ratio of correct positive predictions
   - **Recall**: Ratio of detected actual positives

### 5. **Model Optimization**
   - Data augmentation techniques (rotation, scaling, color jittering)
   - Hyperparameter optimization
   - Model calibration and threshold tuning

### 6. **Deployment**
   - Streamlit web application development
   - User interface design and testing
   - Documentation and usage guidelines

---

## 📈 Results

### Performance Metrics (Best Model)

| Metric | Value |
|--------|-------|
| **Overall mAP@50** | 0.545 |
| **Precision** | 0.696 |
| **Recall** | 0.476 |
| **Epocs** | 50 |
| **Training duration** | 1.9 h |

### Key Findings:
- ✅ Strong detection capability for visible, medium-to-large waste objects
- ✅ High precision indicates low false positive rate
- ⚠️ Recall can be improved for small or occluded objects
- 📊 Performance varies by waste category and environmental context

### Sample Detections:
The model successfully identifies:
- Plastic bottles in various orientations
- Shopping bags in cluttered environments
- Packaging materials in diverse lighting conditions

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-username/detection-dechets-plastiques.git
cd detection-dechets-plastiques
```

2. **Create Virtual Environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Usage Options

#### Option 1: Web Interface (Recommended)
```bash
streamlit run streamlit_app/app.py
```
Then open your browser at `http://localhost:8501`


---

## 🚧 Challenges

Throughout this project, we encountered and addressed several challenges:

1. **Environmental Variability**
   - Diverse lighting conditions (sunny beaches, shaded urban areas)
   - Different backgrounds and textures
   - Weather-related image quality variations

2. **Object Detection Difficulties**
   - Small or distant objects difficult to detect
   - Partial occlusion by vegetation or other objects
   - Similar colors between waste and background

3. **Dataset Limitations**
   - Class imbalance in waste categories
   - Limited samples for certain waste types
   - Geographic bias in image collection

4. **Computational Constraints**
   - Training time optimization
   - Balancing model accuracy vs. inference speed
   - Memory limitations during batch processing

---

## 📄 License

This project was developed for academic purposes as part of a Master's program in Artificial Intelligence.

**Dataset License**: TACO dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

For commercial use or redistribution, please review the TACO dataset terms and contact the project authors.

---

## 📧 Contact

### Project Authors

**AIT EL ARBI Ezzahra** | **EL AAMRANI Zahira** | **ERRAMI Hafssa**

📧 **Email**: [Contact the team](mailto:your-email@example.com)  
🔗 **GitHub**: [Project Repository](https://github.com/your-username/detection-dechets-plastiques)  
🎓 **Institution**: Master in Artificial Intelligence

### Supervisor

**Dr. Issam QAFFOU**  
Computer Vision & Deep Learning

---

## 🙏 Acknowledgments

- **TACO Dataset** creators for providing high-quality annotated data
- **Ultralytics** for the YOLOv8 framework
- **Dr. Issam QAFFOU** for guidance and supervision
- **Open-source community** for invaluable tools and resources

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for a cleaner planet 🌍

</div>
