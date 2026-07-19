<div align="center">

# 🛰️ Satellite Land-Use Classification & Temporal Change Detection

### Deep Learning • Transfer Learning • Computer Vision • Remote Sensing • Streamlit

<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv">
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/License-MIT-success?style=for-the-badge">

</p>

### 🚀 Intelligent Satellite Image Analysis using ResNet18 & Feature Embeddings

Predict land-use categories from satellite imagery, compare two satellite images captured at different time periods, detect land-use changes using deep feature embeddings and cosine similarity, visualize changes using heatmaps, and generate downloadable analysis reports.

---

### 🌐 Live Demo

> 🚧 **Coming Soon**  
> https://your-streamlit-app.streamlit.app

### 🎥 Project Demo

> 🚧 **Coming Soon**  
> https://youtu.be/your-demo-video

</div>

---

# 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Project Workflow](#-project-workflow)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Datasets](#-datasets)
- [Project Structure](#-project-structure)
- [Model Training Pipeline](#-model-training-pipeline)
- [Results](#-results)
- [Dashboard Preview](#-dashboard-preview)
- [Installation Guide](#-installation-guide)
- [Run Locally](#-run-locally)
- [How to Use](#-how-to-use)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)
- [License](#-license)

---

# 📖 Project Overview

Satellite imagery plays a crucial role in monitoring urban development, agriculture, forests, transportation networks, and environmental changes. Manual inspection of satellite images is both time-consuming and resource-intensive.

This project presents a complete **Deep Learning-based Land-Use Classification and Temporal Change Detection System** capable of automatically classifying satellite images into land-use categories and identifying significant land-cover changes between two different time periods.

The application leverages **Transfer Learning with ResNet18**, deep feature embeddings, cosine similarity, and an interactive **Streamlit Dashboard** to provide accurate predictions and intuitive visualizations.

Unlike traditional image classification systems, this project not only predicts the land-use category but also determines whether meaningful environmental changes have occurred between two satellite images.

---

# 🎯 Project Objectives

The major objectives of this project are:

- 🌍 Classify satellite images into multiple land-use categories
- 🧠 Utilize Transfer Learning for improved classification accuracy
- 📈 Compare two satellite images using deep feature embeddings
- 📊 Measure similarity using Cosine Similarity
- 🔥 Generate heatmaps highlighting changed regions
- 🚨 Detect significant land-use changes automatically
- 📄 Generate downloadable analysis reports
- 💻 Provide an interactive Streamlit dashboard for end users

---

# ✨ Key Features

✅ Land-Use Classification

✅ Transfer Learning using ResNet18

✅ Feature Embedding Extraction (512-dimensional vectors)

✅ Cosine Similarity Based Change Detection

✅ Heatmap Visualization

✅ Confidence Score Prediction

✅ Before vs After Image Comparison

✅ Downloadable Analysis Report

✅ Interactive Streamlit Dashboard

✅ Clean Modular Code Structure

✅ Jupyter Notebook Based Training Pipeline

✅ End-to-End Deep Learning Workflow

---

# 🔄 Project Workflow

```text
                Satellite Images
                       │
                       ▼
             Image Preprocessing
                       │
                       ▼
             Transfer Learning (ResNet18)
                       │
      ┌────────────────┴────────────────┐
      │                                 │
      ▼                                 ▼
Land-Use Classification        Feature Extraction
      │                                 │
      └──────────────┬──────────────────┘
                     ▼
            Cosine Similarity
                     │
                     ▼
          Temporal Change Detection
                     │
                     ▼
            Heatmap Generation
                     │
                     ▼
          Streamlit Interactive Dashboard
```

---

# 🏗️ System Architecture

```text
                EuroSAT Dataset
                        │
                        ▼
                 Image Preprocessing
                        │
                        ▼
              Transfer Learning Model
                 (ResNet18 Backbone)
                        │
      ┌─────────────────┴─────────────────┐
      ▼                                   ▼
Land-Use Prediction               Feature Embeddings
      │                                   │
      └─────────────────┬─────────────────┘
                        ▼
              Cosine Similarity Score
                        │
            Similarity > Threshold ?
                 │                  │
              Yes                  No
                 │                  │
         No Significant      Land-Use Change
              Change             Detected
                        │
                        ▼
                 Heatmap Visualization
                        │
                        ▼
             Downloadable Report + Dashboard
```

---

# 💻 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Transfer Learning | ResNet18 |
| Machine Learning | Scikit-Learn |
| Image Processing | OpenCV, Pillow |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Notebook | Jupyter Notebook |
| Version Control | Git & GitHub |

---
