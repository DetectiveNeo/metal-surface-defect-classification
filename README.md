# Metal Surface Defect Classification

This project focuses on classifying six types of metal surface defects using a Convolutional Neural Network (CNN) build using **Pytorch**. In real-world manufacturing environments, surface defect identification is still predominantly manual. This project automates defect classification, enabling faster quality control and improved reliability in industrial processes.

This project follows an industry-standard modular architecture with a fully structured training pipeline, unit tests, and reproducible configuration management. Transfer learning using ResNet18 is combined with checkpointing and early stopping to achieve optimized performance.

---

## 1. Problem Statement

Hot-rolled steel products commonly exhibit surface defects such as cracks, inclusions, roll marks, and stains.  
Manual inspection is:

- slow  
- inconsistent  
- non-scalable  
- prone to human error  

This project implements an **AI-driven defect classification system** that automatically identifies six types of defects with high accuracy.

---

## 2. Dataset Description

The dataset used is the **NEU Metal Surface Defect Database**, containing **1,800 grayscale images** equally divided into six defect categories:

| Class| Description         | Samples |
|------|---------------------|---------|
| 1    | Rolled              | 300     |
| 2    | Patches             | 300     |
| 3    | Crazing             | 300     |
| 4    | Pitted              | 300     |
| 5    | Inclusion           | 300     |
| 6    | Scratches           | 300     |

### Dataset Split Used in This Project

- **Training:** 276 images/class  
- **Validation:** 12 images/class  
- **Testing:** 12 images/class  

Images were resized, normalized, and augmented during preprocessing.

---

## 3. Project Structure

```
metal-surface-defect-classification/
│
├── configs/config.yaml # Contain all the parameters that controls the program
├── data # Contains the dataset
├── src/
│ ├── config/ # Config class for paths & training params
│ ├── data/ dataset.py # Dataset Class and Data loaders
│ ├── models # Model architecture + training + prediction
│ ├── utils/ # Helper utilities (plots, metrics, logging)
│
├── reports/ # Saved checkpoints, experiments, metrics
├── notebooks/ # EDA, initial experiments
├── tests/ # Unit tests for training & model pipeline
├── requirements.txt
└── README.md
```
This modular structure follows **industry best practices** and enables easy reuse, testing, and automation.

---

## 4. Model Architecture

This project uses **ResNet18** with transfer learning:

- Pretrained on ImageNet  
- Backbone optionally frozen during initial training  
- Final fully connected layer replaced for **6-class classification**

Model architecture is Frozen backbone ResNet18 connected to a linear classifer Layer for **6-class classification**.
Number of Trainable Parameters : **3078**


## 5. Training Pipeline

Training is handled by `src/models/train.py` and includes:

- reproducible seed setup  
- dataloader creation  
- training and validation loops  
- loss/accuracy tracking  
- checkpoint saving  
- metric storage (JSON + CSV)  
- curve plotting  

Parameters are defined in the `configs/config.yaml` file:
- Batch Size: **32**
- Epochs: **10**
- Learning Rate: **0.001**
- Loss Function: **CrossEntropyLoss**
- Patience Level: **5**  

---

## 6. How to Train / predict the Model

Run:
For training the model
``bash
python src/models/train.py

For running the predictions
``bash
python src/models/predictions.py

---

## 7. Results 

- Training Accuracy : **97.40%**
- Validation Accuracy : **100.00%
- Test Accuracy : **98.61%**


