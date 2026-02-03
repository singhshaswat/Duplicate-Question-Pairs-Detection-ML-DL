# Duplicate Question Pair Detection using Machine Learning & Deep Learning

This project aims to identify whether two questions are semantically duplicate using Natural Language Processing techniques.  
It implements and compares **traditional Machine Learning models** with **Deep Learning and Transformer-based approaches** on the Quora Question Pairs dataset.

---

## 📌 Project Motivation

Duplicate questions are common on platforms like Quora and StackOverflow.  
Automatically detecting them helps:

- Reduce redundant content  
- Improve search relevance  
- Enhance moderation systems  
- Save computational and human effort  

This repository demonstrates **two complete pipelines**:

- 📊 Feature-based ML classification  
- 🤖 Neural networks & transformer-based models  

---

## 📂 Repository Structure

```
Duplicate-Question-Pairs-Detection-ML-DL/
│
├── notebooks/
│   ├── Duplicate_question_pairs_using_ML.ipynb
│   └── duplicate-question-pair-using-dl.ipynb
│
├── dataset/
│
├── requirements.txt
└── README.md
```


---

## 📊 Dataset Description

Source: Quora Question Pairs dataset

Each row contains two questions and a label indicating whether they are duplicates.

| Column | Description |
|------|-----------|
| qid1 | ID of first question |
| qid2 | ID of second question |
| question1 | First question text |
| question2 | Second question text |
| is_duplicate | 1 = duplicate, 0 = not duplicate |

---

# ⚙️ Notebook 1 — Machine Learning Approach

📄 **File:** `Duplicate_question_pairs_using_ML.ipynb`

This notebook follows a **classical NLP + feature engineering pipeline**.

---

## 🔍 Pipeline Steps

### ✅ Text Preprocessing
- Lowercasing
- Removing punctuation & special characters
- Tokenization
- Stopword removal
- Lemmatization

---

### ✅ Feature Engineering

For each pair of questions, the notebook computes:

- Length of each question
- Absolute length difference
- Word overlap ratio
- Common word count
- Fuzzy similarity scores
- Token-level statistics

These numerical features are used for ML classification.

---

### ✅ Models Trained

- Logistic Regression  
- Random Forest Classifier  

---

### ✅ Evaluation

Models are evaluated using:

- Accuracy score
- Confusion matrix
- Classification report

Visualizations are included to analyze:

- Feature distributions
- Duplicate vs non-duplicate patterns

---

# 🤖 Notebook 2 — Deep Learning Approach

📄 **File:** `duplicate-question-pair-using-dl.ipynb`

This notebook focuses on **neural and transformer-based NLP models** using PyTorch and HuggingFace Transformers.

---

## 🔍 Pipeline Steps

### ✅ Tokenization
- Transformer tokenizer
- Padding & truncation

---

### ✅ Deep Learning Models

- Neural network classifier
- Transformer-based architecture (BERT-style encoder)
- Fine-tuning on question pairs

---

### ✅ Training Strategy

- Binary classification objective
- Adam optimizer
- GPU-accelerated training
- Validation monitoring

---

### ✅ Evaluation

- Accuracy
- Loss curves
- Validation performance

---

# 📊 ML vs DL — Model Performance Comparison

The table below summarizes the quantitative performance of classical machine learning models versus the transformer-based deep learning approach.

| Model                      | Validation Size | Train Acc | Val/Test Acc | Weighted F1 | ROC–AUC   |
| -------------------------- | --------------- | --------- | ------------ | ----------- | --------- |
| Logistic Regression        | 9,970           | 83.6%     | 75.8%        | 0.75        | 0.79      |
| Random Forest              | 9,970           | 87.3%     | 76.1%        | 0.75        | 0.80      |
| Transformer (MiniLM-L6-v2) | 80,870          | 91.9%     | **89.1%**    | **0.89**    | **0.955** |

---
# 🔍 Key Insights

- Transformer-based models achieved a +13% absolute accuracy improvement over Random Forest baselines.

- Weighted F1-score improved from 0.75 → 0.89, demonstrating superior handling of class imbalance.

- ROC–AUC of 0.955 indicates excellent separability between duplicate and non-duplicate questions.

- Early stopping during training ensured stable convergence and prevented overfitting.

- Classical ML models rely heavily on hand-crafted lexical features, whereas transformers learn semantic representations end-to-end.

- The deep learning model follows a Siamese setup built on sentence-transformers/all-MiniLM-L6-v2.

---
# ⚙️ Deep Learning Model Details

- Architecture: Siamese Transformer Encoder

- Backbone: sentence-transformers/all-MiniLM-L6-v2

- Objective: Binary classification

- Optimizer: Adam

- Regularization: Early stopping & checkpointing

- Hardware: GPU-accelerated training

- Validation Set Size: 80,870 question pairs

- Metrics Used: Accuracy, Weighted F1, ROC–AUC, confusion matrix

---
# 📌 Summary

- Classical ML pipelines provide strong baselines with fast training and simple deployment.

- Transformer models scale better and capture deep semantic relationships.

- The DL system is production-ready with state-of-the-art performance on large validation sets.

---

# 🚀 How to Run Locally

### 1️⃣ Clone the repository



git clone https://github.com/singhshaswat/Duplicate-Question-Pairs-Detection-ML-DL.git

cd Duplicate-Question-Pairs-Detection-ML-DL


---

### 2️⃣ Install dependencies



pip install -r requirements.txt


---

### 3️⃣ Launch notebooks



jupyter notebook


Open files inside the `notebooks/` directory.

---

# 📦 Dependencies

Libraries used in this project:

- numpy  
- pandas  
- matplotlib  
- seaborn  
- nltk  
- scipy  
- scikit-learn  
- tensorflow  
- torch  
- transformers  

---

# 🧑‍💻 Author

**Shaswat Singh**  
---
