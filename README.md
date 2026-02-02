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

Duplicate-Question-Pairs-Detection-ML-DL/
│
├── notebooks/
│ ├── Duplicate_question_pairs_using_ML.ipynb
│ └── duplicate-question-pair-using-dl.ipynb
│
├── dataset/
│ └── questions.csv
│
├── requirements.txt
└── README.md


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

# 📈 ML vs DL — Comparison

| Aspect | Machine Learning | Deep Learning |
|------|----------------|-------------|
| Input Representation | Hand-crafted features | Learned embeddings |
| Preprocessing | Heavy feature engineering | Minimal manual features |
| Models | Logistic Regression, Random Forest | Neural nets, Transformers |
| Compute Cost | Low–medium | High (GPU recommended) |
| Training Time | Faster | Slower |
| Scalability | Limited | Strong |
| Performance | Moderate–good | Higher |
| Interpretability | Easier | Harder |
| Deployment Simplicity | Easier | More complex |

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
