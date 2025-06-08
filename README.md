
# Aspect-Based Sentiment Analysis (ABSA)

This repository contains the official implementation for the research-project **"Aspect-Based Sentiment Analysis (ABSA)"** by **Nikhil Singh**, **Sanket Madaan**, and **Digvijay Singh Rathore**.

---

## 1. Project Description

This project presents a comprehensive study on **Aspect-Based Sentiment Analysis (ABSA)** - a fine-grained sentiment classification task that identifies and analyzes sentiments expressed towards specific aspects of entities in text. 

We implement and conduct a head-to-head comparison between two distinct paradigms:

- A **traditional machine learning pipeline** employing feature engineering with classifiers like **SVM (Support Vector Machine)** and **Logistic Regression**.  
- A **transformer-based pipeline** leveraging **BERT (Bidirectional Encoder Representations from Transformers)** for fine-tuning and deep contextual understanding.  

Our goal is to benchmark these approaches on the **SemEval-2014 Task 4 dataset** under a unified framework, providing open-source, reproducible implementations for future research.

---

## 2. Directory Structure

```
INLPPROJECT-MAIN/
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   ├── bert/
│   │   ├── __init__.py
│   │   ├── BERTModel.py
│   │   └── HybridBERTClassifier.py
│   ├── traditional/
│   │   ├── __init__.py
│   │   ├── SVMModel.py
│   │   └── train_modelLR.py
│   └── utils/
│       ├── __init__.py
│       └── aspect_extractor.py
├── outputs/
│   ├── analysis/
│   ├── bert/
│   ├── hybrid/
│   └── traditional/
├── preprocessing/
│   ├── __init__.py
│   ├── bert_preprocess.py
│   ├── preprocess.py
│   └── preprocesstest.py
├── runners/
│   ├── run_bert_classifier.py
│   └── run_hybrid_classifier.py
├── .gitignore
├── comparison.py
├── download_nltk.py
├── error_analysis.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 3. Methodology

### Traditional Pipeline

- **Preprocessing**: Clean, tokenize, and structure text into aspect-sentence pairs.
- **Feature Engineering**: Generate TF-IDF vectors, n-grams, sentiment lexicons, etc.
- **Classification**: Train SVM and Logistic Regression classifiers.

### Transformer-Based Pipeline

- **Input Formatting**: Format each input as `[CLS] sentence [SEP] aspect [SEP]`.
- **Fine-tuning**: Fine-tune `bert-base-uncased` with a classification layer using cross-entropy loss.

---

## 4. How to Run the Project

### Step 1: Setup and Installation

Clone the repository:

```bash
git clone https://github.com/Nikhil-1920/Aspect-Based-Sentiment-Analysis.git
cd Aspect-Based-Sentiment-Analysis
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Download NLTK & spaCy resources:

```bash
python3 -m spacy download en_core_web_sm
python3 download_nltk.py
```

Download the dataset:

- Download the **SemEval-2014 Task 4** dataset from the official source.
- Unzip and place files like `Restaurants_Train.xml`, `Laptops_Test_Gold.xml` in `data/raw/`.

---

### Step 2: Data Preprocessing

Run scripts to preprocess the data.

Traditional model data:

```bash
python3 preprocessing/preprocess.py        # For Train/Val
python3 preprocessing/preprocesstest.py    # For Test
```

BERT model data:

```bash
python3 preprocessing/bert_preprocess.py
```

---

### Step 3: Train Traditional Models

Train the Logistic Regression model:

```bash
python3 models/traditional/train_modelLR.py
```

Train the SVM model:

```bash
python3 models/traditional/SVMModel.py
```

Outputs are saved in `outputs/traditional/`.

---

### Step 4: Train Transformer Models (BERT & Hybrid)

Train the BERT-based model:

```bash
python3 runners/run_bert_classifier.py
```

Train the Hybrid model:

```bash
python3 runners/run_hybrid_classifier.py
```

Outputs are saved in `outputs/bert/` and `outputs/hybrid/`.

---

### Step 5: Analysis and Comparison

Run comprehensive error analysis:

```bash
python3 error_analysis.py
```

Run model comparison script:

```bash
python3 comparison.py
```

---

## 5. Evaluation

Models are evaluated using standard metrics:

- **Accuracy**: Overall prediction correctness.
- **Precision, Recall, F1-Score**: Macro-averaged across sentiment classes.
- **Confusion Matrix**: To visualize true vs predicted sentiment distributions.

---

## Contact

For any queries or contributions, feel free to open issues or pull requests in this repository.
