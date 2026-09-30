# Toxic Comment Detection with Scikit-learn

## Project overview

This repository contains my project for **Advanced Python for NLP**.

The project implements a **multi-label toxic comment classifier** using the
**Jigsaw Toxic Comment Classification Challenge** dataset and techniques from
the course, especially **scikit-learn**.

The classifier predicts six independent labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

Because a comment can receive several labels at the same time, this is a
**multi-label classification** task.

## Research questions

1. How effectively can TF-IDF features combined with Logistic Regression detect
   the six toxicity categories in the Jigsaw dataset?
2. How does model performance differ between frequent and rare toxicity labels?
3. What types of comments cause errors, especially comments containing negation?

## Repository structure

```text
toxic-comment-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── toxicity_sklearn.ipynb
├── data_loader.py
│
│
├── data/
│   └── README.md
│
├── models/
│   └── .gitkeep
│
└── report/
    ├── Toxic_Comment_Detection_Report.pdf
    
```

### Main entry point

The main entry point is:

```text
toxicity_sklearn.ipynb
```

Run the notebook from top to bottom to reproduce the training and evaluation.

### Modules

- `toxicity_sklearn.ipynb` - complete training, evaluation, error analysis, model saving, and optional Gradio interface.
- `data_loader.py` - locates `train.csv`, loads the dataset, checks the required columns, and separates comments from labels.
- `data/README.md` - explains how to obtain and place the dataset.
- `report/` - written project report.


## Dataset

The project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.

Source:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

The training set used by the project contains **159,571 comments**.

Positive label frequencies in the full dataset are:

| Label | Positive examples |
|---|---:|
| toxic | 15,294 |
| obscene | 8,449 |
| insult | 7,877 |
| severe_toxic | 1,595 |
| identity_hate | 1,405 |
| threat | 478 |

This shows a strong class imbalance, especially for `threat` and
`identity_hate`.

### Dataset setup

The dataset is not committed to GitHub.

Download the Jigsaw competition data and extract it so that `train.csv` is
somewhere inside:

```text
data/jigsaw-toxic-comment-classification-challenge/
```

The project searches recursively for `train.csv`, so an additional extraction
subfolder is fine.

## Installation

Python 3 is required.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The required libraries are:

- pandas
- scikit-learn
- joblib
- gradio
- jupyter

## Running the project

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Jigsaw dataset and place it inside the `data/` directory as
   explained above.
4. Open `toxicity_sklearn.ipynb` in VS Code or Jupyter.
5. Select the Python environment containing the required packages.
6. Run all cells from top to bottom.

The notebook performs the following pipeline:

```text
Load and filter data
        ↓
70 / 20 / 10 split
        ↓
TF-IDF vectorization
        ↓
One-vs-Rest Logistic Regression
        ↓
Validation
        ↓
Final test evaluation
        ↓
Qualitative error analysis
        ↓
Save model and vectorizer
        ↓
Optional Gradio interface
```

## Data split

The data is split reproducibly using `random_state=42`:

| Split | Comments |
|---|---:|
| Training | 111,699 |
| Validation | 31,914 |
| Test | 15,958 |

The final test set is kept separate until model and preprocessing choices are
fixed.

## Text representation

Text is represented with `TfidfVectorizer`.

Important settings:

```python
TfidfVectorizer(
    lowercase=True,
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

`ngram_range=(1, 2)` includes both:

- unigrams such as `stupid`
- bigrams such as `not stupid` and `you are`

The vectorizer is fitted only on the training data. The validation and test
sets are transformed using the vocabulary learned from the training set.

## Classifier

The classifier is:

```python
OneVsRestClassifier(
    LogisticRegression(...)
)
```

One-vs-Rest trains one Logistic Regression classifier for each toxicity label.
This is suitable for the multi-label task because several labels can be
predicted for the same comment.

## Evaluation

The project reports:

- precision
- recall
- F1-score
- micro precision
- micro recall
- micro F1
- macro F1
- subset accuracy
- Hamming loss

### Final test results

| Metric | Result |
|---|---:|
| Micro Precision | 0.9026 |
| Micro Recall | 0.5118 |
| Micro F1 | 0.6532 |
| Macro F1 | 0.4530 |
| Subset Accuracy | 0.9211 |
| Hamming Loss | 0.0195 |

Per-label results:

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| toxic | 0.93 | 0.58 | 0.71 |
| severe_toxic | 0.59 | 0.23 | 0.33 |
| obscene | 0.94 | 0.56 | 0.71 |
| threat | 1.00 | 0.11 | 0.20 |
| insult | 0.85 | 0.48 | 0.62 |
| identity_hate | 0.81 | 0.09 | 0.16 |

The classifier has high precision but lower recall. Rare classes such as
`threat` and `identity_hate` perform considerably worse than common labels,
which is consistent with the class imbalance in the dataset.

## Qualitative analysis

The notebook also tests manually written comments such as:

```text
I love you
you are stupid
you are not stupid
not stupid
you are fat
you are not fat
not fat
you are dumb
you are not dumb
```

A major error pattern is **negation**. For example, phrases such as
`you are not stupid` can still receive high toxicity probabilities.

This illustrates a limitation of TF-IDF and short n-gram features: they capture
strong lexical patterns but do not fully represent sentence-level semantics.

## Saving the model

After training, the notebook saves:

```text
models/toxicity_sklearn_model.joblib
models/tfidf_vectorizer.joblib
```

These generated files are excluded from Git by default because the project can
retrain quickly from the original data.

## Gradio interface

The final notebook cell can launch a local Gradio interface. It allows a user
to enter a comment and view a probability and binary prediction for each of the
six labels.

## Limitations

The main limitations observed are:

- strong class imbalance;
- low recall for rare categories;
- limited semantic understanding;
- difficulty with negation;
- one fixed decision threshold (`0.5`) for every label.

Possible future improvements include:

- class weighting;
- label-specific decision thresholds;
- comparison with additional scikit-learn classifiers;
- more detailed preprocessing and feature engineering;
- additional qualitative error categories.

## Report

The complete written report is available in the `report/` folder.

## Reproducibility

The project uses a fixed `random_state=42` for the dataset splits. Dataset
files and generated model files are excluded from Git, while all code and
instructions required to reproduce the results are included.

## References

- Jigsaw / Conversation AI. *Toxic Comment Classification Challenge*. Kaggle.  
  https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*.
  Journal of Machine Learning Research, 12, 2825-2830.  
  https://jmlr.org/papers/v12/pedregosa11a.html
- Scikit-learn documentation.  
  https://scikit-learn.org/
- Gradio documentation.  
  https://www.gradio.app/

## Course

Advanced Python for NLP  
Heinrich-Heine-Universität Düsseldorf
