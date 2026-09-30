"""Utilities for loading and filtering the Jigsaw toxicity dataset."""

import os
import pandas as pd

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def find_train_csv(data_folder):
    """Return the first train.csv found recursively inside data_folder."""
    candidates = []

    for root, _, files in os.walk(data_folder):
        if "train.csv" in files:
            candidates.append(os.path.join(root, "train.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find train.csv anywhere inside: {data_folder}"
        )

    return candidates[0]


def load_jigsaw_data(data_folder):
    """Load Jigsaw train.csv and return dataframe, comments, labels, and path."""
    data_path = find_train_csv(data_folder)
    df = pd.read_csv(data_path)

    required_columns = ["comment_text", *LABELS]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df["comment_text"].fillna("")
    y = df[LABELS]

    return df, X, y, data_path
