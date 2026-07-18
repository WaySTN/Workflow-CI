"""
modelling.py - Training Model Machine Learning dengan MLflow Autolog
Proyek Akhir MSML - Wahyu Setiawan
(Versi untuk MLflow Project / Workflow CI)
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

def load_preprocessed_data(data_dir="stroke_preprocessing"):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    print(f"X_train shape: {X_train.shape}")
    
    mlflow.sklearn.autolog()
    mlflow.set_experiment("Stroke_Risk_Prediction")
    
    with mlflow.start_run(run_name="GradientBoosting_CI"):
        model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n=== Hasil Evaluasi Model CI ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print("\nModel berhasil dilatih melalui CI Pipeline!")

if __name__ == "__main__":
    train_model()
