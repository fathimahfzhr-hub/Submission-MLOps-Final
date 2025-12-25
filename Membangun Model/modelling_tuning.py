import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- KONFIGURASI DAGSHUB ---
REPO_OWNER = "fathimahfzhr-hub"
REPO_NAME = "Submission-MLOps-Fathim"

def load_data():
    # Pastikan data bersih sudah ada
    try:
        train = pd.read_csv('water_potability_train_clean.csv')
        test = pd.read_csv('water_potability_test_clean.csv')
    except FileNotFoundError:
        print("❌ Error: File CSV tidak ditemukan. Pastikan file ada di folder Colab.")
        return None, None, None, None

    X_train = train.drop('Potability', axis=1)
    y_train = train['Potability']
    X_test = test.drop('Potability', axis=1)
    y_test = test['Potability']

    return X_train, y_train, X_test, y_test

def main():
    print("🚀 Memulai proses Training & Tuning...")
    
    # 1. Hubungkan ke DagsHub
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Eksperimen_Kriteria_2_Fathim")

    X_train, y_train, X_test, y_test = load_data()
    if X_train is None: return

    # 2. Hyperparameter Tuning (Syarat Skilled)
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    print("⚙️ Sedang mencari model terbaik (GridSearch)...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"✅ Model Terbaik ditemukan: {best_params}")

    # 3. Logging Manual ke MLflow (Syarat Advanced)
    with mlflow.start_run(run_name="Best_Model_Tuning"):
        
        # A. Log Parameter Terbaik
        mlflow.log_params(best_params)

        # B. Log Metrics
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # C. Log Model
        mlflow.sklearn.log_model(best_model, "model")

        # D. Log Artefak Gambar (Syarat Advanced: Minimal 2 Gambar)
        
        # Gambar 1: Confusion Matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # Gambar 2: Feature Importance
        plt.figure(figsize=(8,6))
        pd.Series(best_model.feature_importances_, index=X_train.columns).nlargest(10).plot(kind='barh')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        print("🎉 Selesai! File modelling_tuning.py sudah valid untuk submission.")

if __name__ == "__main__":
    main()
