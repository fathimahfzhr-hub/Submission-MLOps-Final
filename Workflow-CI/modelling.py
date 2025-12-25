import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Load Data
train = pd.read_csv('water_potability_train_clean.csv')
test = pd.read_csv('water_potability_test_clean.csv')

X_train = train.drop('Potability', axis=1)
y_train = train['Potability']
X_test = test.drop('Potability', axis=1)
y_test = test['Potability']

# Setup MLflow Autolog (Syarat Basic)
mlflow.autolog()

# Train Model Sederhana
with mlflow.start_run(run_name="Basic_Model"):
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    print("Model Basic berhasil dilatih!")
