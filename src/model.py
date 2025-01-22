import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import load_npz
import pandas as pd


X = load_npz("../data/processed/tfidf_matrix.npz")
y = pd.read_csv("../data/processed/labels.csv")['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Upload the model
loaded_model = joblib.load("../models/best_model.pkl")

# Test the model
y_pred_loaded = loaded_model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred_loaded))
