"""
Model prediction.
"""

import pickle
from keras.models import load_model
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def predict():
    model = load_model('./models/model.keras')
    
    with open('./data/processed/x_test.pkl', 'rb') as f:
        x_test = pickle.load(f)
    
    with open('./data/processed/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
        
    y_pred = model.predict(x_test, batch_size=1000)
    
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = np.array(y_test).reshape(-1, 1)

    # Metrics calculations
    report = classification_report(y_test, y_pred_binary, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)

    # Organize metrics in a dictionary
    metrics = {
        "classification_report": {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "support": report["1"]["support"]
        },
        "confusion_matrix": confusion_mat.tolist(),  # Convert numpy array to list
        "accuracy": accuracy
    }

    with open('./data/prediction/prediction.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
if __name__ == "__main__":
    predict()
