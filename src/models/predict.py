"""
Obtain performance metrics of a model
"""

import pickle
import json
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

def accuracy(model, seed = 42, sample_percentage = 1):
    """
    Compute accuracy for a given model.
    """
    with open('./data/processed/x_test.pkl', 'rb') as file:
        x_test = pickle.load(file)

    with open('./data/processed/y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)

    x_test_sampled, _, y_test_sampled, _ = train_test_split(x_test, y_test, test_size=1 - sample_percentage, stratify=y_test, random_state=seed)

    _, model_accuracy = model.evaluate(x_test_sampled, y_test_sampled)

    return model_accuracy

def predict(model):
    """
    Model prediction.
    """
    with open('./data/processed/x_test.pkl', 'rb') as file:
        x_test = pickle.load(file)

    with open('./data/processed/y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)

    y_pred = model.predict(x_test, batch_size=1000)

    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = np.array(y_test).reshape(-1, 1)

    # Metrics calculations
    report = classification_report(y_test, y_pred_binary, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    model_accuracy = accuracy_score(y_test, y_pred_binary)

    # Organize metrics in a dictionary
    metrics = {
        "classification_report": {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "support": report["1"]["support"]
        },
        "confusion_matrix": confusion_mat.tolist(),  # Convert numpy array to list
        "accuracy": model_accuracy
    }

    return metrics


if __name__ == "__main__":
    trained_model = load_model('./models/model.keras')

    model_metrics = predict(trained_model)

    with open('./data/prediction/prediction.json', 'w', encoding='utf-8') as prediction_file:
        json.dump(model_metrics, prediction_file, indent=4)
