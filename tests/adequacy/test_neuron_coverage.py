import pytest
from keras.models import load_model
import pickle
import numpy as np
import math

from sklearn.model_selection import train_test_split


@pytest.fixture
def trained_model():
  trained_model = load_model('./models/model.keras')
  yield trained_model

@pytest.fixture
def test_data(sample_percentage = 1.0):
    with open('data/processed/x_test.pkl', 'rb') as file:
        x_test = pickle.load(file)
    
    if (sample_percentage < 1):
        x_test_sampled, _ = train_test_split(x_test, test_size=1 - sample_percentage, random_state=42)
    else:
        x_test_sampled = x_test

    return x_test_sampled

def test_neuron_coverage(trained_model, test_data):
    # TODO: Correct computation of number of activated neurons
    layer_activations = trained_model.predict(test_data)
    
    total_neurons = 0
    total_activated_neurons = 0
    
    for layer, activation in zip([layer for layer in trained_model.layers if 'dense' in layer.name or 'conv' in layer.name], layer_activations):
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]

        total_neurons += math.prod([dim for dim in weights.shape]) + biases.shape[0]
        total_activated_neurons += np.sum(activation > 0, axis=0)

    print("Total neuron coverage: ", total_activated_neurons / total_activated_neurons * 100)
    