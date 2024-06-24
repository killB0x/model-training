import pytest
import pickle
import numpy as np

from keras.layers import Conv2D, Dense
from keras.models import load_model, Model
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


@pytest.fixture
def layer_outputs(trained_model, test_data):
    # Extract the outputs of specific layers
    layer_outputs = [layer.output for layer in trained_model.layers if isinstance(layer, (Conv2D, Dense))]
    if not layer_outputs:
        raise ValueError("No suitable layer outputs found. Check layer names and types.")

    activation_model = Model(inputs=trained_model.input, outputs=layer_outputs)
    activations = activation_model.predict(test_data)
    return activations


def test_neuron_coverage(layer_outputs):
    # Compute the ratio of activated neurons
    threshold = 0.3
    total_neurons = 0
    activated_neurons = 0

    for layer_activations in layer_outputs:
        total_neurons += np.prod(layer_activations.shape[1:])  # Exclude the batch size dimension
        activated_neurons += np.sum(layer_activations > threshold)

    neuron_coverage = activated_neurons / total_neurons * 100 if total_neurons > 0 else 0
    print("Total neuron coverage: ", neuron_coverage)
    return neuron_coverage
    