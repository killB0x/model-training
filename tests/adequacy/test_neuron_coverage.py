import pytest
import pickle
import numpy as np
from keras.models import load_model, Model, Sequential
from keras.layers import Dense, Conv1D
from sklearn.model_selection import train_test_split

@pytest.fixture
def trained_model():
    model = load_model('./models/model.keras')

    yield model

@pytest.fixture
def test_data(sample_percentage=1.0):
    with open('data/processed/x_test.pkl', 'rb') as file:
        x_test = pickle.load(file)
    
        x_test_sampled, _ = train_test_split(x_test, test_size=1 - 0.05, random_state=42)
    # if sample_percentage < 1:
    # else:
    #     x_test_sampled = x_test

    return x_test_sampled

def test_neuron_coverage(trained_model, test_data):
    # intermediate_models = trained_model
    # coverage = neuron_coverage(intermediate_models, test_data)

    trained_model(test_data)

    
    intermediate_model = Sequential()

    activated_neurons = 0
    total_neurons = 0

    # Define a threshold for activation (e.g., 0.5)
    threshold = 0.5

    for layer in trained_model.layers:
        intermediate_model.add(layer)
        if isinstance(layer, (Conv1D, Dense)):
            intermediate_output = intermediate_model.predict(test_data)

            print(intermediate_output.shape)
            print(layer.output.shape)

            activated_neurons += np.sum(np.array(intermediate_output[1:]) > threshold)
            total_neurons += np.prod(layer.output.shape[1:])

    

    # Calculate neuron coverage
    # activated_neurons = sum((np.array(activations) > threshold).sum(axis=0))
    # total_neurons = sum([np.prod(layer.output_shape[1:]) for layer in trained_model.layers])

    coverage = activated_neurons / total_neurons

    print(f'Neuron Coverage: {coverage:.2%}')
    assert coverage > 0, "No neurons are being activated!"

# You may need to adjust pytest to properly call and test this code automatically.