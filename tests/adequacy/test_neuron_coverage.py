import math
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
def test_dataset():
    def make_dataset(sample_percentage):
        with open('data/processed/x_test.pkl', 'rb') as file:
            x_test = pickle.load(file)
        
        if sample_percentage < 1:
            x_test_sampled, _ = train_test_split(x_test, test_size=1 - sample_percentage, random_state=42)
        else:
            x_test_sampled = x_test

        return x_test_sampled
    
    return make_dataset

def test_neuron_coverage(trained_model, test_dataset, sample_percentage):
    print(f'Running tests on {sample_percentage:.0%} of the dataset.')

    test_data = test_dataset(sample_percentage)

    trained_model(test_data)
    
    intermediate_model = Sequential()

    activated_neurons = 0
    total_neurons = 0

    # Define a threshold for activation
    threshold = 0.5

    for layer in trained_model.layers:
        intermediate_model.add(layer)
        if isinstance(layer, (Conv1D, Dense)):
            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]

            total_neurons += math.prod([dim for dim in weights.shape]) + biases.shape[0]

            intermediate_output = intermediate_model.predict(test_data)

            # scale output to [0, 1]
            intermediate_output = (intermediate_output - intermediate_output.min()) / (intermediate_output.max() - intermediate_output.min())

            above_threshold = intermediate_output[1:] > threshold
            # count true values
            activated_neurons += above_threshold.sum()
            all_output = intermediate_output[1:]
            total_neurons += np.prod(all_output.shape)

    coverage = activated_neurons / total_neurons

    print(f'Neuron Coverage: {coverage:.2%}')
    assert coverage > 0, "No neurons are being activated!"