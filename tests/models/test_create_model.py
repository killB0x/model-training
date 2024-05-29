import pytest

from models.create_model import create_model
from models.predict import accuracy

from keras.models import load_model


@pytest.fixture
def trained_model():
  trained_model = load_model('./models/model.keras')
  yield trained_model

# ML Infrastructure -> Test the reproducibility of training.
# def test_nondeterminism_robustness(trained_model):
#   original_score = accuracy(trained_model) # score between 0..100

#   for seed in [1]:
#     model_variant = create_model(seed)

#     assert abs(original_score - accuracy(model_variant)) <=0.03

