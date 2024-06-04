import pytest

from models.create_model import create_model
from models.predict import accuracy

from keras.models import load_model

SAMPLE_PERCENTAGE = 0.1

# ML Infrastructure -> Test the reproducibility of training.
def test_nondeterminism_robustness():
  original_score = accuracy(create_model(42, SAMPLE_PERCENTAGE), 42, SAMPLE_PERCENTAGE)

  for seed in [1, 2]:
    model_variant = create_model(seed, SAMPLE_PERCENTAGE)

    assert abs(original_score - accuracy(model_variant, seed, SAMPLE_PERCENTAGE)) <=0.03

