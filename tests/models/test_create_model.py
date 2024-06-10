import pytest

from models.create_model import create_model
from models.predict import accuracy

# ML Infrastructure -> Test the reproducibility of training.
def test_nondeterminism_robustness(sample_percentage):
  original_score = accuracy(create_model(42, sample_percentage), 42, sample_percentage)

  for seed in [1, 2]:
    model_variant = create_model(seed, sample_percentage)
    threshold = 0.03 / sample_percentage # be more lenient with less data

    assert abs(original_score - accuracy(model_variant, seed, sample_percentage)) <= threshold

