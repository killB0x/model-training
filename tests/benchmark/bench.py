import pytest

from models.create_model import create_model

# Monitoring Tests for ML -> Test for dramatic or slow-leak regressions in training speed, serving latency, throughput, or RAM usage.
def test_training_time(benchmark):
    benchmark(create_model, seed=42, sample_percentage=0.1)