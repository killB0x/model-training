# URL Phishing Detection CNN

This project is a URL Phishing Detection using a Convolutional Neural Network (CNN) with Python.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed the latest version of Python and Pip.

## Installing and Using Project Title

To install and set up the environment, follow these steps:

1. Clone the repository:
2. Make sure you have Poetry installed:
```bash
pip install poetry
```
3. Install the required libraries using Poetry:
```bash
poetry install
poetry shell
```
4. Run the project:
```dvc pull```
```dvc repro```
5. Check experiments:
```dvc exp show```

To run a custom experiment, update ```params.yml``` and run ```dvc repro``` followed by ```dvc exp show```.

## Testing
We use [PyTest](https://docs.pytest.org/en/8.2.x/) for testing. 

You can run the tests as follows:
```bash
poetry run pytest tests/ --ignore=benchmark
```

### Benchmarking
We use [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/stable/) for benchmarking.

You can run the benchmarks as follows:
```bash
poetry run pytest tests/benchmark/bench.py --benchmark-json output.json
```
This will generate an `output.json` file with the benchmark results.

*Note: by default, all tests will be run on a 10% random sample of the dataset. To run on the full dataset, add the `--full` flag to either of the commands listed above.*

## Code Quality

We use [dslinter](https://github.com/SERG-Delft/dslinter) to detect code smells specific to Machine Learning.
The linter is configured using the [.pylintrc configuration file provided by dslinter](https://github.com/Hynn01/dslinter/blob/main/docs/pylint-configuration-examples/pylintrc-with-only-dslinter-settings/.pylintrc).

You can run the Pylint linter (which is configured using dslinter) as follows:
```bash
poetry run pylint src
```