# URL Phising Detection CNN

This project is a URL Phising Detection using Convolutional Neural Network (CNN) with Python.

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed the latest version of Python and Pip.

## Installing and Using Project Title

To install and set up the environment, follow these steps:

1. Clone the repository:
2. Install the required libraries using pipenv:
```bash
pipenv install
```
3. Run the project:
```dvc pull```
```dvc repro```
4. Check experiments:
```dvc exp show```

To run custom experiment, update ```params.yml``` file and run  ```dvc repro``` and then ```dvc exp show```.


## Code Quality Reports

We use [dslinter](https://github.com/SERG-Delft/dslinter) to detect code smells specific to Machine Learning.
The linter is configured using the [.pylintrc configuration file provided by dslinter](https://github.com/Hynn01/dslinter/blob/main/docs/pylint-configuration-examples/pylintrc-with-only-dslinter-settings/.pylintrc).

You can run the linter as follows:
```bash
pylint <path_to_sources>
```

This will create a code quality report in the `reports` folder
