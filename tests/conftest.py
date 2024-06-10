def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="run tests on full dataset")

def pytest_generate_tests(metafunc):
    if "sample_percentage" in metafunc.fixturenames:
        if metafunc.config.getoption("full"):
            sample_percentage = 1
            print("Running tests on full dataset.")
        else:
            sample_percentage = 0.1
            print("Running tests on full dataset.")
        metafunc.parametrize("sample_percentage", [sample_percentage])