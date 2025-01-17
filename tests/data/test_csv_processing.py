import json
import os
from typing import Any

import numpy.testing as npt
import pytest

from src.stimulus.data.csv import CsvProcessing
from src.stimulus.data.experiments import DnaToFloatExperiment, ProtDnaToFloatExperiment


class DataCsvProcessing:
    """It stores the CsvProcessing objects initialized on a given csv data and the expected values.

    One can use this class to create the data fixtures.

    Args:
        filename (str): The path to the CSV file.
        experiment (type): The class type of the experiment to be instantiated.

    Attributes:
        experiment (Experiment): An instance of the experiment class.
        csv_path (str): The absolute path to the CSV file.
        csv_processing (CsvProcessing): An instance of the CsvProcessing class for handling CSV data.
        data_length (int or None): The length of the data. Initialized to None.
        expected_split (List[int] or None): The expected split values after adding split. Initialized to None.
        expected_transformed_values (Any or None): The expected values after split and transformation. Initialized to None.
    """

    def __init__(self, filename: str, experiment: Any):
        self.experiment = experiment()
        self.csv_path = os.path.abspath(filename)
        self.csv_processing = CsvProcessing(self.experiment, self.csv_path)
        self.data_length = None
        self.expected_split = None
        self.expected_transformed_values = None


@pytest.fixture
def dna_test_data():
    """This stores the basic dna test csv"""
    data = DataCsvProcessing("tests/test_data/dna_experiment/test.csv", DnaToFloatExperiment)
    data.data_length = 2
    data.expected_split = [1, 0]
    data.expected_transformed_values = {
        "pet:meta:str": ["cat", "dog", "cat", "dog"],
        "hola:label:float": [12.676405, 12.540016, 12.676405, 12.540016],
        "hello:input:dna": ["ACTGACTGATCGATNN", "ACTGACTGATCGATNN", "NNATCGATCAGTCAGT", "NNATCGATCAGTCAGT"],
        "split:split:int": [1, 0, 1, 0],
    }
    return data


@pytest.fixture
def dna_test_data_long():
    """This stores the long dna test csv"""
    data = DataCsvProcessing("tests/test_data/dna_experiment/test_shuffling_long.csv", DnaToFloatExperiment)
    data.data_length = 1000
    return data


@pytest.fixture
def dna_test_data_long_shuffled():
    """This stores the shuffled long dna test csv"""
    data = DataCsvProcessing(
        "tests/test_data/dna_experiment/test_shuffling_long_shuffled.csv",
        ProtDnaToFloatExperiment,
    )
    data.data_length = 1000
    return data


@pytest.fixture
def dna_config():
    """This is the config file for the dna experiment"""
    with open("tests/test_data/dna_experiment/test_config.json") as f:
        return json.load(f)


@pytest.fixture
def prot_dna_test_data():
    """This stores the basic prot-dna test csv"""
    data = DataCsvProcessing("tests/test_data/prot_dna_experiment/test.csv", ProtDnaToFloatExperiment)
    data.data_length = 2
    data.expected_split = [1, 0]
    data.expected_transformed_values = {
        "pet:meta:str": ["cat", "dog", "cat", "dog"],
        "hola:label:float": [12.676405, 12.540016, 12.676405, 12.540016],
        "hello:input:dna": ["ACTGACTGATCGATNN", "ACTGACTGATCGATNN", "NNATCGATCAGTCAGT", "NNATCGATCAGTCAGT"],
        "split:split:int": [1, 0, 1, 0],
        "bonjour:input:prot": ["GPRTTIKAKQLETLX", "GPRTTIKAKQLETLX", "GPRTTIKAKQLETLX", "GPRTTIKAKQLETLX"],
    }
    return data


@pytest.fixture
def prot_dna_config():
    """This is the config file for the prot experiment"""
    with open("tests/test_data/prot_dna_experiment/test_config.json") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "fixture_name",
    [
        ("dna_test_data"),
        ("dna_test_data_long"),
        ("dna_test_data_long_shuffled"),
        ("prot_dna_test_data"),
    ],
)
def test_data_length(request, fixture_name):
    """Test that data is loaded with the correct length.

    Args:
        request: Pytest fixture request object.
        fixture_name (str): Name of the fixture to test.
            Can be one of: dna_test_data, dna_test_data_long,
            dna_test_data_long_shuffled, or prot_dna_test_data.
    """
    data = request.getfixturevalue(fixture_name)
    assert len(data.csv_processing.data) == data.data_length


@pytest.mark.parametrize(
    "fixture_data_name,fixture_config_name",
    [
        ("dna_test_data", "dna_config"),
        ("prot_dna_test_data", "prot_dna_config"),
    ],
)
def test_add_split(request, fixture_data_name, fixture_config_name):
    """Test that the add_split function properly adds the split column.

    Args:
        request: Pytest fixture request object.
        fixture_data_name (str): Name of the data fixture to test.
            Can be either dna_test_data or prot_dna_test_data.
        fixture_config_name (str): Name of the config fixture to use.
            Can be either dna_config or prot_dna_config.
    """
    data = request.getfixturevalue(fixture_data_name)
    config = request.getfixturevalue(fixture_config_name)

    data.csv_processing.add_split(config["split"])
    assert data.csv_processing.data["split:split:int"].to_list() == data.expected_split


@pytest.mark.parametrize(
    "fixture_data_name,fixture_config_name",
    [
        ("dna_test_data", "dna_config"),
        ("prot_dna_test_data", "prot_dna_config"),
    ],
)
def test_transform_data(request, fixture_data_name, fixture_config_name):
    """Test that transformation functionalities properly transform the data.

    Args:
        request: Pytest fixture request object.
        fixture_data_name (str): Name of the data fixture to test.
            Can be either dna_test_data or prot_dna_test_data.
        fixture_config_name (str): Name of the config fixture to use.
            Can be either dna_config or prot_dna_config.
    """
    data = request.getfixturevalue(fixture_data_name)
    config = request.getfixturevalue(fixture_config_name)

    data.csv_processing.add_split(config["split"])
    data.csv_processing.transform(config["transform"])

    for key, expected_values in data.expected_transformed_values.items():
        observed_values = list(data.csv_processing.data[key])
        observed_values = [round(v, 6) if isinstance(v, float) else v for v in observed_values]
        assert observed_values == expected_values


def test_shuffle_labels(dna_test_data_long, dna_test_data_long_shuffled):
    """Test that shuffling of labels works correctly.

    This test verifies that when labels are shuffled with a fixed seed,
    they match the expected shuffled values from a pre-computed dataset.
    Currently only tests the long DNA test data.

    Args:
        dna_test_data_long: Fixture containing the original unshuffled DNA test data.
        dna_test_data_long_shuffled: Fixture containing the expected shuffled DNA test data.
    """
    dna_test_data_long.csv_processing.shuffle_labels(seed=42)
    npt.assert_array_equal(
        dna_test_data_long.csv_processing.data["hola:label:float"],
        dna_test_data_long_shuffled.csv_processing.data["hola:label:float"],
    )
