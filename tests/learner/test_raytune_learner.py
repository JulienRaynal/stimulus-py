import pytest
import torch.nn as nn
import ray

from src.stimulus.data.experiments import DnaToFloatExperiment
from src.stimulus.learner.raytune_learner import TuneWrapper, TuneModel
from tests.test_model.dnatofloat_model import ModelSimple

@pytest.fixture()
def raytune_learner():
    return TuneWrapper(
        config_path = "tests/test_model/dnatofloat_model_cpu.yaml",
        model_class = ModelSimple,
        experiment_object = DnaToFloatExperiment(),
        data_path = "tests/test_data/dna_experiment/test_with_split.csv",
        max_cpus=2,
        max_gpus=0,
    )

@pytest.fixture(autouse=True)
def shutdown_ray():
    yield
    ray.shutdown()

def test_ray_tuner_initialization(raytune_learner):
    assert ray.is_initialized()

def test_ray_init_without_learner():
    ray.init()
    print(ray.cluster_resources())
    assert ray.is_initialized()

def test_tuning(raytune_learner):
    raytune_learner.tune()
