import pickle

import numpy as np
import pytest
from pathlib import Path
from estim8.datatypes import Simulation
from estim8.models import Estim8Model, FmuModel, fmpy


def checObsInSim(obs: str, sim: Simulation):
    return any([obs in s.name for s in sim.model_predictions])


class DummyModel(Estim8Model):
    def retrieve_variables(self):
        self.parameters = {"param1": 1.0, "param2": 2.0}
        self.observables = ["obs1", "obs2"]

    def simulate(self, t0, t_end, stepsize, parameters, observe, replicate_ID=None):
        time = np.arange(t0, t_end, stepsize)
        sim_data = {obs: np.sin(time) for obs in observe}
        sim_data["time"] = time
        return Simulation(sim_data, replicate_ID=replicate_ID)


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_retrieve_variables(dummy_model):
    assert dummy_model.parameters == {"param1": 1.0, "param2": 2.0}
    assert dummy_model.observables == ["obs1", "obs2"]


def test_simulate(dummy_model):
    sim = dummy_model.simulate(0, 10, 1, {}, observe=["obs1", "obs2"])
    assert isinstance(sim, Simulation)
    assert checObsInSim("obs1", sim)
    assert checObsInSim("obs2", sim)


class TestFmuModel:
    test_fmu_path = Path(__file__).absolute().parent / "test_data/growth.fmu"

    def test_retrieve_variables(self):
        fmu_model = FmuModel(self.test_fmu_path)
        assert "X" in fmu_model.observables
        assert all([param in fmu_model.parameters for param in ["X0", "mu_max"]])
        del fmu_model

    @pytest.mark.parametrize("fmi_type", ["ModelExchange", "CoSimulation", "Error"])
    def test_FMI_setter_and_instantiateFMU(self, fmi_type):
        fmu_model = FmuModel(self.test_fmu_path)
        if fmi_type in ["ModelExchange", "CoSimulation"]:
            fmu_model.fmi_type = fmi_type
            assert fmu_model.fmi_type == fmi_type
        else:
            with pytest.raises(ValueError):
                fmu_model.fmi_type = fmi_type
        del fmu_model

    @pytest.mark.parametrize("fmi_type", ["ModelExchange", "CoSimulation"])
    def test_simulation(self, fmi_type):
        fmu_model = FmuModel(self.test_fmu_path)
        fmu_model.fmi_type = fmi_type
        assert fmu_model.fmi_type == fmi_type
        print(fmu_model.observables)
        sim = fmu_model.simulate(0, 10, 1)
        assert isinstance(sim, Simulation)
        assert checObsInSim("X", sim)
        del fmu_model

    def test_pickle(self):
        fmu_model = FmuModel(self.test_fmu_path)
        pickle.dumps(fmu_model)
