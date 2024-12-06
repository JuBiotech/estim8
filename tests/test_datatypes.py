from typing import get_args

import numpy as np
import pandas as pd
import pytest

from estim8.datatypes import (
    Constants,
    Experiment,
    Measurement,
    ModelPrediction,
    Simulation,
    TimeSeries,
)
from estim8.error_models import LinearErrorModel


@pytest.fixture
def sample_data():
    timepoints = np.array([0, 1, 2, 3, 4])
    values = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    return timepoints, values


class TestTimeSeries:
    def test_init(self, sample_data):
        timepoints, values = sample_data
        ts = TimeSeries("test", timepoints, values)
        assert ts.name == "test"
        assert np.array_equal(ts.timepoints, timepoints)
        assert np.array_equal(ts.values, values)
        assert ts.replicate_ID == Constants.SINGLE_ID

    def test_equal_shapes(self, sample_data):
        timepoints, values = sample_data
        with pytest.raises(ValueError):
            TimeSeries("test", timepoints, values[:-1])

    def test_drop_nans(self):
        timepoints = np.array([0, 1, 2, 3, 4])
        values = np.array([1.0, np.nan, 0.8, 1.1, np.nan])
        ts = TimeSeries("test", timepoints, values)
        assert len(ts.timepoints) == 3
        assert len(ts.values) == 3


class TestMeasurement:
    def test_init(self, sample_data):
        timepoints, values = sample_data
        errors = np.ones_like(values) * 0.1
        measurement = Measurement("test", timepoints, values, errors=errors)
        assert np.array_equal(measurement.errors, errors)
        assert isinstance(measurement.error_model, LinearErrorModel)

    @pytest.mark.parametrize("metric", get_args(Constants.VALID_METRICS))
    def test_get_loss(self, sample_data, metric):
        timepoints, values = sample_data
        measurement = Measurement("test", timepoints, values)
        prediction = ModelPrediction("test", timepoints, values)

        loss = measurement.get_loss(prediction, metric=metric)
        assert isinstance(loss, float)

    def test_get_sampling(self, sample_data):
        timepoints, values = sample_data
        measurement = Measurement("test", timepoints, values)
        samples = measurement.get_sampling(n_samples=5)
        assert len(samples) == 5
        assert all(isinstance(s, Measurement) for s in samples)

    def test_getLoss_wrongMetric(self, sample_data):
        timepoints, values = sample_data
        measurement = Measurement("test", timepoints, values)
        prediction = ModelPrediction("test", timepoints, values)
        with pytest.raises(NotImplementedError):
            measurement.get_loss(prediction, metric="wrong_metric")


class TestExperiment:
    def test_init_with_dataframe(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=[0, 1, 2])
        exp = Experiment(df)
        assert len(exp.measurements) == 2
        assert all(isinstance(m, Measurement) for m in exp.measurements)

    def test_init_with_measurements(self, sample_data):
        timepoints, values = sample_data
        measurements = [
            Measurement("m1", timepoints, values),
            Measurement("m2", timepoints, values),
        ]
        exp = Experiment(measurements)
        assert len(exp.measurements) == 2

    def test_getitem(self, sample_data):
        timepoints, values = sample_data
        measurements = [
            Measurement("m1", timepoints, values),
            Measurement("m2", timepoints, values),
        ]
        exp = Experiment(measurements)
        assert isinstance(exp["m1"], Measurement)
        with pytest.raises(KeyError):
            exp["nonexistent"]

    def test_generate_mc_samples(self, sample_data):
        timepoints, values = sample_data
        measurements = [Measurement("m1", timepoints, values)]
        exp = Experiment(measurements)
        samples = exp.generate_mc_samples(n_samples=3)
        assert len(samples) == 3
        assert all(isinstance(s, Experiment) for s in samples)

    def test_unequalDFshapes(self, sample_data):
        timepoints, values = sample_data
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=[0, 1, 2])

        errors = pd.DataFrame({"A": [1, 2], "B": [4, 5]}, index=[0, 1])

        with pytest.raises(ValueError):
            Experiment(df, errors=errors)

    def test_wrong_measurement_input(self):
        data = [1, 1]
        with pytest.raises(ValueError):
            Experiment(data)


class TestSimulation:
    data = {
        "time": np.array([0, 1, 2]),
        "var1": np.array([1, 2, 3]),
        "var2": np.array([4, 5, 6]),
    }

    def test_init(self):
        sim = Simulation(self.data)
        assert list([mp.name for mp in sim.model_predictions]) == ["var1", "var2"]
        assert sim.replicate_ID == Constants.SINGLE_ID

    def test_get_prediction(self):
        sim = Simulation(self.data)
        pred = sim["var1"]
        assert isinstance(pred, ModelPrediction)
        assert np.array_equal(pred.values, np.array([1, 2, 3]))

    def test_wrongKey(self):
        sim = Simulation(self.data)
        with pytest.raises(KeyError):
            sim["nonexistent"]
