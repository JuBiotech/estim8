import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from estim8 import Estimator
from estim8.datatypes import Experiment, Measurement, ModelPrediction, Simulation
from estim8.visualization import (
    plot_distributions,
    plot_estimates,
    plot_estimates_many,
    plot_heatmap,
    plot_measurement,
    plot_model_prediction,
    plot_pairs,
    plot_predictives_many,
    plot_profile_likelihood,
    plot_simulation,
)

from . import mockmodel


@pytest.fixture
def sample_data():
    timepoints = np.array([0, 1, 2, 3, 4])
    values = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
    errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    return timepoints, values, errors


@pytest.fixture
def measurement(sample_data):
    timepoints, values, errors = sample_data
    return Measurement(name="", timepoints=timepoints, values=values, errors=errors)


@pytest.fixture
def model_prediction(sample_data):
    timepoints, values, _ = sample_data
    return ModelPrediction(name="", timepoints=timepoints, values=values)


@pytest.fixture
def simulation(sample_data):
    timepoints, values, _ = sample_data
    return Simulation(
        {"time": timepoints, "values": values}, replicate_ID="test_replicate"
    )


@pytest.fixture
def estimator(measurement):
    model = mockmodel.MockModel()
    data = Experiment(measurements=[measurement], observation_mapping={"": "y"})
    bounds = {"base": (0, 10), "offset": (-10, 10)}
    return Estimator(model=model, data=data, bounds=bounds, t=[0, 1, 1])


def test_plot_measurement(measurement):
    fig, ax = plt.subplots()
    plot_measurement(ax, measurement)
    assert isinstance(ax, Axes)
    assert len(ax.lines) > 0  # Check if something was plotted
    plt.close(fig)


def test_plot_model_prediction(model_prediction):
    fig, ax = plt.subplots()
    plot_model_prediction(ax, model_prediction)
    assert isinstance(ax, Axes)
    assert len(ax.lines) > 0
    plt.close(fig)


def test_plot_distributions():
    # Test data
    data = pd.DataFrame(
        {"param1": np.random.normal(0, 1, 100), "param2": np.random.normal(2, 0.5, 100)}
    )

    fig = plot_distributions(data, ci_level=0.95)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # Should have 2 subplots
    plt.close(fig)


def test_plot_heatmap():
    # Test data
    data = pd.DataFrame(
        {"param1": np.random.normal(0, 1, 100), "param2": np.random.normal(2, 0.5, 100)}
    )

    fig = plot_heatmap(data, thresholds=5)
    assert isinstance(fig, Axes)


def test_plot_predictives_many():
    timepoints = np.linspace(0, 10, 100)
    trajectories = np.random.normal(0, 1, (50, 100))  # 50 trajectories

    fig, ax = plt.subplots()
    plot_predictives_many(ax, timepoints, trajectories, color="blue")
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_simulation(simulation, measurement):
    # Test basic simulation plot
    fig = plot_simulation(simulation)
    assert isinstance(fig, Figure)
    assert len(fig.axes) > 0

    # Test with observe parameter
    fig = plot_simulation(simulation, observe=["values"])
    assert isinstance(fig, Figure)

    # Test with experiment parameter
    experiment = Experiment(
        measurements=[measurement], observation_mapping={"": "values"}
    )
    fig = plot_simulation(simulation, experiment=experiment)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_plot_estimates(estimator):
    estimates = {"base": 5.0, "offset": 0.0}

    # Test basic plot
    fig = plot_estimates(estimates=estimates, estimator=estimator)
    assert isinstance(fig, Figure)

    # Test with only_measured=True
    fig = plot_estimates(estimates=estimates, estimator=estimator, only_measured=True)
    assert isinstance(fig, Figure)
    plt.close("all")


def test_plot_estimates_many(estimator):
    # Create test samples data
    samples = pd.DataFrame(
        {"base": np.random.normal(5, 0.1, 10), "offset": np.random.normal(0, 0.1, 10)}
    )

    # Test basic plot
    fig = plot_estimates_many(mc_samples=samples, estimator=estimator)
    assert isinstance(fig, Figure)

    # Test with only_measured=True
    fig = plot_estimates_many(
        mc_samples=samples, estimator=estimator, only_measured=True
    )
    assert isinstance(fig, Figure)
    plt.close("all")


def test_plot_profile_likelihood():
    # Create mock profile likelihood results matching actual estimator output format
    pl_results = {
        "param1": [
            {"value": val, "loss": -(val**2)} for val in np.linspace(-2, 2, 20)
        ],
        "param2": [
            {"value": val, "loss": -((val - 2.5) ** 2)} for val in np.linspace(0, 5, 20)
        ],
    }

    fig = plot_profile_likelihood(pl_results)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # One subplot per parameter
    plt.close(fig)


def test_plot_pairs():
    # Create test multivariate normal distribution with 3 parameters
    samples = pd.DataFrame(
        {
            "param1": np.random.normal(0, 1, 100),
            "param2": np.random.normal(2, 0.5, 100),
            "param3": np.random.normal(-1, 0.3, 100),
        }
    )

    # Test default kde plot
    fig = plot_pairs(samples)
    assert isinstance(fig, seaborn.axisgrid.PairGrid)
    assert len(fig.axes.flat) == 9  # 3x3 grid for 3 parameters

    # Test with scatter plot
    fig = plot_pairs(samples, kind="scatter")
    assert isinstance(fig, seaborn.axisgrid.PairGrid)
    plt.close("all")
