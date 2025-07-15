import numpy as np
import pandas as pd
import pygmo
import pytest

from estim8.generalized_islands import (
    Estim8_mp_island,
    PygmoEstimationInfo,
    PygmoHelpers,
    UDproblem,
)
from estim8.objective import Objective


def dummy_objective(x):
    return np.sum(x**2)


@pytest.fixture
def bounds():
    return {"x1": (-5, 5), "x2": (-5, 5)}


@pytest.fixture
def ud_problem(bounds):
    return UDproblem(dummy_objective, bounds)


@pytest.fixture
def archi(bounds):
    algos = ["de", "pso"]
    algos_kwargs = [{}, {}]
    archi, info = PygmoHelpers.create_archipelago(
        objective=dummy_objective,
        bounds=bounds,
        algos=algos,
        algos_kwargs=algos_kwargs,
        pop_size=10,
        report=False,
        n_processes=1,
    )
    return archi


@pytest.fixture
def archi_with_trace(bounds):
    algos = ["de", "pso"]
    algos_kwargs = [{}, {}]
    archi, info = PygmoHelpers.create_archipelago(
        objective=dummy_objective,
        bounds=bounds,
        algos=algos,
        algos_kwargs=algos_kwargs,
        pop_size=10,
        report=True,  # Enable reporting for trace
        n_processes=1,
    )
    return archi, info


def test_ud_problem_fitness(ud_problem):
    theta = np.array([1, 2])
    fitness = ud_problem.fitness(theta)
    assert fitness == np.array([5])


def test_ud_problem_bounds(ud_problem):
    lower_bounds, upper_bounds = ud_problem.get_bounds()
    assert np.array_equal(lower_bounds, np.array([-5, -5]))
    assert np.array_equal(upper_bounds, np.array([5, 5]))


def test_create_archipelago(archi):
    assert isinstance(archi, pygmo.archipelago)
    assert len(archi) == 2


def test_get_estimates_from_archipelago(archi):
    estimates, loss = PygmoHelpers.get_estimates_from_archipelago(archi)
    assert isinstance(estimates, dict)
    assert isinstance(loss, float)


def test_get_archipelago_results(archi):
    estimation_info = PygmoEstimationInfo(archi)
    estimates, updated_info = PygmoHelpers.get_archipelago_results(
        archi, estimation_info
    )
    assert isinstance(estimates, dict)
    assert isinstance(updated_info, PygmoEstimationInfo)
    assert updated_info.fun < np.inf


def test_get_archi_f_evals(archi):
    estimation_info = PygmoEstimationInfo(archi)
    assert estimation_info.get_f_evals() == 20


def test_str_archi(archi):
    estimation_info = PygmoEstimationInfo(archi)
    assert "fun" in str(estimation_info)
    assert "n_evos" in str(estimation_info)


@pytest.mark.parametrize("algo", PygmoHelpers.algo_default_kwargs.keys())
def test_get_pygo_algo_instance(algo):
    PygmoHelpers.get_pygmo_algorithm_instance(algo)


def test_raise_wrong_algo():
    with pytest.raises(ValueError):
        PygmoHelpers.get_pygmo_algorithm_instance("wrong_algo")


def test_estim8_mp_island_creation():
    island = Estim8_mp_island()
    assert island.evo_count == 0
    assert isinstance(island.evo_trace, pd.DataFrame)
    assert len(island.evo_trace) == 0
    assert isinstance(island.evo_trace.index, pd.MultiIndex)
    assert island.evo_trace.index.names == ["evolution", "algorithm"]


def test_estim8_mp_island_copy():
    island = Estim8_mp_island()
    island.evo_count = 5

    # Create new data with MultiIndex
    new_data = pd.DataFrame(
        {
            "island_id": [id(island)],
            "champion_loss": [0.5],
            "champion_theta": [[1.0, 2.0]],
        },
        index=pd.MultiIndex.from_arrays(
            [[1], ["test_algo"]], names=["evolution", "algorithm"]
        ),
    )
    island.evo_trace = pd.concat([island.evo_trace, new_data])

    copied_island = island.__copy__()
    assert copied_island.evo_count == 5
    assert len(copied_island.evo_trace) == 1
    assert not copied_island.evo_trace.empty
    assert isinstance(copied_island.evo_trace.index, pd.MultiIndex)


def test_evolution_trace(archi_with_trace):
    archi, info = archi_with_trace

    archi.evolve(1)
    archi.wait_check()
    # Get results which should populate the trace
    _, updated_info = PygmoHelpers.get_archipelago_results(archi, info)

    # Check trace exists and has correct structure
    assert updated_info.evo_trace is not None
    assert isinstance(updated_info.evo_trace, pd.DataFrame)
    assert len(updated_info.evo_trace) > 0

    # Verify index structure
    assert isinstance(updated_info.evo_trace.index, pd.MultiIndex)
    assert updated_info.evo_trace.index.names == ["evolution", "algorithm"]

    # Verify columns
    expected_columns = ["island_id", "champion_loss", "champion_theta"]
    assert all(col in updated_info.evo_trace.columns for col in expected_columns)

    # Check data types
    assert updated_info.evo_trace.index.get_level_values("evolution").dtype == "int64"
    assert updated_info.evo_trace["champion_loss"].dtype == "float64"
