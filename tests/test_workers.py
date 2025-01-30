import multiprocessing
import platform
import time
from dataclasses import dataclass

import numpy as np
import pytensor_federated
import pytest

from estim8.datatypes import Experiment, Measurement
from estim8.workers import (
    FederatedWorker,
    Worker,
    init_logging,
    run_worker_pool,
    run_worker_service,
)

from .mockmodel import MockModel


@dataclass
class TestExperiment(Experiment):
    def __init__(self):
        t = np.array([0, 1, 2, 3])
        data = t + 1
        errors = np.ones_like(t) * 0.1
        times = t
        super().__init__(
            measurements=[
                Measurement(name="y", values=data, errors=errors, timepoints=times)
            ]
        )

    def calculate_loss(self, simulation, metric="SS"):
        return 1.0  # Simple loss value for testing


class TestEstimator:
    def __init__(self):
        self.model = MockModel()
        self.replicate_IDs = ["r1", "r2"]
        self.data = {"r1": TestExperiment(), "r2": TestExperiment()}
        self.metric = "SS"
        self.t = [0, 10, 0.1]

    def objective_for_replicate(self, parameters, data, metric):
        return 1.0


class TestRaiseErrorEstimator(TestEstimator):
    def objective_for_replicate(self, parameters, data, metric):
        raise ValueError("Error in objective_for_replicate")


@pytest.fixture
def test_estimator():
    return TestEstimator()


def test_worker_init(test_estimator):
    worker = Worker(test_estimator)
    assert worker.estimator == test_estimator
    assert worker.mc_sampling is False

    worker = Worker(test_estimator, mc_sampling=True)
    assert worker.mc_sampling is True


def test_worker_call(test_estimator):
    worker = Worker(test_estimator)
    theta = np.array([1.0, 2.0, 0])  # Parameters + replicate ID
    result = worker(theta)
    assert isinstance(result, np.ndarray)
    assert result == np.array(1.0)


def test_federated_worker_error_handling():
    worker = FederatedWorker(TestRaiseErrorEstimator())
    # Force an error by passing invalid parameters
    theta = np.array([np.nan, np.nan, 0])  # Invalid parameters to trigger error
    result = worker(theta)
    assert isinstance(result, np.ndarray)
    assert result == np.array(np.inf)


def test_run_worker_service(test_estimator):
    if platform.system() == "Windows":
        host = "localhost"
    else:
        host = "127.0.0.1"
    port = 9500
    p = multiprocessing.Process(
        target=run_worker_service, args=(host, port, test_estimator)
    )
    try:
        p.start()
        time.sleep(5)
        client = pytensor_federated.common.LogpServiceClient(host, port)
        res = client.evaluate([0])
        np.testing.assert_almost_equal(
            test_estimator.objective_for_replicate(None, None, None), res
        )
    finally:
        p.terminate()
        p.join()
    pass


def test_run_worker_pool(test_estimator):
    if platform.system() == "Windows":
        host = "localhost"
    else:
        host = "127.0.0.1"
    ports = [9502, 9503]
    processes = run_worker_pool(host, ports, test_estimator)

    try:
        assert len(processes) == 2
        assert all(isinstance(p, multiprocessing.Process) for p in processes)
    finally:
        # Clean up processes
        for p in processes:
            p.terminate()
            p.join()
    pass


def test_init_logging():
    # Test passes if no exceptions are raised
    init_logging()
    pass
