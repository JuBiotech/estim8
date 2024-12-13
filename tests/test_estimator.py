import time
from copy import deepcopy
from typing import Dict, List, get_args

import numpy as np
import pytest

from estim8 import Estimator
from estim8.datatypes import Constants, Experiment, Measurement
from estim8.error_models import LinearErrorModel
from estim8.models import Estim8Model, FmuModel

from .mockmodel import MockModel


@pytest.fixture
def model() -> MockModel:
    return MockModel()


@pytest.fixture
def data_single_replicate(model: MockModel) -> Experiment:
    sim = model.simulate(t0=0, t_end=10, stepsize=1)
    return Experiment(
        measurements=[
            Measurement(
                name="y",
                timepoints=sim["y"].timepoints,
                values=sim["y"].values,
                error_model=LinearErrorModel(slope=1e-5, offset=1e-5),
            )
        ],
    )


@pytest.fixture
def data_multiple_replicates(
    data_single_replicate: Experiment,
) -> Dict[str, Experiment]:
    data = {}
    for rid in map(str, list(range(2))):
        data[rid] = data_single_replicate
        data[rid].replicate_ID = rid
    return data


@pytest.fixture
def bounds() -> Dict[str, List[float]]:
    return {"offset": [-2, 2], "slope": [0, 5]}


@pytest.fixture
def t() -> List[float]:
    return [0, 10, 1]


@pytest.fixture
def estimator_single_replicate(
    model: MockModel,
    data_single_replicate: Experiment,
    bounds: Dict[str, List[float]],
    t: List[float],
) -> Estimator:
    return Estimator(model=model, data=data_single_replicate, bounds=bounds, t=t)


@pytest.fixture
def estimator_multiple_replicates(
    model: MockModel,
    data_multiple_replicates: Dict[str, Experiment],
    bounds: Dict[str, List[float]],
    t: List[float],
) -> Estimator:
    return Estimator(model=model, data=data_multiple_replicates, bounds=bounds, t=t)


# check if estimates are almost equal true parameters
def all_almost_equal(estimates: Dict[str, float], true_parameters) -> bool:
    return all(
        [
            val == pytest.approx(true_parameters[key], rel=0.1, abs=0.01)
            for key, val in estimates.items()
        ]
    )


class TestEstimatorSingleReplicate:
    @pytest.fixture(autouse=True)
    def init_estimator(self, estimator_single_replicate: Estimator) -> None:
        self.estimator = estimator_single_replicate

    @pytest.mark.parametrize("metric", get_args(Constants.VALID_METRICS))
    def test_estimate_single_core(self, metric: Constants.VALID_METRICS) -> None:
        self.estimator.metric = metric
        res, _ = self.estimator.estimate(method="de", max_iter=100, n_jobs=1)
        assert all_almost_equal(res, self.estimator.model.parameters)

    def test_estimate_parallel(self) -> None:
        res, _ = self.estimator.estimate(method="de", max_iter=100, n_jobs=2)
        assert all_almost_equal(res, self.estimator.model.parameters)

    def test_estimate_federated(self) -> None:
        """Test parameter estimation using federated workers"""
        res, _ = self.estimator.estimate(
            method="de", max_iter=100, n_jobs=1, federated_workers=2
        )
        assert all_almost_equal(res, self.estimator.model.parameters)


class TestEstimatorMultiReplicates:
    @pytest.fixture(autouse=True)
    def init_estimator(self, estimator_multiple_replicates: Estimator) -> None:
        self.estimator = estimator_multiple_replicates

    def test_estimate_single_core(self) -> None:
        res, _ = self.estimator.estimate(method="de", max_iter=100, n_jobs=1)
        assert all_almost_equal(res, self.estimator.model.parameters)

    def test_estimate_parallel(self) -> None:
        res, _ = self.estimator.estimate(method="de", max_iter=100, n_jobs=2)
        assert all_almost_equal(res, self.estimator.model.parameters)

    def test_estimate_federated(self) -> None:
        """Test parameter estimation with multiple replicates using federated workers"""

        self.estimator.launch_workers(2, host="127.0.0.1", start_at_port=9000)
        time.sleep(10)
        res, _ = self.estimator.estimate(
            method="de",
            max_iter=100,
            n_jobs=1,
            federated_workers=2,
            # worker_kwargs={"start_at_port": 9000},
        )
        assert all_almost_equal(res, self.estimator.model.parameters)


class TestMonteCarlo:
    @pytest.fixture(autouse=True)
    def init_estimator(self, estimator_single_replicate: Estimator) -> None:
        self.estimator = estimator_single_replicate

    def test_mc_sampling_single_core(self) -> None:
        """Test Monte Carlo sampling with single core"""
        results = self.estimator.mc_sampling(
            method="de", n_jobs=1, max_iter=50, n_samples=2, mcs_at_once=1
        )

        assert len(results) == 2
        for res, info in results:
            assert isinstance(res, dict)
            assert isinstance(info, dict)
            assert all(param in res for param in self.estimator.bounds)
            assert "fun" in info

    def test_mc_sampling_parallel(self) -> None:
        """Test Monte Carlo sampling with parallel processing"""
        results = self.estimator.mc_sampling(
            method="de", n_jobs=1, max_iter=50, n_samples=2, mcs_at_once=2
        )

        assert len(results) == 2
        for res, info in results:
            assert isinstance(res, dict)
            assert isinstance(info, dict)
            assert all(param in res for param in self.estimator.bounds)
            assert "fun" in info

    def test_mc_sampling_federated(self) -> None:
        """Test Monte Carlo sampling with federated workers"""
        results = self.estimator.mc_sampling(
            method="de",
            n_jobs=1,
            max_iter=50,
            n_samples=2,
            mcs_at_once=2,
            federated_workers=2,
            worker_kwargs={"host": "127.0.0.1", "start_at_port": 9100},
        )

        assert len(results) == 2
        for res, info in results:
            assert isinstance(res, dict)
            assert isinstance(info, dict)
            assert all(param in res for param in self.estimator.bounds)
            assert "fun" in info


class TestProfileLikelihood:
    @pytest.fixture(autouse=True)
    def init_estimator(self, estimator_single_replicate: Estimator) -> None:
        self.estimator = estimator_single_replicate
        self.estimator.metric = "negLL"  # Profile likelihood requires negLL metric

    def test_profile_likelihood_single_core(self) -> None:
        """Test profile likelihood calculation with single core"""
        # First get optimal parameters
        p_opt, _ = self.estimator.estimate(method="de", max_iter=50, n_jobs=1)

        # Calculate profile likelihood
        results = self.estimator.profile_likelihood(
            p_opt=p_opt,
            method="de",
            max_iter=50,
            n_jobs=1,
            n_points=3,
            dp_rel=0.1,
            p_inv=["offset"],  # Test only one parameter for speed
        )

        assert isinstance(results, dict)
        assert "offset" in results
        assert len(results["offset"]) == 3
        for point in results["offset"]:
            assert "value" in point
            assert "loss" in point

    def test_profile_likelihood_parallel(self) -> None:
        """Test profile likelihood calculation with parallel processing"""
        p_opt, _ = self.estimator.estimate(method="de", max_iter=50, n_jobs=1)

        results = self.estimator.profile_likelihood(
            p_opt=p_opt,
            method="de",
            max_iter=50,
            n_jobs=2,
            p_at_once=2,
            n_points=3,
            dp_rel=0.1,
            p_inv=["offset", "slope"],
        )

        assert isinstance(results, dict)
        assert len(results) == 2
        for param in ["offset", "slope"]:
            assert param in results
            assert len(results[param]) == 3

    def test_profile_likelihood_federated(self) -> None:
        """Test profile likelihood calculation with federated workers"""
        p_opt, _ = self.estimator.estimate(method="de", max_iter=50, n_jobs=1)

        results = self.estimator.profile_likelihood(
            p_opt=p_opt,
            method="de",
            max_iter=50,
            federated_workers=2,
            n_points=3,
            dp_rel=0.1,
            p_inv=["offset"],
            worker_kwargs={"host": "127.0.0.1", "start_at_port": 9300},
        )

        assert isinstance(results, dict)
        assert "offset" in results
        assert len(results["offset"]) == 3

    def test_profile_likelihood_invalid_inputs(self) -> None:
        """Test profile likelihood with invalid inputs"""
        p_opt, _ = self.estimator.estimate(method="de", max_iter=50, n_jobs=1)

        # Test invalid dp_rel
        with pytest.raises(ValueError):
            self.estimator.profile_likelihood(
                p_opt=p_opt,
                method="de",
                max_iter=50,
                dp_rel=1.5,  # Should be between 0 and 1
            )

        # Test invalid parameter name
        with pytest.raises(KeyError):
            self.estimator.profile_likelihood(
                p_opt=p_opt, method="de", max_iter=50, p_inv=["invalid_param"]
            )

        # Test invalid p_opt
        with pytest.raises(KeyError):
            self.estimator.profile_likelihood(
                p_opt={"invalid_param": 1.0}, method="de", max_iter=50
            )


def test_get_time_from_data(data_single_replicate, bounds, model) -> None:
    estimator = Estimator(model=model, data=data_single_replicate, bounds=bounds)
    assert len(estimator.t) == 3
    assert estimator.t[0] == 0
    assert estimator.t[1] <= 10


def test_raise_wrong_metric(data_single_replicate, bounds, model) -> None:
    with pytest.raises(ValueError):
        Estimator(
            model=model,
            data=data_single_replicate,
            bounds=bounds,
            metric="wrong_metric",
        )
