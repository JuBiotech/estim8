from functools import partial

import numpy as np
import pytest

from estim8.objective import global_objective
from estim8.optimizers import Objective, Optimization
from estim8.utils import EstimatorHelpers, ModelHelpers

SINGLE_ID = None


def rosen(theta: np.ndarray) -> float:
    # the rosenbrock function with a=1,b=100, minimum at (1,...,1)
    return np.square(1 - theta[0]) + 100 * np.square((theta[-1] - np.square(theta[0])))


@pytest.fixture
def mockobjective():
    func = partial(global_objective, local_objective=rosen)

    model_parameters = {
        "a": 1,
        "b": 1,
        "c": 1,
    }  # b in this case is a default parameter

    parameter_mapping = ModelHelpers.ParameterMapping(
        [], default_parameters=model_parameters  # empty parameter mapping
    )

    obj = Objective(
        func=func,
        bounds={"a": [0.0, 2], "c": [0, 2.0]},
        parameter_mapping=parameter_mapping,
    )

    return obj


class TestOptimization:
    @pytest.mark.parametrize("method", Optimization.optimization_funcs)
    def test_scipy_optimizers(self, method, mockobjective: Objective):
        opt = Optimization(
            objective=mockobjective,
            method=method,
            bounds=mockobjective.bounds,
            use_parallel=False,
        )

        res, info = opt.optimize()
        assert isinstance(res, dict)
        assert not np.isinf(info["fun"])

    @pytest.mark.parametrize("method", Optimization.pygmo_algos)
    def test_pygmo_optimizers(self, method, mockobjective: Objective):
        _method = [method]
        opt = Optimization(
            objective=mockobjective,
            method=_method,
            bounds=mockobjective.bounds,
            optimizer_kwargs={"n_jobs": 1},
        )

        res, info = opt.optimize()
        assert isinstance(res, dict)
        assert not np.isinf(info.loss)

    def test_estimate_archi_continued(self, mockobjective: Objective):
        _method = [Optimization.pygmo_algos[0]]
        opt = Optimization(
            objective=mockobjective,
            method=_method,
            bounds=mockobjective.bounds,
            optimizer_kwargs={"n_jobs": 1},
        )

        _, est_info = opt.optimize()

        opt = Optimization(
            method=est_info,
            objective=mockobjective,
            bounds=mockobjective.bounds,
            optimizer_kwargs={"n_jobs": 1},
        )

        res, est_info = opt.optimize()

        assert isinstance(res, dict)
        assert not np.isinf(est_info.loss)
