import numpy as np
import pytest
from scipy.stats import chi2

from estim8.objective import Objective
from estim8.optimizers import Optimization
from estim8.profile import (
    ProfileSampler,
    approximate_confidence_interval,
    calculate_negll_thresshold,
)


@pytest.fixture
def sample_profile_data():
    x = np.linspace(-2, 2, 100)
    y = x**2  # Simple quadratic function
    return x, y


def test_calculate_negll_threshold():
    """Test the calculation of negative log-likelihood threshold"""
    alpha = 0.05
    df = 1
    mle_negll = 0

    expected = chi2.ppf(1 - alpha, df) / 2
    result = calculate_negll_thresshold(alpha, df, mle_negll)

    assert result == expected


def test_approximate_confidence_interval(sample_profile_data):
    """Test the confidence interval approximation"""
    x, y = sample_profile_data
    threshold = 1.0

    lower, upper = approximate_confidence_interval(x, y, threshold)

    # For quadratic function y = x^2 with threshold 1.0,
    # confidence interval should be approximately (-1, 1)
    assert np.isclose(lower, -1.0, atol=0.1)
    assert np.isclose(upper, 1.0, atol=0.1)


def test_approximate_confidence_interval_errors(sample_profile_data):
    """Test error handling in confidence interval approximation"""
    x, y = sample_profile_data

    # Test with threshold above all y values
    with pytest.raises(ValueError):
        approximate_confidence_interval(x, y, threshold=100)


class TestProfileSampler:
    @pytest.fixture
    def mock_optimizer(self):
        class MockOptimizer:
            def __init__(self):
                self.objective = Objective(
                    func=lambda x: x,  # Simple quadratic function
                    bounds={"test_param": [-1, 1]},
                    parameter_mapping=type(
                        "mapping", (), {"set_parameter": lambda x, y, z: None}
                    )(),
                )
                self.task_id = "pl_job_1_0"

            def optimize(self):
                return {}, {"fun": 1.0}

        return MockOptimizer()

    @pytest.fixture
    def profile_sampler(self, mock_optimizer):
        return ProfileSampler(
            parameter="test_param",
            mle=0.0,
            mle_negll=0.0,
            negll_threshold=2.0,
            optimizer=mock_optimizer,
            bounds=[-1, 1],
            direction=1,
            stepsize=0.1,
        )

    def test_initialization(self, profile_sampler):
        """Test ProfileSampler initialization"""
        assert profile_sampler.parameter == "test_param"
        assert profile_sampler.mle == 0.0
        assert profile_sampler.direction == 1
        assert profile_sampler.stepsize == 0.1
        assert not profile_sampler.finished
        assert len(profile_sampler.samples) == 1  # Initial point

    def test_next_step(self, profile_sampler):
        """Test next step calculation"""
        next_val = profile_sampler.next_step()
        assert (
            next_val
            == 0.0
            + profile_sampler.direction
            * profile_sampler.stepsize
            * (profile_sampler.mle if profile_sampler.mle != 0 else 1)
        )

    def test_bounds_handling(self, profile_sampler):
        """Test bounds handling in next_step"""
        profile_sampler.samples[-1][0] = 0.95  # Near upper bound
        next_val = profile_sampler.next_step()
        assert next_val == 1.0  # Should hit upper bound
        assert profile_sampler.finished  # Should be finished when bound is hit

    def test_max_steps(self):
        """Test max_steps functionality"""
        sampler = ProfileSampler(
            parameter="test_param",
            mle=0.0,
            mle_negll=0.0,
            negll_threshold=2.0,
            optimizer=None,
            bounds=[-1, 1],
            direction=1,
            stepsize=0.1,
            max_steps=1,
        )

        sampler.next_step()
        assert sampler.finished

    def test_walk_profile(self, profile_sampler):
        """Test complete profile walking"""
        samples, param = profile_sampler.walk_profile()

        assert isinstance(samples, np.ndarray)
        assert param == "test_param"
        assert profile_sampler.finished
