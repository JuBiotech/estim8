import numpy as np
import pytest
from numpy.core.multiarray import array as array
from scipy.stats import rv_continuous

import estim8.error_models as em
from estim8.datatypes import Experiment, Measurement, ModelPrediction


class gauss(rv_continuous):
    def _pdf(self, x):
        return np.exp(-(x**2) / 2.0) / np.sqrt(2.0 * np.pi)


class AbsoluteErrorModel(em.BaseErrorModel):
    abs_err = 0.1
    error_distribution = gauss()

    def generate_error_data(self, values: np.array) -> np.array:
        return values + self.abs_err


@pytest.fixture
def linearErrorModel():
    return em.LinearErrorModel()


class TestErrorModels:
    y_true = np.linspace(0, 100, 101)
    sim = ModelPrediction(name="test", values=y_true, timepoints=y_true)

    @pytest.mark.parametrize(
        "error_model", [AbsoluteErrorModel(), em.LinearErrorModel()]
    )
    def test_generate_errors(self, error_model: em.BaseErrorModel):
        errs = error_model.generate_error_data(self.y_true)
        assert isinstance(errs, np.ndarray)

    @pytest.mark.parametrize(
        "error_model", [AbsoluteErrorModel(), em.LinearErrorModel()]
    )
    def test_log_likelihood_calc(self, error_model: em.BaseErrorModel):
        measurement = Measurement(
            name="test",
            values=self.y_true,
            timepoints=self.y_true,
            error_model=error_model,
        )
        loss = measurement.get_loss(metric="negLL", model_prediction=self.sim)
        assert isinstance(loss, float)

    @pytest.mark.parametrize(
        "error_model", [AbsoluteErrorModel(), em.LinearErrorModel()]
    )
    def test_resample_data(self, error_model: em.BaseErrorModel):
        measurement = Measurement(
            name="test",
            values=self.y_true,
            timepoints=self.y_true,
            error_model=error_model,
        )
        resampling = measurement.get_sampling(1)

        for sample in resampling:
            loss = sample.get_loss(metric="negLL", model_prediction=self.sim)
            assert isinstance(loss, float)

    def test_bad_input_LinearErrorModel(self):
        errormodel = em.LinearErrorModel()
        with pytest.raises(IOError):
            errormodel.error_model_params = {"slope": 1, "wrong_arg": 1}
