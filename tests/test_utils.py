import numpy as np
import pandas as pd
import pytest

from estim8.datatypes import Experiment, Measurement
from estim8.utils import EstimatorHelpers, ModelHelpers


class TestModelHelpers:
    par_defaults = {"a": 0, "b": 0, "c": 0}
    replicate_specific_parameters = ["b", "c"]
    values = [1, 2]
    replicate_IDs = ["1st", "2nd"]

    parameter_mappings = []
    for rid, value in zip(replicate_IDs, values):
        for param in replicate_specific_parameters:
            parameter_mappings.append(
                ModelHelpers.ParameterMapper(
                    global_name=param, replicate_ID=rid, value=value
                )
            )
    parameter_mapping = ModelHelpers.ParameterMapping(
        mappings=parameter_mappings,
        replicate_IDs=replicate_IDs,
        default_parameters=par_defaults,
    )

    @pytest.mark.parametrize(
        "parameters,rid,result",
        [
            ({}, "1st", 2),
            ({"a": 1, "b_1st": 0, "b_2nd": 4}, "1st", 2),
            ({"a": 1, "b_1st": 0, "b_2nd": 4}, "2nd", 7),
        ],
    )
    def test_replicate_handling(self, parameters, rid, result):
        params = self.parameter_mapping.replicate_handling(
            parameters=parameters, replicate_ID=rid
        )

        assert sum(params.values()) == result


class TestEstimatorHelpers:
    t_vals = np.linspace(0, 100, 101)
    x_vals = np.linspace(0, 100, 101)

    df = pd.DataFrame(dict(a=x_vals), index=t_vals)

    input_data_types = [df, Experiment(measurements=df)]
    input_errors = [None, pd.DataFrame(dict(a=x_vals * 0.01), index=t_vals)]
    input_error_model = [None]
    input_rids = [None, "1st"]

    @pytest.mark.parametrize("data", input_data_types)
    @pytest.mark.parametrize("errors", input_errors)
    @pytest.mark.parametrize("error_model", input_error_model)
    @pytest.mark.parametrize("replicate_ID", input_rids)
    def test_make_replicate(self, data, errors, error_model, replicate_ID):
        replicate = EstimatorHelpers.make_replicate(
            data=data, errors=errors, error_model=error_model, replicate_ID=replicate_ID
        )
        replicate = replicate[replicate_ID]

        assert replicate_ID == replicate.replicate_ID
