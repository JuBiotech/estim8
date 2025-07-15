# estim8
# Copyright (C) 2025 Forschungszentrum Jülich GmbH

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""This module implements utility functions for the estim8 package.
"""
from __future__ import annotations

import abc
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from warnings import warn

import numpy as np
import pandas as pd
import pytensor
import pytensor_federated
import pytensor_federated.rpc

from . import error_models
from .datatypes import Constants, Experiment

SINGLE_ID = Constants.SINGLE_ID


class ModelHelpers:
    class ParameterMapper:
        """Defines a replicate specific parameter."""

        def __init__(
            self,
            global_name: str,
            replicate_ID: str,
            value: float = 0,
            local_name: str | None = None,
        ):
            """
            Parameters
            ----------
            global_name: str
                Parameter key on a global scope.
            replicate_ID : str
                Replicate ID.
            value : float, optional
                Parameter value, by default 0
            local_name: str, optional
                Parameter key on a local scope, by default None.

            Raises
            ------
            ValueError
                If global_name is not a string.
            ValueError
                If replicate_ID is not a string.
            """
            if not isinstance(global_name, str):
                raise ValueError("Global parameter name must be a string.")
            if not isinstance(replicate_ID, str):
                raise ValueError("Replicate ID must be a string.")

            self.global_name = global_name
            self.value = value
            self.replicate_ID = replicate_ID
            if local_name is None:
                self.local_name = f"{global_name}_{replicate_ID}"
            else:
                self.local_name = local_name

    class ParameterMapping:
        def __init__(
            self,
            mappings: List[ModelHelpers.ParameterMapper],
            default_parameters: dict,
            replicate_IDs: list[str | None] = [SINGLE_ID],
        ):
            self._mapping = mappings
            self.default_parameters = default_parameters
            self.replicate_IDs = replicate_IDs

        @property
        def mapping(self) -> pd.DataFrame:
            """
            A pretty presentation of parameter mapping as a pandas.DataFrame.

            Returns
            -------
            pd.DataFrame
                The parameter mapping.
            """
            map = {
                rID: {
                    param: {
                        "replicate ID": rID,
                        "global name": param,
                        "local name": param,
                        "value": value,
                    }
                    for param, value in self.replicate_handling(
                        replicate_ID=rID
                    ).items()
                }
                for rID in self.replicate_IDs
            }

            # overwrite local names
            for _par_map in self._mapping:
                map[_par_map.replicate_ID][_par_map.global_name][
                    "local name"
                ] = _par_map.local_name

            df = (
                pd.concat(
                    [pd.DataFrame(rid_pars.values()) for rid_pars in map.values()],
                    axis=0,
                )
                .set_index(["global name", "replicate ID"])
                .sort_index()
            )
            return df

        def replicate_handling(
            self, replicate_ID: str | None, parameters: Dict[str, float] = dict()
        ):
            """
            Handle parameter mapping for a specific replicate.

            Parameters
            ----------
            replicate_ID : str
                The replicate ID.
            parameters : Dict[str, float], optional
                The parameters, by default dict().

            Returns
            -------
            dict
                The complete parameters for the replicate.
            """
            local_params = list(
                filter(lambda _par: _par.replicate_ID == replicate_ID, self._mapping)
            )

            complete_params = self.default_parameters.copy()
            # and update with parametermapping
            complete_params.update(
                {_par.global_name: _par.value for _par in local_params}
            )

            for par, val in parameters.items():
                # local and estimated
                if par in [_par.local_name for _par in local_params]:
                    for _par in local_params:
                        if _par.local_name == par:
                            complete_params[_par.global_name] = val
                # global and estimated
                elif par in self.default_parameters and not par in [
                    _par.global_name for _par in local_params
                ]:
                    complete_params[par] = val
                else:
                    pass

            return complete_params

        def set_parameter(self, name: str, value: float):
            """
            Set a parameter value by its name (either local or global).

            Parameters
            ----------
            name : str
                The parameter name (can be local or global)
            value : float
                The new parameter value

            Raises
            ------
            ValueError
                If the parameter name is not found in either local or global parameters
            """
            # Check if it's a local parameter
            for param in self._mapping:
                if param.local_name == name:
                    param.value = value
                    return

            # Check if it's a global parameter
            if name in self.default_parameters:
                self.default_parameters[name] = value
                return

            raise ValueError(
                f"Parameter '{name}' not found in local or global parameters"
            )


class EstimatorHelpers:
    @staticmethod
    def make_replicate(
        data: Experiment | pd.DataFrame,
        errors: pd.DataFrame = None,
        error_model: error_models.BaseErrorModel = error_models.LinearErrorModel(),
        replicate_ID: str | None = SINGLE_ID,
    ):
        """
        Create a replicate from data.

        Parameters
        ----------
        data : Experiment or pd.DataFrame
            The data for the replicate.
        errors : pd.DataFrame, optional
            The errors for the data, by default None.
        error_model : error_models.BaseErrorModel, optional
            The error model, by default None.
        replicate_ID : str, optional
            The replicate ID, by default SINGLE_ID.

        Returns
        -------
        dict
            The replicate data.
        """
        if isinstance(data, pd.DataFrame):
            if error_model is None:
                error_model = error_models.LinearErrorModel()
            _data = Experiment(
                data, errors=errors, replicate_ID=replicate_ID, error_model=error_model
            )
        elif isinstance(data, Experiment):
            _data = data
            if _data.replicate_ID != replicate_ID:
                warn(
                    f"Specified Replicate ID does not match the data. Setting Replicate ID to {replicate_ID}."
                )
                _data.replicate_ID = replicate_ID
        else:
            raise ValueError(
                f"For using data of a single experiment pass a pandas.DataFrame or a datatypes.Experiment instance, not {type(data)}."
            )
        return {replicate_ID: _data}

    @staticmethod
    def get_t_from_data(data: Dict[str, Experiment]) -> Tuple[float, float, float]:
        """
        Get the time vector from data.

        Parameters
        ----------
        data : Dict[str, Experiment]
            The data.

        Returns
        -------
        Tuple[float, float, float]
            The time vector for simulations given by (t_start, t_end, stepsize).
        """
        t_end = np.array(
            [
                [measurement.timepoints.max() for measurement in _exp.measurements]
                for _exp in data.values()
            ]
        ).max()
        max_data_points = np.array(
            [
                [len(measurement.timepoints) for measurement in _exp.measurements]
                for _exp in data.values()
            ]
        ).max()
        return 0.0, t_end, t_end / max_data_points

    @staticmethod
    def generate_mc_samples(estimator_data: dict[str, Experiment], n_samples: int):
        """
        Generate Monte Carlo samples from the data.

        Parameters
        ----------
        estimator_data : dict[str, Experiment]
            The data.
        n_samples : int
            The number of samples.

        Returns
        -------
        list
            The Monte Carlo samples.
        """
        return [
            dict(zip(estimator_data.keys(), sample))
            for sample in zip(
                *[
                    replicate.generate_mc_samples(n_samples)
                    for replicate in estimator_data.values()
                ]
            )
        ]

    @staticmethod
    def make_tensor_function(
        functions: list[Callable], replicate_IDs: list[str | None]
    ) -> pytensor.function:
        """
        Create a tensor function.

        Parameters
        ----------
        functions : list[callable]
            The functions.
        replicate_IDs : list[str]
            The replicate IDs.

        Returns
        -------
        pytensor.function
            The tensor function.
        """
        parameter_set_like = [pytensor.tensor.vector(str(rid)) for rid in replicate_IDs]
        fn_result = pytensor.tensor.sum(
            [
                func(replicate_parameters)
                for func, replicate_parameters in zip(functions, parameter_set_like)
            ]
        )
        return pytensor.function(
            [[replicate_parameters] for replicate_parameters in parameter_set_like],
            [fn_result],
        )

    @staticmethod
    def update_optimizer_kwargs(opt_kwargs: dict, **kwargs):
        """
        Update optimizer keyword arguments.

        Parameters
        ----------
        opt_kwargs : dict
            The optimizer keyword arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        dict
            The updated optimizer keyword arguments.
        """
        for _kwarg, value in kwargs.items():
            if _kwarg not in opt_kwargs:
                opt_kwargs[_kwarg] = value
        return opt_kwargs

    @staticmethod
    def test_worker(
        host_and_port: Tuple[str, int], theta_test: np.array, retries: int
    ) -> bool:
        """
        Test a worker.

        Parameters
        ----------
        host_and_port : Tuple[str, int]
            The host and port of the worker.
        theta_test : np.array
            The test parameter vector.
        retries : int
            The number of retries.

        Returns
        -------
        bool
            Whether the worker is up.
        """
        host, port = host_and_port
        client = pytensor_federated.common.LogpServiceClient(host=host, port=port)
        worker_up = False

        # TODO retries
        test_result = client.evaluate(theta_test)
        if isinstance(test_result, np.ndarray):
            worker_up = True

        del client
        return worker_up

    @staticmethod
    def test_workers(hosts_and_ports, theta_test, retries=2):
        """
        Test multiple workers.

        Parameters
        ----------
        hosts_and_ports : list of Tuple[str, int]
            The hosts and ports of the workers.
        theta_test : np.array
            The test parameter vector.
        retries : int, optional
            The number of retries, by default 2.

        Returns
        -------
        bool or list
            Whether all workers are up or the workers that are down.
        """
        worker_responses = [
            EstimatorHelpers.test_worker(host_and_port, theta_test, retries)
            for host_and_port in hosts_and_ports
        ]

        if all(worker_responses):
            return True
        else:
            workers_down = np.array(hosts_and_ports)[~np.array(worker_responses)]
            print(workers_down)
            return workers_down

    @staticmethod
    def get_worker_loads(
        hosts_and_ports: List[Tuple[str, int]]
    ) -> List[pytensor_federated.rpc.GetLoadResult]:
        """
        Get the loads of the workers.

        Parameters
        ----------
        hosts_and_ports : List[Tuple[str, int]]
            The hosts and ports of the workers.

        Returns
        -------
        List[pytensor_federated.rpc.GetLoadResult]
            The loads of the workers.
        """
        load_task = pytensor_federated.service.get_loads_async(hosts_and_ports)
        loop = pytensor_federated.utils.get_useful_event_loop()
        loads = loop.run_until_complete(load_task)
        return loads

    @staticmethod
    def workers_up(hosts_and_ports: List[Tuple[str, int]]) -> List[bool]:
        """
        Check if the workers are up.

        Parameters
        ----------
        hosts_and_ports : List[Tuple[str, int]]
            The hosts and ports of the workers.

        Returns
        -------
        List[bool]
            Whether the workers are up.
        """
        load_results = EstimatorHelpers.get_worker_loads(hosts_and_ports)
        return [
            isinstance(res, pytensor_federated.rpc.GetLoadResult)
            for res in load_results
        ]

    @staticmethod
    def wait_for_worker_launch(
        hosts_and_ports: List[Tuple[str, int]], timeout=30
    ) -> bool:
        """
        Wait for the federated workers to launch.

        Parameters
        ----------
        hosts_and_ports : List[Tuple[str, int]]
            The hosts and ports of the workers.
        timeout : int, optional
            The timeout in seconds, by default 30.

        Returns
        -------
        bool
            Whether the workers launched successfully.
        """
        t_start = time.time()
        launch_complete = False
        while not launch_complete and time.time() <= t_start + timeout:
            launch_complete = all(EstimatorHelpers.workers_up(hosts_and_ports))
        return launch_complete


class WorkerNotUpError(Exception):
    def __init__(self, host_and_ports: List[Tuple[str, int]]) -> None:
        """
        Exception raised when workers are not up.

        Parameters
        ----------
        host_and_ports : List[Tuple[str, int]]
            The hosts and ports of the workers.
        """
        self.message = (
            f"The following workers did not respond correctly: {host_and_ports}"
        )
        super().__init__(self.message)
