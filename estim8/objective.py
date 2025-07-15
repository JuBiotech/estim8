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
"""This module implements the objective functions that are passed to an optimization algorithm.
"""
import abc
import os
import pickle
from typing import Callable, Dict, List, Optional

import numpy as np
import pytensor

from . import utils
from .workers import Worker

# global container for objective objects on subprocesses
_objectives: dict[str | None, Callable] = dict()


class Objective:
    """
    Objective function that is passed to an optimization algorithm.
    When called to evaluate a parameter set theta, conducts a replicate handling step and sends request for loss calculation to federated worker nodes via a compiled pytensor.function
    """

    def __init__(
        self,
        func: Callable,
        bounds: Dict[str, List[float]],
        parameter_mapping: utils.ModelHelpers.ParameterMapping,
        mc_sample: Optional[int] = 0,
    ) -> None:
        """
        Initialize the Objective function.

        Parameters
        ----------
        func : callable
            The function to evaluate the parameter sets of all model replicates. In case of a federated Worker setup, the compiled `pytesor.function` will send evaluation requests asynchrounously.
        bounds : Dict[str, List[float]]
            The bounds for the parameters.
        parameter_mapping : utils.ModelHelpers.ParameterMapping
            The parameter mapping for replicate handling.
        mc_sample : Optional[int], optional
            The Monte Carlo sample index, by default 0.
        """
        self.func = func
        self.bounds = bounds
        self.parameter_mapping = parameter_mapping
        self.mc_sample = (
            mc_sample  # NOTE: only used from workers if setup for mc sampling
        )

    def __call__(self, theta) -> np.array:
        """
        Evaluate the objective function for a given parameter set.

        Parameters
        ----------
        theta : np.array
            The parameter set to evaluate.

        Returns
        -------
        np.array
            The loss value.
        """
        # map to replicates
        replicate_parameters = [
            np.append(
                np.array(
                    list(
                        self.parameter_mapping.replicate_handling(
                            parameters=dict(zip(self.bounds, theta)), replicate_ID=rid
                        ).values()
                    )
                ),
                (
                    self.mc_sample,
                    rid_code,
                ),  # append by index of Monte Carlo sample and replicate ID
            )
            for rid_code, rid in enumerate(self.parameter_mapping.replicate_IDs)
        ]

        # calculate loss in parallel
        loss = float(self.func(*replicate_parameters)[0])

        return loss


def objective_function_wrapper(
    theta: np.array, objective, task_id: str | None = None
) -> float:
    """
    A wrapper around the objective. Creates global copy of the objective function in the current process to avoid serialization on every function call,
    and stores it in a dictionary with a unique task ID.

    Parameters
    ----------
    theta : np.array
        The parameter set to evaluate.
    objective: tuple
       A serialized callable.
    task_id : str, optional
        The task ID, by default None.

    Returns
    -------
    float
        The loss value.
    """

    global _objectives

    if task_id not in _objectives:
        _objectives[task_id] = pickle.loads(objective)

    return _objectives[task_id](theta)


def global_objective(*replicate_par_sets, local_objective: Callable):
    """
    Calculates a global objective as the sum of local objective mapped to replicate parameter sets.

    Parameters
    ----------
    local_objective : callable
        The local objective function.

    Returns
    -------
    list
        The global objective value.
    """

    return [np.sum(list(map(local_objective, replicate_par_sets)))]
