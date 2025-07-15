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
"""
This module implements Workers that serve as a computaion backend for an objective function.
Further it implements the federated Worker class and functions for launching a pool of such services.
"""

from __future__ import annotations

import abc
import logging
import multiprocessing
import sys
from typing import TYPE_CHECKING, Dict, List

import grpclib
import numpy as np
import pytensor_federated
from numpy.core.multiarray import array as array

if TYPE_CHECKING:
    from .estimator import Estimator


def init_logging():
    """
    Configure logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("worker.log")],
    )


class Worker:
    """
    The Worker class wraps an instance of Estimator class and turns its objective for a single replicate into a callable.
    """

    def __init__(self, estimator: "Estimator", mc_sampling: bool = False):
        """
        Initializes the Worker class with the given model, time parameters, replicate IDs, data, and metric.

        Parameters
        ----------
        estimator : estim8.Estimator
            An instance of estim8.Estimator.
        mc_sampling : bool, optional
            If the Worker is set up for MC sampling or not, default is False.
        """
        self.estimator = estimator
        self.mc_sampling = mc_sampling

    def __call__(self, theta: np.array) -> np.array:
        """
        Calculates the objective function given the data by simulating the model with the parameter vector `theta` to evaluate.

        Parameters
        ----------
        theta : np.array
            The parameter vector `theta` to evaluate. Indices of replicate ID and eventually Monte Carlo sample are at last positions:
            theta = [
                theta_i,
                ..
                theta_j,
                mc_sample_ID,
                replicate_ID
            ]

        Returns
        -------
        np.array
            The value of the objective function.
        """

        # map theta to model parameters
        replicate_ID: str | None = self.estimator.replicate_IDs[
            int(theta[-1])
        ]  # extract encoded replicate ID
        if self.mc_sampling:
            mc_sample = int(theta[-2])  # extract encoded MC sample index
            data = self.estimator.resampling[mc_sample][replicate_ID]
        else:
            data = self.estimator.data[replicate_ID]

        parameters = dict(zip(self.estimator.model.parameters, theta))

        loss = self.estimator.objective_for_replicate(
            parameters=parameters, data=data, metric=self.estimator.metric
        )

        return np.array(loss)


class FederatedWorker(Worker):
    """
    A subclass of Worker that adds error handling and logging for federated computation.
    """

    logger = logging.getLogger(__name__)

    def __call__(self, theta: np.array) -> np.array:
        """
        Calculates the objective function given the data by simulating the model with the parameter vector `theta` to evaluate.

        Parameters
        ----------
        theta : np.array
            The parameter vector `theta` to evaluate.

        Returns
        -------
        np.array
            The value of the objective function.
        """
        try:
            return super().__call__(theta)
        except Exception as e:
            self.logger.error(f"Error in Worker: {e}")
            self.logger.exception(e)
            return np.array(np.inf)


def run_worker_service(
    host: str, port: int, estimator: "Estimator", mc_sampling: bool = False
):
    """
    Runs a FederatedWorker as a service bound to a host and port address.

    Parameters
    ----------
    host : str
        The host address of the service.
    port : int
        The port number of the service.
    estimator : estim8.Estimator
        An instance of estim8.Estimator.
    mc_sampling : bool, optional
        If the Worker is set up for MC sampling or not, default is False.

    Notes
    -----
    The loop is run forever if not stopped externally.
    """
    logger = logging.getLogger(__name__)
    model_fn = FederatedWorker(estimator=estimator)

    async def run_server():
        a2a_server = pytensor_federated.service.ArraysToArraysService(
            pytensor_federated.common.wrap_logp_func(model_fn)
        )

        server = grpclib.server.Server([a2a_server])

        try:
            await server.start(host, port)
            logger.info(f"Worker service started on {host}:{port}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Error running worker service on {host}:{port}: {e}")
            logger.exception(e)

    loop = pytensor_federated.utils.get_useful_event_loop()
    loop.run_until_complete(run_server())

    return


def run_worker_pool(
    host: str, ports: List[int], estimator: "Estimator", mc_sampling: bool = False
) -> List[multiprocessing.Process]:
    """
    Launches a pool of FederatedWorker services in parallel.

    Parameters
    ----------
    host : str
        The host address of worker services that are launched.
    ports : List[int]
        The ports of worker services.
    estimator : estim8.Estimator
        An instance of estim8.Estimator.
    mc_sampling : bool, optional
        If the Worker is set up for MC sampling or not, default is False.

    Returns
    -------
    List[multiprocessing.Process]
        The worker processes.
    """
    logger = logging.getLogger(__name__)

    server_processes = [
        multiprocessing.Process(
            target=run_worker_service,
            args=[host, port, estimator, mc_sampling],
        )
        for port in ports
    ]

    # launch processes
    for sp in server_processes:
        try:
            sp.start()
        except Exception as e:
            logger.error(f"Error launching worker: {e}")
            logger.exception(e)

    return server_processes
