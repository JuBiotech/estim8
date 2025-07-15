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
"""This module implements the Estimator class for parameter estimation and uncertainty quantification.
"""
import concurrent.futures as futures
import pickle
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Tuple, get_args
from warnings import warn

import fmpy
import numpy as np
import pandas as pd
import pytensor_federated

from . import error_models, models, utils
from .datatypes import Constants, Experiment
from .objective import Objective, global_objective
from .optimizers import Optimization, generalized_islands
from .profile import ProfileSampler, calculate_negll_thresshold
from .workers import Worker, init_logging, run_worker_pool

SINGLE_ID = Constants.SINGLE_ID
VALID_METRICS = Constants.VALID_METRICS


class Estimator:
    """
    Parameter estimation and uncertainty quantification manager.

    This class provides a comprehensive interface for:
    - Parameter estimation using various optimization algorithms
    - Monte Carlo sampling for uncertainty quantification
    - Profile likelihood analysis
    - Handling multiple experimental replicates
    - Distributed computation through federated workers

    Parameters
    ----------
    model : models.Estim8Model
        Model to fit to experimental data
    bounds : dict[str, List[float]]
        Parameter bounds formatted as {parameter_name: [lower_bound, upper_bound]}
    data : Union[Experiment, Dict[str, Experiment], pd.DataFrame, Dict[str, pd.DataFrame]]
        Experimental measurements to fit model against. Can be:
        - Single experiment as Experiment or DataFrame
        - Multiple replicates as Dict[str, Experiment] or Dict[str, DataFrame]
    t : List[float], optional
        Simulation time settings as [t_start, t_end, stepsize]. If None, derived from data
    metric : Literal["SS", "WSS", "negLL"], optional
        Loss function for optimization:
        - "SS": Sum of squares
        - "WSS": Weighted sum of squares
        - "negLL": Negative log-likelihood
        Default is "SS"
    error_model : error_models.BaseErrorModel, optional
        Error model for experimental data. Only used when data is DataFrame.
        Default is LinearErrorModel()
    parameter_mapping : Union[List[utils.ModelHelpers.ParameterMapper], utils.ModelHelpers.ParameterMapping], optional
        Maps parameters between replicates. Either:
        - List of ParameterMapper objects
        - Complete ParameterMapping object
        Default is None (no mapping)

    Attributes
    ----------
    replicate_IDs : List[str]
        IDs of experimental replicates
    model : models.Estim8Model
        Reference to model being fit
    bounds : dict
        Parameter bounds
    metric : str
        Current loss function
    error_model : error_models.BaseErrorModel
        Current error model
    parameter_mapping : utils.ModelHelpers.ParameterMapping
        Current parameter mapping

    Methods
    -------
    estimate(method, max_iter, optimizer_kwargs={}, n_jobs=1, federated_workers=False)
        Main parameter estimation method
    mc_sampling(method, n_jobs, max_iter, n_samples=100, mcs_at_once=1)
        Monte Carlo sampling for uncertainty quantification
    profile_likelihood(p_opt, method, max_iter, n_points=3, dp_rel=0.1)
        Profile likelihood analysis for parameter confidence intervals

    Notes
    -----
    - Supports both single-core and parallel computation
    - Can use federated worker architecture for distributed computing, enabling parallel simulation of model replicates within a single objective function call
    - Handles replicate-specific parameters through parameter mapping
    - Provides multiple optimization algorithms through scipy, scikit-optimize and pygmo

    Examples
    --------
    Basic parameter estimation:

    >>> estimator = Estimator(model=model, bounds=bounds, data=data)
    >>> estimates, info = estimator.estimate(method='de', max_iter=1000)

    Parallelized parameter estimation:
    >>> estimates, info = estimator.estimate(method='de', n_jobs=2, max_iter=1000)

    With federated workers:

    >>> estimates, info = estimator.estimate(method='de', max_iter=1000, federated_workers=4)

    Monte Carlo sampling:

    >>> results = estimator.mc_sampling(method='de', n_jobs=4, max_iter=100, n_samples=10)
    """

    _valid_metrics = get_args(VALID_METRICS)
    _valid_optimizers = [
        *Optimization.optimization_funcs.keys(),
        *Optimization.pygmo_algos,
    ]

    def __init__(
        self,
        model: models.Estim8Model,
        bounds: dict[str, List[float]],
        data: Experiment
        | Dict[str, Experiment]
        | pd.DataFrame
        | Dict[str, pd.DataFrame],
        t: List[float] | None = None,
        metric: VALID_METRICS = "SS",
        error_model: error_models.BaseErrorModel = error_models.LinearErrorModel(),
        parameter_mapping: list[utils.ModelHelpers.ParameterMapper]
        | utils.ModelHelpers.ParameterMapping
        | None = None,
    ) -> None:
        """Initialize an Estimator instance for parameter estimation.

        Parameters
        ----------
        model : models.Estim8Model
            The model class to be fitted. Must be a subclass of Estim8Model that runs the simulations.

        bounds : dict[str, List[float]]
            Dictionary defining the parameter bounds for estimation.
            Format: {"param_name": [lower_bound, upper_bound]}

        data : Union[Experiment, Dict[str, Experiment], pd.DataFrame, Dict[str, pd.DataFrame]]
            Experimental data for model fitting. Can be provided in several formats:
            - Single experiment as Experiment object or DataFrame
            - Multiple replicates as dict of Experiment objects or DataFrames
            - DataFrames must contain time points and measurement values
            - Each replicate must have consistent observation mappings

        t : List[float], optional
            Time vector specification as [t_start, t_end, step_size].
            If None, will be automatically determined from the experimental data.

        metric : Literal["SS", "WSS", "negLL"], optional
            Loss function for optimization:
            - "SS": Sum of squared residuals (default)
            - "WSS": Weighted sum of squared residuals
            - "negLL": Negative log-likelihood

        error_model : error_models.BaseErrorModel, optional
            Error model used when data is provided as DataFrame.
            Default is LinearErrorModel.
            Only used for converting DataFrame data to Experiment objects.

        parameter_mapping : Union[List[ParameterMapper], ParameterMapping], optional
            Defines parameter relationships between replicates:
            - List of ParameterMapper objects defining individual mappings
            - Complete ParameterMapping object
            - None means all parameters are applied to all replicates(default)

        Notes
        -----
        - The model must provide observables matching the values in the data observation mappings
        - When using DataFrames, observation mappings will be auto-generated
        - Parameter mappings allow different parameters per replicate if needed
        - Time vector can be automatically determined but explicit definition recommended
        """
        self.model = model
        self.bounds = bounds
        self.metric = metric
        self.error_model = error_model
        self.data = data
        self.t = t
        self.parameter_mapping = parameter_mapping

    # %% Properties
    @property
    def data(self):
        return self._data

    @data.setter
    def data(
        self,
        value: Experiment
        | Dict[str, Experiment]
        | pd.DataFrame
        | Dict[str, pd.DataFrame],
    ) -> None:
        """Set the experimental data and corresponding replicate IDs.

        Parameters
        ----------
        value : Experiment | Dict[str, Experiment] | pd.DataFrame | Dict[str, pd.DataFrame]
            The experimental data.


        Raises
        ------
        ValueError
            If the data type is not supported.
        """
        # setup internal data structure as dictionary with {replicate_ID: Experiment}
        self._data = {}
        if not isinstance(value, dict):
            self._data.update(
                utils.EstimatorHelpers.make_replicate(
                    value, error_model=self.error_model
                )
            )

        elif isinstance(value, dict):
            for r_id, replicate_data in value.items():
                self._data.update(
                    utils.EstimatorHelpers.make_replicate(
                        replicate_data, replicate_ID=r_id, error_model=self.error_model
                    )
                )

        else:
            raise ValueError(
                f"{type(value)} is not supported. Please use a datatype of {[Experiment, Dict[str,Experiment], pd.DataFrame, Dict[str, pd.DataFrame]]}"
            )
        # get replicate IDs from data dictionary
        self.replicate_IDs: List[str | None] = list(self._data.keys())

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value: Tuple[float, float, float] | None):
        """Set the time vector for model simulations.

        Parameters
        ----------
        value : Tuple[float, float, float]
            The time vector as [t_start, t_end, stepsize].

        Raises
        ------
        ValueError
            If the time vector is not specified correctly.
        """
        if value is None:
            if self.data is not None:
                self._t = utils.EstimatorHelpers.get_t_from_data(self.data)
                warn(
                    f"No timepoints specified. Timepoints derived from data were set to {self._t}.",
                    UserWarning,
                )
            else:
                raise ValueError(
                    f"No experimental data given to get simulation timevector from."
                )
        elif (
            isinstance(value, list)
            and all([isinstance(_val, (float, int)) for _val in value])
            and len(value) == 3
        ):
            self._t = value
        else:
            raise ValueError(
                f"t must be specified as list of 3 entries, containing [t0, t_end, stepsize]"
            )

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value: str):
        """Set the discrepancy measure between model predictions and experimental data.

        Parameters
        ----------
        value : str
            The discrepancy measure.

        Raises
        ------
        ValueError
            If the discrepancy measure is not valid.
        """
        if value in self._valid_metrics:
            self._metric = value
        else:
            raise ValueError(f"{value} is not a valid Loss function for optimization.")

    @property
    def parameter_mapping(self):
        return self._parameter_mapping

    @parameter_mapping.setter
    def parameter_mapping(
        self,
        value: List[utils.ModelHelpers.ParameterMapper]
        | utils.ModelHelpers.ParameterMapping,
    ):
        """Set the parameter mappings for replicate handling.

        Parameters
        ----------
        value : List[utils.ModelHelpers.ParameterMapper] | utils.ModelHelpers.ParameterMapping
            The parameter mappings.
        """
        if isinstance(value, utils.ModelHelpers.ParameterMapping):
            self._parameter_mapping = value
        else:
            if value is None:
                _value = list()
            else:
                _value = value
            self._parameter_mapping = utils.ModelHelpers.ParameterMapping(
                mappings=_value,
                default_parameters=self.model.parameters,
                replicate_IDs=self.replicate_IDs,
            )

    @property
    def hosts_and_ports(self):
        return self._hosts_and_ports

    @hosts_and_ports.setter
    def hosts_and_ports(self, value: List[Tuple[str, int]]):
        """Set the hosts and ports for federated workers and compile a function for federated loss computation.

        Parameters
        ----------
        value : tuple[str, int]
            The hosts and ports.
        """
        self._hosts_and_ports = value
        clients = [
            pytensor_federated.common.LogpServiceClient(hosts_and_ports=value)
            for _ in self.replicate_IDs
        ]
        remote_functions = [
            pytensor_federated.AsyncLogpOp(client.evaluate_async) for client in clients
        ]

        self.func = utils.EstimatorHelpers.make_tensor_function(
            remote_functions, self.replicate_IDs
        )

    def objective_for_replicate(
        self,
        parameters: dict,
        data: Experiment,
        metric: Literal["SS", "WSS", "negLL"] | None = None,
    ) -> float:
        """
        Calculate loss function value for a single experimental replicate.

        Parameters
        ----------
        parameters : dict
            Model parameters as {parameter_name: value}
        data : Experiment
            Experimental data for this replicate
        metric : Literal["SS", "WSS", "negLL"], optional
            Loss function type, by default None (uses instance metric)

        Returns
        -------
        float
            Loss function value. Returns np.inf if simulator crashes with RuntimeError, ValueError,
            ZeroDivisionError, OverflowError or fmpy.fmi1.FMICallException,.
        """
        if metric is None:
            metric = self.metric
        t0, t_end, stepsize = self.t

        try:
            sim = self.model.simulate(
                t0=t0,
                t_end=t_end,
                stepsize=stepsize,
                parameters=parameters,
                observe=list(data.observation_mapping.values()),
            )
            return data.calculate_loss(simulation=sim, metric=metric)
        except (
            # catch simulation exceptions in a more generic fashion
            RuntimeError,
            ValueError,
            ZeroDivisionError,
            OverflowError,
            fmpy.fmi1.FMICallException,
        ) as e:
            warn(
                "Simulation of the model using parameters {parameters} failed with {e}."
            )
            return np.inf

    def objective(self, theta: np.array) -> float:
        """
        Calculate total loss function value across all replicates.

        Parameters
        ----------
        theta : np.array
            Parameter vector to evaluate

        Returns
        -------
        float
            Total loss function value
        """
        replicate_parameters = {
            rID: self.parameter_mapping.replicate_handling(
                replicate_ID=rID, parameters=dict(zip(self.bounds, theta))
            )
            for rID in self.replicate_IDs
        }

        loss: float = 0.0
        for rid, parameters in replicate_parameters.items():
            loss += self.objective_for_replicate(
                parameters=parameters, data=self.data[rid], metric=self.metric
            )
        return loss

    def launch_workers(
        self,
        n_workers: int | None = None,
        host: str = "localhost",
        ports: List[int] | None = None,
        start_at_port: int = 9500,
        mc_sampling=False,
    ) -> None:
        """
        Launch federated worker processes for distributed computation.

        Parameters
        ----------
        n_workers : int, optional
            Number of worker processes, by default None
        host : str, optional
            Host address for workers, by default "localhost"
        ports : List[int], optional
            Specific ports for workers, by default None
        start_at_port : int, optional
            Starting port number when auto-assigning, by default 9500
        mc_sampling : bool, optional
            Whether workers will be used for Monte Carlo sampling, by default False

        Raises
        ------
        ValueError
            If neither n_workers nor ports are specified
        """
        if ports is None:
            if n_workers is None:
                raise ValueError("Either n_workers or ports must be specified.")
            else:
                _ports = [start_at_port + i for i in range(n_workers)]
        else:
            if len(ports) != n_workers:
                warn(f"n_workers argument overriden by ports.")
            _ports = ports

        print(f"Launching {len(_ports)} workers...")
        # start worker subprocesses
        self.server_processes = run_worker_pool(
            host=host, ports=_ports, estimator=self, mc_sampling=mc_sampling
        )
        self.hosts_and_ports = [(host, port) for port in _ports]

    def test_workers(self):
        """Test the responsiveness of the federated workers.

        Raises
        ------
        utils.WorkerNotUpError
            If any worker is not responsive.
        """
        # check for worker responsivness
        launch_complete = utils.EstimatorHelpers.wait_for_worker_launch(
            hosts_and_ports=self._hosts_and_ports
        )
        if launch_complete is False:
            raise utils.WorkerNotUpError(self._hosts_and_ports)
        print("testing workers..")
        test_theta = np.append(np.array(list(self.model.parameters.values())), (0, 0))
        test_result = utils.EstimatorHelpers.test_workers(
            self.hosts_and_ports, theta_test=test_theta
        )
        if test_result is True:
            print("Worker test succesfull.")
        else:
            raise utils.WorkerNotUpError(test_result)

    def shutdown_workers(self):
        """Shut down the federated workers."""
        print("Shutting down workers...")
        if hasattr(self, "server_processes"):
            for sp in self.server_processes:
                sp.terminate()
                sp.join()
            del self.server_processes
        del self._hosts_and_ports

    @staticmethod
    def use_federated_workers(func: Callable):
        """
        Decorator for managing federated worker lifecycle.

        This decorator handles:
        1. Worker initialization
        2. Worker testing
        3. Function execution
        4. Worker cleanup
        5. Error handling

        Parameters
        ----------
        func : callable
            Function to wrap

        Returns
        -------
        callable
            Wrapped function with worker lifecycle managementy
        """

        def inner_func(self, *args, **kwargs):
            error = False
            try:
                # extract kwargs for worker launch
                n_workers = kwargs.pop("federated_workers")
                worker_kwargs = kwargs.pop("worker_kwargs", {})
                mc_sampling = kwargs.pop("mc_sampling", None)
                # launch workers:
                if not hasattr(self, "hosts_and_ports"):
                    # init worker logging
                    init_logging()
                    self.launch_workers(
                        n_workers, mc_sampling=mc_sampling, **worker_kwargs
                    )
                # test worker responsiveness
                self.test_workers()
                result = func(self, *args, **kwargs)
                return result
            except Exception as e:
                error = True
                return None, None
            finally:
                self.shutdown_workers()
                if error:
                    raise

        return inner_func

    def check_problem_input(self) -> None:
        """
        Validate estimation problem setup.

        Checks:
        1. Model observables match data observation mappings
        2. Data completeness and consistency

        Raises
        ------
        KeyError
            If observation mappings don't match model observables
        """
        # asssert values of data observation_mapping correspond to model observables
        if any(
            [
                set(experiment.observation_mapping.values())
                - set(self.model.observables)
                for experiment in self.data.values()
            ]
        ):
            wrong_keys = {
                obs
                for experiment in self.data.values()
                for obs in experiment.observation_mapping.values()
                if obs not in self.model.observables
            }
            raise KeyError(
                f"The values {wrong_keys} in observation mapping(s) are not contained in the model.observables {self.model.observables}"
            )
        # TODO: maybe add some other checks, e.g. no error is None

    def estimate(
        self,
        method: str | List[str],
        max_iter: int,
        optimizer_kwargs: dict = {},
        n_jobs: int = 1,
        federated_workers: int | bool = False,
        worker_kwargs: dict = {},
    ):
        """Estimate the parameters of the model.

        Parameters
        ----------
        method : str | List[str]
            The optimization method(s).
        max_iter : int
            The maximum number of iterations.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for the optimizer, by default {}.
        n_jobs: int, optional
            The number of parallel jobs, by default 1.
        federated_workers : int | bool, optional
            The number of federated workers, by default False.
        worker_kwargs : dict, optional
            Additional keyword arguments for federated workers. These are passed to the
            :meth:`launch_workers` method (see :meth:`launch_workers` for details), by default {}.

        Returns
        -------
        tuple
            The optimization result and additional information.
        """
        optimizer_kwargs = utils.EstimatorHelpers.update_optimizer_kwargs(
            optimizer_kwargs, max_iter=max_iter, n_jobs=n_jobs
        )

        # sanity check problem
        self.check_problem_input()

        # check for parallelizatioon
        if any(
            [
                isinstance(method, list),
                n_jobs > 1,
                federated_workers > 1,
                isinstance(method, generalized_islands.PygmoEstimationInfo),
            ]
        ):
            if not federated_workers:
                # define the objective function
                self.func = partial(global_objective, local_objective=Worker(self))
                return self._estimate_parallel(
                    method=method, optimizer_kwargs=optimizer_kwargs
                )
            else:
                # objective function defined upon worker launch or setting the hosts_and_ports attribute
                return self._estimate_parallel_federated(
                    method=method,
                    optimizer_kwargs=optimizer_kwargs,
                    federated_workers=federated_workers,
                    worker_kwargs=worker_kwargs,
                )
        elif isinstance(method, str):
            return self._estimate(method=method, optimizer_kwargs=optimizer_kwargs)

    def _estimate(self, method: str, optimizer_kwargs: dict) -> tuple:
        """
        Perform single-core parameter estimation.

        Parameters
        ----------
        method : str
            Optimization method name
        optimizer_kwargs : dict
            Keywords arguments for optimizer

        Returns
        -------
        tuple
            (parameter_estimates, optimization_info)

        Raises
        ------
        ValueError
            If method not available for single-core estimation
        """
        if method not in Optimization.optimization_funcs:
            raise ValueError(
                f"{method} is not available for singlecore parameter estimation."
            )

        optimizer = Optimization(
            objective=self.objective,
            method=method,
            bounds=self.bounds,
            optimizer_kwargs=optimizer_kwargs,
            use_parallel=False,
        )
        return optimizer.optimize()

    @use_federated_workers
    def _estimate_parallel_federated(self, *args, **kwargs):
        """Parallel parameter estimation using federated workers.

        Returns
        -------
        tuple
            The optimization result and additional information.
        """
        return self._estimate_parallel(*args, **kwargs)

    def _estimate_parallel(
        self,
        method: str | List[str],
        optimizer_kwargs: dict = {},
    ) -> tuple:
        """
        Perform parallel parameter estimation.

        Parameters
        ----------
        method : Union[str, List[str]]
            Optimization method(s)
        optimizer_kwargs : dict, optional
            Keywords arguments for optimizer, by default {}

        Returns
        -------
        tuple
            (parameter_estimates, optimization_info)
        """
        # define objective
        objective = Objective(
            func=self.func,
            bounds=self.bounds,
            parameter_mapping=self.parameter_mapping,
        )

        # define optimization job
        optimizer = Optimization(
            objective=objective,
            method=method,
            bounds=self.bounds,
            optimizer_kwargs=optimizer_kwargs,
        )

        return optimizer.optimize()

    def profile_likelihood(
        self,
        p_opt: dict,
        method: str | List[str],
        max_iter: int,
        n_jobs: int = 1,
        federated_workers: int = 0,
        optimizer_kwargs: dict = {},
        p_at_once=1,
        max_steps: int | None = None,
        stepsize: float = 0.02,
        p_inv: list | None = None,
        alpha: float = 0.05,
        worker_kwargs: dict = {},
    ):
        """Calculate the profile likelihood for the estimated parameters.

        Parameters
        ----------
        p_opt : dict
            The optimal parameter values.
        method : str | List[str]
            The optimization method(s).
        max_iter : int
            The maximum number of iterations.
        n_jobs : int, optional
            The number of parallel jobs, by default 1.
        federated_workers : int, optional
            The number of federated workers, by default 0.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for the optimizer, by default {}.
        p_at_once : int, optional
            The number of parameters to profile at once, by default 1.
        n_points : int, optional
            The number of points to evaluate, by default 3.
        dp_rel : float, optional
            The relative parameter variation width, by default 0.1.
        p_inv : list, optional
            The parameters to investigate, by default None.
        worker_kwargs : dict, optional
            Additional keyword arguments for federated workers. These are passed to the
            :meth:`launch_workers` method (see :meth:`launch_workers` for details), by default {}.

        Returns
        -------
        dict
            The profile likelihood results.

        Raises
        ------
        KeyError
            If the parameter names in p_inv or p_opt are not specified in the bounds.
        ValueError
            If dp_rel is not in the range (0,1).
        """
        # check
        if p_inv is None:
            p_inv = list(self.bounds.keys())

        if set(p_inv) - set(self.bounds):
            raise KeyError(
                f"The parameter names in p_inv: {set(p_inv) - set(self.bounds)} are not specified in the bounds"
            )
        if set(p_opt) - set(self.bounds):
            raise KeyError(
                f"The parameter names in p_opt: {set(p_inv) - set(self.bounds)} are not specified in the bounds"
            )

        ## dp_rel
        if (stepsize > 1) or (stepsize <= 0):
            raise ValueError(
                f"Relative parameter variation width dp_rel must be in (0,1), not {stepsize}."
            )

        if self.metric != "negLL":
            warn("Setting metric to negLL for profile likelihood calculation.")
            self.metric = "negLL"
        optimizer_kwargs = utils.EstimatorHelpers.update_optimizer_kwargs(
            optimizer_kwargs, max_iter=max_iter, n_jobs=n_jobs
        )

        # federated worker setup
        if federated_workers:
            return self._profile_likelihood_federated(
                p_opt=p_opt,
                method=method,
                optimizer_kwargs=optimizer_kwargs,
                federated_workers=federated_workers,
                p_at_once=p_at_once,
                max_steps=max_steps,
                stepsize=stepsize,
                p_inv=p_inv,
                alpha=alpha,
                worker_kwargs=worker_kwargs,
            )

        else:
            # define the objective function
            self.func = partial(global_objective, local_objective=Worker(self))
            return self._profile_likelihood(
                p_opt=p_opt,
                method=method,
                optimizer_kwargs=optimizer_kwargs,
                p_at_once=p_at_once,
                max_steps=max_steps,
                stepsize=stepsize,
                alpha=alpha,
                p_inv=p_inv,
            )

    @use_federated_workers
    def _profile_likelihood_federated(self, *args, **kwargs):
        """Calculate the profile likelihood using federated workers.

        Returns
        -------
        dict
            The profile likelihood results.
        """
        return self._profile_likelihood(*args, **kwargs)

    def _profile_likelihood(
        self,
        p_opt: dict,
        method: str | List[str],
        optimizer_kwargs: dict,
        p_inv: list[str],
        p_at_once: int = 1,
        max_steps: int | None = None,
        stepsize: float = 0.1,
        alpha: float = 0.05,
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Calculate profile likelihood for parameter uncertainty.

        Parameters
        ----------
        p_opt : dict
            Optimal parameter values
        method : Union[str, List[str]]
            Optimization method(s)
        optimizer_kwargs : dict
            Keywords arguments for optimizer
        p_inv : list, optional
            Parameters to investigate
        p_at_once : int, optional
            Parameters to profile simultaneously, by default 1
        max_points : int, optional
            Maximum number of steps per parameter and direction, by default None.
        stepsize : float, optional
            Relative parameter variation per iteration step, by default 0.02


        Returns
        -------
        Dict[str, List[Dict[str, float]]]
            Profile likelihood results for each parameter
        """
        # define negLL threshhold as stopping criterion
        mle_negll = self.objective(list(p_opt.values()))
        threshhold = calculate_negll_thresshold(
            alpha=alpha, df=len(p_opt), mle_negll=mle_negll
        )

        # Prepare Bounds, initial point, and investigated values for each parameter to investigate
        pl_jobs = []
        # define a task counter
        i = 0
        for invest in p_inv:
            p0_i = p_opt.copy()
            p_opt_i = p0_i.pop(invest)
            bounds_i = {
                par: _bound for par, _bound in self.bounds.items() if not par == invest
            }
            directions: List[Literal[-1, 1]] = [-1, 1]

            for direction in directions:
                # deepcopy parameter mapping and update fixed value
                _parameter_mapping = deepcopy(self.parameter_mapping)

                # define objective
                objective = Objective(
                    func=self.func,
                    bounds=bounds_i,
                    parameter_mapping=_parameter_mapping,
                )

                optimizer = Optimization(
                    objective=objective,
                    bounds=bounds_i,
                    method=method,
                    optimizer_kwargs=optimizer_kwargs,
                    use_parallel=True,
                    task_id=f"pl_job_{i}_0",
                )

                # Define the ProfileSampler, serialize it and add to jobs
                pl_jobs.append(
                    (
                        pickle.dumps(
                            ProfileSampler(
                                parameter=invest,
                                mle=p_opt_i,
                                mle_negll=mle_negll,
                                negll_threshold=threshhold,
                                optimizer=optimizer,
                                bounds=list(self.bounds[invest]),
                                direction=direction,
                                stepsize=stepsize,
                                max_steps=max_steps,
                            )
                        )
                    )
                )

        # initialize results
        task_results: Dict[str, list] = {p: [] for p in p_inv}
        with futures.ProcessPoolExecutor(max_workers=p_at_once) as executor:
            opt_results = [
                executor.submit(_profile_likelihood_calc, pl_job) for pl_job in pl_jobs
            ]

            for opt_result in futures.as_completed(opt_results):
                samples, _inv = opt_result.result()
                task_results[_inv].append(samples)

        result: Dict[str, np.ndarray] = {
            parameter: np.unique(np.vstack(value), axis=0)
            for parameter, value in task_results.items()
        }

        return result

    def mc_sampling(
        self,
        method: str | List[str],
        n_jobs: int,
        max_iter: int,
        federated_workers: int = 0,
        optimizer_kwargs: dict = {},
        n_samples: int = 100,
        mcs_at_once: int = 1,
        worker_kwargs: dict = {},
    ):
        """Perform Monte Carlo sampling for parameter estimation problem.

        Parameters
        ----------
        method : str | List[str]
            The optimization method(s).
        n_jobs : int
            The number of parallel jobs.
        max_iter : int
            The maximum number of iterations.
        federated_workers : int, optional
            The number of federated workers, by default 0.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for the optimizer, by default {}.
        n_samples : int, optional
            The number of Monte Carlo samples, by default 100.
        mcs_at_once : int, optional
            The number of Monte Carlo samples to process at once, by default 1.
        worker_kwargs : dict, optional
            Additional keyword arguments for federated workers. These are passed to the
            :meth:`launch_workers` method (see :meth:`launch_workers` for details), by default {}.

        Returns
        -------
        list
            The results of the Monte Carlo sampling.
        """
        # generate mc samples of data if not already defined
        if not hasattr(self, "resampling"):
            self.resampling = utils.EstimatorHelpers.generate_mc_samples(
                self.data, n_samples
            )

        # update optimizer kwargs
        optimizer_kwargs = utils.EstimatorHelpers.update_optimizer_kwargs(
            optimizer_kwargs, max_iter=max_iter, n_jobs=n_jobs
        )

        if federated_workers:
            return self._mc_sampling_federated(
                method=method,
                optimizer_kwargs=optimizer_kwargs,
                federated_workers=federated_workers,
                mc_sampling=True,
                n_samples=n_samples,
                mcs_at_once=mcs_at_once,
                worker_kwargs=worker_kwargs,
            )
        else:
            self.func = partial(
                global_objective, local_objective=Worker(self, mc_sampling=True)
            )
            return self._mc_sampling(
                method=method,
                optimizer_kwargs=optimizer_kwargs,
                n_samples=n_samples,
                mcs_at_once=mcs_at_once,
            )

    @use_federated_workers
    def _mc_sampling_federated(self, *args, **kwargs):
        """Performs Monte Carlo sampling using federated workers.

        Returns
        -------
        list
            The results of the Monte Carlo sampling.
        """
        return self._mc_sampling(*args, **kwargs)

    def _mc_sampling(
        self,
        method: str | List[str],
        optimizer_kwargs: dict,
        n_samples: int,
        mcs_at_once: int,
    ) -> List[tuple]:
        """
        Perform Monte Carlo sampling estimation.

        Parameters
        ----------
        method : Union[str, List[str]]
            Optimization method(s)
        optimizer_kwargs : dict
            Keywords arguments for optimizer
        n_samples : int
            Number of Monte Carlo samples
        mcs_at_once : int
            Number of samples to process simultaneously

        Returns
        -------
        List[tuple]
            List of (parameter_estimates, optimization_info) for each sample
        """
        mc_jobs = []
        # define the objective
        for i in range(n_samples):
            # create a unique objective
            objective = Objective(
                func=self.func,
                bounds=self.bounds,
                parameter_mapping=self.parameter_mapping,
                mc_sample=i,
            )
            # define optimization task, serialize it here and add to task list
            mc_jobs.append(
                pickle.dumps(
                    Optimization(
                        objective=objective,
                        method=method,
                        bounds=self.bounds,
                        optimizer_kwargs=optimizer_kwargs,
                        use_parallel=True,
                        task_id=f"mc_job_{i}",
                    )
                )
            )
        # initialize results
        res_mc = []

        with futures.ProcessPoolExecutor(max_workers=mcs_at_once) as executor:
            results = [executor.submit(_mc_estimate, mc_job) for mc_job in mc_jobs]

            # TODO: backup possibility
            for _res in results:
                res_mc.append(_res.result())
                # Print Callback Message
                print(f"---- Sample {len(res_mc)} completed")

        return res_mc


def _profile_likelihood_calc(profile_sampler: Any):
    """
    Runs a ProfileSampler for a single parameter in descending or ascending direction.

    Parameters
    ----------
    profile_sampler : tuple
        A tuple containing the serialized ProfileSampler object.

    Returns
    -------
    np.ndarray
        An array containing the fixed parameter values and profile likelihood values.
    """

    # deserialize Optimization object
    _profile_sampler: ProfileSampler = pickle.loads(profile_sampler)

    return _profile_sampler.walk_profile()


def _mc_estimate(optimize_job):
    """
    Perform estimation for a single given optimization job of a Monte Carlo sample.

    Parameters
    ----------
    optimize_job : bytes
        The serialized Optimization object.

    Returns
    -------
    tuple
        The optimization result and additional information.
    """
    _optimize_job: Optimization = pickle.loads(optimize_job)
    mc_estimate, _ = _optimize_job.optimize()
    del _
    return mc_estimate


# %%
