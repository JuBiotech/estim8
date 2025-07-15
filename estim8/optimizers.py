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
"""This submodule implements the optimization of an objective function using various optimization algorithms.
"""
import inspect
import pickle
from functools import partial
from typing import Callable, List, Tuple
from warnings import warn

import joblib
import numpy as np
import scipy.optimize
import skopt

from . import generalized_islands
from .objective import Objective, objective_function_wrapper


class Optimization:
    """
    Class that manages the optimization of an objective function.
    """

    optimization_funcs = {
        "local": scipy.optimize.minimize,
        "de": scipy.optimize.differential_evolution,
        "bh": scipy.optimize.basinhopping,
        "shgo": scipy.optimize.shgo,
        "dual_annealing": scipy.optimize.dual_annealing,
        "gp": skopt.gp_minimize,
    }

    pygmo_algos = {
        "scipy_optimize",
        "de1220",
        "bee_colony",
        "gaco",
        "pso",
        "sga",
        "sea",
        "compass_search",
        "gwo",
        "cmaes",
        "simulated_annealing",
        "nsga2",
        "mbh",
        "ihs",
        "xnes",
        "de",
    }

    def __init__(
        self,
        objective: Objective | Callable,
        method: str | List[str],
        bounds: dict,
        optimizer_kwargs: dict = {},
        use_parallel=True,
        task_id: str = "estimate",
    ):
        """Initialize the Optimization class.

        Parameters
        ----------
        objective : callable
            The objective function to be optimized.
        method : str | List[str]
            The optimization method to be used.
        bounds : dict
            The bounds for the parameters.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for the optimizer, by default {}.
        use_parallel : bool, optional
            Whether to use parallel optimization, by default True.
        task_id : str, optional
            The task ID, by default "estimate".
        """
        self.bounds = bounds
        self.use_parallel = use_parallel
        self.objective = objective

        if isinstance(method, generalized_islands.PygmoEstimationInfo):
            self.optimize_func = Optimization.optimize_pygmo_archipelago_continued

        else:
            self.optimize_func = self.get_optimization_method(method)

        # prepare optimizer kwargs
        self.optimizer_kwargs = self.prepare_optimizer_kwargs(method, optimizer_kwargs)
        self._method = method
        self.task_id = task_id

    @staticmethod
    def get_optimization_method(method: str | List[str]) -> Callable:
        """Get the optimization method.

        Parameters
        ----------
        method : str | List[str]
            The optimization method(s).

        Returns
        -------
        callable
            The optimization function.

        Raises
        ------
        KeyError
            If the method is not supported.
        NotImplementedError
            If the method is not implemented.
        """
        if isinstance(method, str):
            try:
                opt_func = Optimization.optimization_funcs[method]
            except KeyError:
                raise KeyError(
                    f"{method} is not supported. Please choose of of {Optimization.optimization_funcs.keys()}"
                )
        elif isinstance(method, list):
            if set(method) - set(Optimization.pygmo_algos):
                raise NotImplementedError(
                    f"Algorithms {set(method)-set(Optimization.pygmo_algos)} are not implemented."
                )
            opt_func = Optimization.optimize_pygmo_archi

        return opt_func

    def prepare_optimizer_kwargs(self, method, optimizer_kwargs):
        """Prepare the optimizer keyword arguments.

        Parameters
        ----------
        method : str | List[str]
            The optimization method(s).
        optimizer_kwargs : dict
            The optimizer keyword arguments.

        Returns
        -------
        dict
            The prepared optimizer keyword arguments.

        Raises
        ------
        ValueError
            If the list of algorithms doesn't match the list of algorithm kwargs.
        """
        _optimizer_kwargs = optimizer_kwargs.copy()

        # continued optimization with pygmo archipelago
        if isinstance(method, generalized_islands.PygmoEstimationInfo):
            _optimizer_kwargs["estimation_info"] = method

        # check for pygmo algos and algos kwargs:
        if isinstance(method, list):
            _optimizer_kwargs["algos"] = method
            if "algos_kwargs" not in optimizer_kwargs:
                _optimizer_kwargs["algos_kwargs"] = [{} for _ in method]
            elif len(set(method)) == 1 and isinstance(
                optimizer_kwargs["algos_kwargs"], dict
            ):
                _optimizer_kwargs["algos_kwargs"] = [
                    optimizer_kwargs["algos_kwargs"] for _ in method
                ]
                if len(method) > 1:
                    warn("Passed algorithm kwargs are applied to all passed algorithms")
            elif len(method) == len(optimizer_kwargs["algos_kwargs"]):
                _optimizer_kwargs["algos_kwargs"] = optimizer_kwargs["algos_kwargs"]
            else:
                raise ValueError(
                    "List of algorithms doesn't match list of algorighm kwargs."
                )

        # update keywargs with bounds if neccessary
        if isinstance(method, list):
            _optimizer_kwargs["bounds"] = self.bounds
            if "mc_job" or "pl_job" in self.task_id:
                _optimizer_kwargs["init_pool"] = False

        elif method in self.optimization_funcs:
            bounds = list(self.bounds.values())
            if method == "gp":
                _optimizer_kwargs["dimensions"] = bounds
            elif method == "bh":
                if not "minimizer_kwargs" in optimizer_kwargs:
                    _optimizer_kwargs["minimizer_kwargs"] = {"bounds": bounds}
                else:
                    _optimizer_kwargs["minimizer_kwargs"]["bounds"] = bounds
            else:
                _optimizer_kwargs["bounds"] = list(self.bounds.values())

        # get starting point if neccessary and not provided
        if (method in ["local", "bh"]) and (not "x0" in optimizer_kwargs):
            _optimizer_kwargs["x0"] = np.array(
                [np.mean(val) for val in self.bounds.values()]
            )

        # define equivalents between methods for parallelization (if possible) and maximum iterations
        eq_kwargs = {
            "n_jobs": ["workers", "n_processes"],
            "max_iter": ["maxiter", "n_calls", "niter"],
        }

        # apply equivalent kwargs:
        for kw, equals in eq_kwargs.items():
            if kw in optimizer_kwargs:
                for eq_kw in equals:
                    _optimizer_kwargs[eq_kw] = optimizer_kwargs[kw]

        appended_kwargs = set(_optimizer_kwargs) - set(optimizer_kwargs)
        # reinsert max_iter and n_jobs
        for default_kw in ["max_iter", "n_jobs"]:
            appended_kwargs.add(default_kw)
        # check for wrpng kwargs
        _method_kwargs = set(inspect.signature(self.optimize_func).parameters)
        wrong_kwargs = set(_optimizer_kwargs) - _method_kwargs
        for wr_kwarg in wrong_kwargs:
            _optimizer_kwargs.pop(wr_kwarg)

        if wrong_kwargs - set(appended_kwargs):
            warn(
                f"The keywords {wrong_kwargs-set(appended_kwargs)} cannot be used for {self.optimize_func.__name__}"
            )

        return _optimizer_kwargs

    def optimize(self):
        """Optimize the objective function.

        Returns
        -------
        tuple
            The optimization result and additional information.
        """
        if self.use_parallel:
            optimize_result = self._optimize_parallel()
        else:
            optimize_result = self._optimize()

        if not isinstance(optimize_result, tuple):
            res, info = (
                dict(zip(list(self.bounds), optimize_result.x)),
                optimize_result,
            )
        else:
            res, info = optimize_result
        return res, info

    def _optimize_parallel(self):
        """Optimize the objective function in parallel.

        Returns
        -------
        tuple
            The optimization result and additional information.
        """
        if self.optimize_func is Optimization.optimize_pygmo_archipelago_continued:
            return self.optimize_func(self.objective, **self.optimizer_kwargs)
        else:
            # use the wrapper and rebuild Objective on subprocess for performance
            # the idea here is to serialize the objective once and thereby sidestep the multiprocessing serialization step on every function call
            return self.optimize_func(
                partial(
                    objective_function_wrapper,
                    objective=pickle.dumps(self.objective),
                    task_id=self.task_id,
                ),
                **self.optimizer_kwargs,
            )

    def _optimize(self):
        """Optimize the objective function single core.

        Returns
        -------
        tuple
            The optimization result and additional information.
        """
        return self.optimize_func(self.objective, **self.optimizer_kwargs)

    @staticmethod
    def optimize_pygmo_archi(
        objective: Callable,
        bounds: dict,
        algos: List[str],
        algos_kwargs: List[dict],
        n_processes: int = joblib.cpu_count(),
        pop_size=50,
        topology=generalized_islands.pygmo.unconnected(),
        max_iter: int = 10,
        report: int = 0,
        init_pool: bool = True,
    ) -> Tuple[dict, generalized_islands.PygmoEstimationInfo]:
        """Optimize the objective function using a pygmo archipelago.

        Parameters
        ----------
        objective : callable
            The objective function to be optimized.
        bounds : dict
            The bounds for the parameters.
        algos : List[str]
            The optimization algorithms to be used.
        algos_kwargs : List[dict]
            The keyword arguments for the optimization algorithms.
        n_processes : int, optional
            The number of processes to use, by default joblib.cpu_count().
        pop_size : int, optional
            The population size, by default 50.
        topology : pygmo.topology, optional
            The topology of the archipelago, by default pygmo.unconnected().
        max_iter : int, optional
            The number of evolutions, by default 10.
        report: int, optional
            The report level during optimization, by default 0.
            A report level of 1 yields data on the archipelago's island champions over evolutions.
            With a report level of 2 the islands current states are printed in the Log.


        Returns
        -------
        Tuple[dict, generalized_islands.PygmoEstimationInfo]
            The optimization result and additional information.
        """
        _, estimation_info = generalized_islands.PygmoHelpers.create_archipelago(
            objective=objective,
            bounds=bounds,
            algos=algos,
            algos_kwargs=algos_kwargs,
            n_processes=n_processes,
            pop_size=pop_size,
            topology=topology,
            report=report,
            init_pool=init_pool,
        )

        return Optimization.optimize_pygmo_archipelago_continued(
            None, estimation_info, max_iter
        )

    @staticmethod
    def optimize_pygmo_archipelago_continued(
        _,
        estimation_info: generalized_islands.PygmoEstimationInfo,
        max_iter: int = 10,
    ) -> Tuple[dict, generalized_islands.PygmoEstimationInfo]:
        """Continue optimizing the objective function using a pygmo archipelago object.

        Parameters
        ----------
        estimation_info : generalized_islands.PygmoEstimationInfo
            The estimation information and the archipelago itself before evolution(s).
        max_iter : int, optional
            The number of evolutions, by default 10.

        Returns
        -------
        Tuple[dict, generalized_islands.PygmoEstimationInfo]
            The optimization result and additional information.
        """
        archi = estimation_info.archi

        # evolve
        archi.evolve(max_iter)
        archi.wait()

        (
            estimates,
            estimation_info,
        ) = generalized_islands.PygmoHelpers.get_archipelago_results(
            archi, estimation_info
        )
        # update number of evolutions
        estimation_info.n_evos += max_iter

        return estimates, estimation_info
