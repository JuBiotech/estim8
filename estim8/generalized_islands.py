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
"""This module implements helper functions for working with pygmo`s generalized islands approach."""

import importlib
import logging
from typing import Any, Callable, Dict, List, Protocol, Tuple, runtime_checkable
from warnings import warn

import joblib
import numpy as np
import pandas as pd

# Configure logger with a handler and formatter
logger = logging.getLogger("estim8.generalized_islands")
logger.setLevel(logging.INFO)

# Add handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


# Define protocol classes for type checking
@runtime_checkable
class PygmoArchipelago(Protocol):
    def get_champions_f(self) -> np.ndarray[List[float]]:
        ...

    def get_champions_x(self) -> List[List[float]]:
        ...

    def wait_check(self) -> None:
        ...

    def __getitem__(self, key: int) -> Any:
        ...

    def push_back(self, **kwargs) -> None:
        ...

    def evolve(self, n_evolutions: int = 1) -> None:
        ...

    def wait(self) -> None:
        ...

    def set_migrant_handling(self, handler: Any) -> None:
        ...

    def __iter__(self) -> Any:
        ...


@runtime_checkable
class PygmoMpIsland(Protocol):
    """Protocol for pygmo.mp_island type checking"""

    def __init__(self, use_pool: bool = True) -> None:
        ...

    def run_evolve(self, algo: Any, pop: Any) -> Any:
        ...

    def shutdown_pool(self) -> None:
        ...

    def init_pool(self, n_processes: int) -> None:
        ...


@runtime_checkable
class PygmoProblem(Protocol):
    def extract(self, cls: Any) -> Any:
        ...

    def get_fevals(self) -> int:
        ...


def optional_import(module_name, mock_name=None):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if mock_name:
            return importlib.import_module(mock_name)
        else:

            class MockArchipelago:
                def __getattr__(self, name):
                    warn(
                        f"Module {module_name} is not installed. In order to use this function try:\n conda install -c conda-forge pygmo"
                    )
                    return None

            class MockModule:
                def __init__(self):
                    self.archipelago = MockArchipelago
                    self.algorithm = MockArchipelago
                    self.problem = MockArchipelago
                    self.mp_island = MockArchipelago
                    self.topology = MockArchipelago
                    self.migrant_handling = MockArchipelago
                    self.fully_connected = lambda: None

                def __getattr__(self, name):
                    warn(
                        f"Module {module_name} is not installed. In order to use this function try:\n conda install -c conda-forge pygmo"
                    )
                    return lambda *args, **kwargs: None

            return MockModule()


pygmo = optional_import("pygmo")

from .objective import Objective


class Estim8_mp_island(pygmo.mp_island):  # type: ignore
    """A custom mp_island implementation with evolution logging capabilities."""

    def __init__(self, use_pool=True, report_level=1):
        """
        report: int, optional
            The report level during optimization, by default 1.
            A report level of 1 yields data on the archipelago's island champions over evolutions.
            With a report level of 2 the islands current states are printed in the Log.
        """
        self.evo_count = 0
        self.report_level = report_level
        # Create empty DataFrame with MultiIndex and explicit dtypes
        self.evo_trace = pd.DataFrame(
            {
                "island_id": pd.Series(dtype="int64"),
                "champion_loss": pd.Series(dtype="float64"),
                "champion_theta": pd.Series(dtype="object"),
            }
        )
        self.evo_trace.index = pd.MultiIndex.from_arrays(
            [pd.Series(dtype="int64"), pd.Series(dtype="str")],
            names=["evolution", "algorithm"],
        )
        super().__init__(use_pool)

    def __copy__(self):
        """Return a copy of the island."""
        new_island = Estim8_mp_island(self._use_pool, self.report_level)
        new_island.evo_count = self.evo_count
        new_island.evo_trace = self.evo_trace.copy()  # Make sure to copy the trace too
        return new_island

    def run_evolve(self, algo, pop):
        """Run evolution with logging of progress."""
        res = super().run_evolve(algo, pop)

        try:
            self.evo_count += 1

            if self.report_level == 2:
                # Log to console
                logger.info(
                    f"## Evolution {self.evo_count} of island {id(self)} completed:\n"
                    f"      Algorithm: {algo.get_name()}\n"
                    f"      Champion loss: {res[1].champion_f[0]:.2e}"
                )

            # Create new row data directly without using pd.Series
            new_data = pd.DataFrame(
                {
                    "island_id": [id(self)],
                    "champion_loss": [
                        float(res[1].champion_f[0])
                    ],  # Explicitly convert to float
                    "champion_theta": [res[1].champion_x],
                },
                index=pd.MultiIndex.from_tuples(
                    [(self.evo_count, algo.get_name())],
                    names=["evolution", "algorithm"],
                ),
            )

            # Use copy=True and ignore_index=False to preserve MultiIndex
            self.evo_trace = pd.concat([self.evo_trace, new_data], copy=True)
            return res

        except Exception as e:
            logger.error(f"Evolution of island {id(self)} failed: {str(e)}")
            raise


class UDproblem:
    """A wrapper class around an Objective function with functions required for creating a user defined pygmo.problem."""

    def __init__(self, objective: Callable, bounds: dict):
        """
        Initialize the UDproblem class.

        Parameters
        ----------
        objective : callable
            The objective function to be optimized.
        bounds : dict
            The bounds for the parameters.
        """
        self.objective = objective
        self.bounds = bounds

    def get_extra_info(self) -> list:
        """
        Get extra information about the problem.

        Returns
        -------
        list
            The keys of the bounds dictionary.
        """
        return list(self.bounds.keys())

    def fitness(self, theta) -> np.array:
        """
        Evaluate the fitness of a solution.

        Parameters
        ----------
        theta : np.array
            The solution to evaluate.

        Returns
        -------
        np.array
            The fitness value.
        """
        return np.array([self.objective(theta)])

    def get_bounds(self) -> tuple:
        """
        Get the bounds of the problem.

        Returns
        -------
        tuple
            The lower and upper bounds.
        """
        lower_bounds = np.array([b[0] for b in self.bounds.values()])
        upper_bounds = np.array([b[1] for b in self.bounds.values()])
        return lower_bounds, upper_bounds


class PygmoEstimationInfo:
    """An object to store additional information from an evolved archipelago as well as the archipelago itself."""

    def __init__(
        self,
        archi: PygmoArchipelago,
        udi_type=pygmo.mp_island,
        fun: float = np.inf,
        n_evos: int = 0,
    ):
        """
        Initialize the PygmoEstimationInfo class.

        Parameters
        ----------

        archi : PygmoArchipelago
            The archipelago.
        fun : float, optional
            Best objective function value among all champions from the evolved archipelago, by default np.inf.
        n_evos : int, optional
            The number of total evolutions of the (evolved) archipelago, by default 0.
        """
        self.fun = fun
        self.n_evos = n_evos
        self.archi = archi
        self.udi_type = udi_type
        self.evo_trace = None

    def get_f_evals(self) -> int:
        """
        Get the number of accumulated objective function evaluations of the archipelago.

        Returns
        -------
        int
            The number of accumulated objective function evaluations.
        """
        return np.sum(
            [
                PygmoHelpers.extract_archipelago_problem(self.archi, i).get_fevals()
                for i, _ in enumerate(self.archi)
            ]
        )

    def __repr__(self) -> str:
        """
        Get a string representation of the PygmoEstimationInfo object.

        Returns
        -------
        str
            A string representation of the PygmoEstimationInfo object.
        """
        return f"fun: {self.fun} \n n_evos: {self.n_evos} \n"


class PygmoHelpers:
    """Helper functions for working with pygmo."""

    # use default algorithm kwargs from pyFOOMB
    algo_default_kwargs: Dict[str, Dict[str, Any]] = {
        "scipy_optimize": {},
        "bee_colony": {"limit": 2, "gen": 10},
        "cmaes": {"gen": 10, "force_bounds": False, "ftol": 1e-8, "xtol": 1e-8},
        "compass_search": {"max_fevals": 100, "start_range": 1, "stop_range": 1e-6},
        "de": {"gen": 10, "ftol": 1e-8, "xtol": 1e-8},
        "de1220": {"gen": 10, "variant_adptv": 2, "ftol": 1e-8, "xtol": 1e-8},
        "gaco": {"gen": 10},
        "ihs": {"gen": 10 * 4},
        "maco": {"gen": 10},
        "mbh": {"algo": "compass_search", "perturb": 0.1, "stop": 2},
        "moead": {"gen": 10},
        "nlopt": {"solver": "lbfgs"},
        "nsga2": {"gen": 10},
        "nspso": {"gen": 10},
        "pso": {"gen": 10},
        "pso_gen": {"gen": 10},
        "sade": {"gen": 10, "variant_adptv": 2, "ftol": 1e-8, "xtol": 1e-8},
        "sea": {"gen": 10 * 4},
        "sga": {"gen": 10},
        "simulated_annealing": {},
        "xnes": {"gen": 10, "ftol": 1e-8, "xtol": 1e-8, "eta_mu": 0.05},
    }

    @staticmethod
    def get_pygmo_algorithm_instance(name: str, **kwargs) -> Any:
        """Creates an instance of a pygmo.algorithm given the name and algorithm kwargs.

        Parameters
        ----------
        name : str
            The name of the optimization algorithm.
        kwargs : dict, optional
            Keyword arguments for the algorithm, by default None, which means the default arguments of the respective pygmo.algorithm will be used.

        Returns
        -------
        pygmo.algorithm
            The pygmo algorithm instance.

        Raises
        ------
        ValueError
            If the algorithm name is not supported.
        """
        if not hasattr(pygmo, name):
            raise ValueError(f"{name} is not a supported pygmo algorithm.")

        _kwargs: Dict[str, Any] = {}
        # use default kwargs if possible
        if name in PygmoHelpers.algo_default_kwargs:
            _kwargs.update(PygmoHelpers.algo_default_kwargs[name])
        if kwargs is not None:
            _kwargs.update(kwargs)

        if name == "mbh":
            # Get inner algorithm
            _inner_kwargs = {}
            _outer_kwargs = {}
            for key, val in _kwargs.items():
                if key.startswith("inner_"):
                    _inner_kwargs[key[6:]] = val
                else:
                    _outer_kwargs[key] = val
            # and continue with outer kwargs
            _kwargs = _outer_kwargs
            _kwargs["algo"] = PygmoHelpers.get_pygmo_algorithm_instance(
                _outer_kwargs["algo"], **_inner_kwargs
            )

        return getattr(pygmo, name)(**_kwargs)

    @staticmethod
    def create_pygmo_pop(args):
        """Create a pygmo population.

        Parameters
        ----------
        args : tuple
            The arguments for creating the population.

        Returns
        -------
        pygmo.population
            The created population.
        """
        problem, pop_size, seed = args
        return pygmo.population(problem, pop_size, seed=seed)

    @staticmethod
    def create_archipelago(
        objective: Callable,
        bounds: dict,
        algos: List[str],
        algos_kwargs: List[dict],
        pop_size: int,
        topology: Any = pygmo.fully_connected(),
        report=0,
        n_processes=joblib.cpu_count(),
        init_pool: bool = True,
    ) -> Tuple[PygmoArchipelago, PygmoEstimationInfo]:
        """Creates a pygmo.archipelago object using the generalized islands model.

        Parameters
        ----------
        objective : callable
            An instance of an estim8.Objective function that is used to create a UDproblem.
        bounds : dict
            The bounds for the parameters.
        algos : list[str]
            A list of optimization algorithms for the individual islands of the archipelago.
        algos_kwargs : list[dict]
            A list of algorithm kwargs corresponding to passed optimizers.
        pop_size : int
            Population size for each individual island.
        topology : pygmo.topology, optional
            Represents the connection policy between the islands of the archipelago, by default pygmo.fully_connected().
        report: int, optional
            The report level during optimization, by default 0. In case of report > 0, the evoltions are runned with :class:`Estim8_mp_island` which logs the evolution trace.
            A report level of 1 yields data on the archipelago's island champions over evolutions.
            With a report level of 2 the islands current states are printed in the Log.
        n_processes : int, optional
            The number of processes to use, by default joblib.cpu_count().

        Returns
        -------
        pygmo.archipelago
            The created archipelago.
        PygmoEstimationInfo
            An estimation info object containing the archipelago
        """

        if report:
            udi = Estim8_mp_island(report_level=report)

        else:
            udi = pygmo.mp_island()

        # init process pool backing mp_islands
        udi.shutdown_pool()
        udi.init_pool(n_processes)

        problem = pygmo.problem(UDproblem(objective, bounds))

        # get optimization algorithm instances
        algos = [
            PygmoHelpers.get_pygmo_algorithm_instance(algo, **algo_kwargs)
            for algo, algo_kwargs in zip(algos, algos_kwargs)
        ]

        archi = pygmo.archipelago(t=topology)
        archi.set_migrant_handling(pygmo.migrant_handling.preserve)

        # create the populations and add to archipelago
        pop_creation_args = ((problem, pop_size, seed) for seed in range(len(algos)))
        pops = list(map(PygmoHelpers.create_pygmo_pop, pop_creation_args))

        for i, (algo, pop) in enumerate(zip(algos, pops)):
            archi.push_back(udi=udi, algo=algo, pop=pop)
            if report:
                print(f">>> Created Island {i+1} using {algos[i]}")
        archi.wait_check()

        # initialize estimation info object
        estimation_info = PygmoEstimationInfo(archi=archi, udi_type=type(udi))

        return archi, estimation_info

    @staticmethod
    def extract_archipelago_problem(archi: PygmoArchipelago, i=0) -> PygmoProblem:
        """Extracts the user defined problem from an archipelago, implemented as pygmo.problem(UDproblem).

        Parameters
        ----------
        archi : PygmoArchipelago
            The evolved archipelago.
        i : int, optional
            Index of the island from which the problem is extracted, by default 0.

        Returns
        -------
        pygmo.problem
            The extracted problem.
        """
        return archi[i].get_population().problem

    @staticmethod
    def get_estimates_from_archipelago(archi: PygmoArchipelago) -> Tuple[dict, float]:
        """Extracts the best estimates and corresponding value of the objective function from an archipelago object.

        Parameters
        ----------
        archi : PygmoArchipelago
            The evolved archipelago.

        Returns
        -------
        Tuple[dict, float]
            Dictionary of best estimates according to the smallest loss aka objective function value.
            The smallest loss value among all islands.
        """
        unknowns = (
            PygmoHelpers.extract_archipelago_problem(archi)
            .extract(UDproblem)
            .get_extra_info()
        )
        loss_vals = archi.get_champions_f()
        best_loss = min(loss_vals)
        champ_id = loss_vals.index(best_loss)
        best_theta = archi.get_champions_x()[champ_id]

        return {
            parameter: val for parameter, val in zip(unknowns, best_theta)
        }, best_loss[0]

    @staticmethod
    def get_archipelago_results(
        archi: PygmoArchipelago, estimation_info: PygmoEstimationInfo
    ) -> Tuple[dict, PygmoEstimationInfo]:
        """Extracts the results of an evolved archipelago and updates additional estimation info.

        Parameters
        ----------
        archi : PygmoArchipelago
            The evolved archipelago.
        estimation_info : PygmoEstimationInfo
            Additional information about the archipelago before evolution(s).

        Returns
        -------
        Tuple[dict, PygmoEstimationInfo]
            Dictionary of best estimates according to the smallest loss value.
            Updated additional information about the evolved archipelago containing the archipelago itself.
        """
        estimates, fun = PygmoHelpers.get_estimates_from_archipelago(archi)

        estimation_info.fun = fun
        estimation_info.archi = archi

        # get evo trace if needed
        if estimation_info.udi_type == Estim8_mp_island:
            evo_trace = pd.concat(
                [island.extract(Estim8_mp_island).evo_trace for island in archi],
                ignore_index=False,  # Changed this to preserve MultiIndex
            )
            estimation_info.evo_trace = evo_trace

        return estimates, estimation_info
