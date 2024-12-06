"""This module implements helper functions for working with pygmo`s generalized islands approach.
"""

from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pygmo

from .objective import Objective


class UDproblem:
    """A wrapper class around an Objective function with functions required for creating a user defined pygmo.problem."""

    def __init__(self, objective: callable, bounds: dict):
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
        return self.bounds.keys()

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

    def __init__(self, archi: pygmo.archipelago, loss: float = np.inf, n_evos: int = 0):
        """
        Initialize the PygmoEstimationInfo class.

        Parameters
        ----------
        archi : pygmo.archipelago
            The archipelago.
        loss : float, optional
            Best loss value among all champions from the evolved archipelago, by default np.inf.
        n_evos : int, optional
            The number of total evolutions of the (evolved) archipelago, by default 0.
        """
        self.loss = loss
        self.n_evos = n_evos
        self.archi = archi

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
        return f"Loss: {self.loss} \n n_evos: {self.n_evos} \n"


class PygmoHelpers:
    """Helper functions for working with pygmo."""

    # use default algorithm kwargs from pyFOOMB
    algo_default_kwargs = {
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
    def get_pygmo_algorithm_instance(name: str, kwargs: dict = None) -> pygmo.algorithm:
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

        _kwargs = {}
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
                _outer_kwargs["algo"], _inner_kwargs
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
    def resize_archi_process_pools(archi: pygmo.archipelago, n_processes: int):
        """Resize the process pools of the archipelago.

        Parameters
        ----------
        archi : pygmo.archipelago
            The archipelago.
        n_processes : int
            The number of processes.
        """
        for island in archi:
            island.extract(pygmo.mp_island).shutdown_pool()
            island.extract(pygmo.mp_island).init_pool(len(n_processes))
        # TODO: check for a better method here, like calling shutdown and init a single time globally

    @staticmethod
    def create_archipelago(
        objective: callable,
        bounds: dict,
        algos: List[str],
        algos_kwargs: List[dict],
        pop_size: int,
        topology: pygmo.topology = pygmo.fully_connected(),
        report=False,
        n_processes=joblib.cpu_count(),
    ) -> pygmo.archipelago:
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
        report : bool, optional
            Whether to report the progress, by default False.
        n_processes : int, optional
            The number of processes to use, by default joblib.cpu_count().

        Returns
        -------
        pygmo.archipelago
            The created archipelago.
        """
        # init process pool backing mp_islands
        pygmo.mp_island.shutdown_pool()
        pygmo.mp_island.init_pool(n_processes)

        problem = pygmo.problem(UDproblem(objective, bounds))

        # get optimization algorithm instances
        algos = [
            PygmoHelpers.get_pygmo_algorithm_instance(algo, algo_kwargs)
            for algo, algo_kwargs in zip(algos, algos_kwargs)
        ]

        archi = pygmo.archipelago(t=topology)
        archi.set_migrant_handling(pygmo.migrant_handling.preserve)

        # create the populations and add to archipelago
        pop_creation_args = ((problem, pop_size, seed) for seed in range(len(algos)))
        with joblib.parallel_backend("loky", n_jobs=len(algos)):
            pops = joblib.Parallel(verbose=int(report))(
                map(joblib.delayed(PygmoHelpers.create_pygmo_pop), pop_creation_args)
            )

        # kill idle processes
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True)

        for i, (algo, pop) in enumerate(zip(algos, pops)):
            archi.push_back(udi=pygmo.mp_island(), algo=algo, pop=pop)
            if report:
                print(f">>> Created Island {i+1} using {algos[i]}")
        archi.wait_check()

        return archi

    @staticmethod
    def extract_archipelago_problem(archi: pygmo.archipelago, i=0) -> pygmo.problem:
        """Extracts the user defined problem from an archipelago, implemented as pygmo.problem(UDproblem).

        Parameters
        ----------
        archi : pygmo.archipelago
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
    def get_estimates_from_archipelago(archi: pygmo.archipelago) -> Tuple[dict, float]:
        """Extracts the best estimates and corresponding value of the objective function from an archipelago object.

        Parameters
        ----------
        archi : pygmo.archipelago
            The evolved archipelago.

        Returns
        -------
        Tuple[dict, float]
            Dictionary of best estimates according to the smallest loss value.
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
        }, best_loss

    @staticmethod
    def get_archipelago_results(
        archi: pygmo.archipelago, estimation_info: PygmoEstimationInfo
    ) -> Tuple[dict, PygmoEstimationInfo]:
        """Extracts the results of an evolved archipelago and updates additional estimation info.

        Parameters
        ----------
        archi : pygmo.archipelago
            The evolved archipelago.
        estimation_info : PygmoEstimationInfo
            Additional information about the archipelago before evolution(s).

        Returns
        -------
        Tuple[dict, PygmoEstimationInfo]
            Dictionary of best estimates according to the smallest loss value.
            Updated additional information about the evolved archipelago containing the archipelago itself.
        """
        estimates, loss = PygmoHelpers.get_estimates_from_archipelago(archi)

        estimation_info.loss = loss
        estimation_info.archi = archi

        return estimates, estimation_info
