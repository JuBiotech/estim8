import numpy as np

from estim8.datatypes import Simulation
from estim8.models import Estim8Model


class MockModel(Estim8Model):
    def __init__(self, default_parameters={}, r_tol=0.0001):
        super().__init__(default_parameters, r_tol)

    def retrieve_variables(self):
        self.parameters = {"slope": 1, "offset": 0}
        self.observables = ["y"]

    def simulate(
        self,
        t0: float,
        t_end: float,
        stepsize: float,
        parameters: dict = dict(),
        observe: list | None = None,
        replicate_ID: str | None = None,
    ):
        params = self.parameters.copy()
        params.update(parameters)

        t = t0 + np.arange(0, t_end) * stepsize
        y = params["slope"] * t + params["offset"]
        return Simulation({"time": t, "y": y})
