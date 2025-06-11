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
"""This module implements the model classes for the Estim8 package.
"""
import abc
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal

import fmpy
import numpy as np

from .datatypes import Constants, Simulation
from .utils import ModelHelpers

SINGLE_ID = Constants.SINGLE_ID


class Estim8Model(abc.ABC):
    def __init__(self, default_parameters={}, r_tol=1e-4):
        """
        Initialize the Estim8Model.

        Parameters
        ----------
        default_parameters : dict, optional
            Default parameters for the model, by default {}.
        r_tol : float, optional
            Relative tolerance for the model, by default 1e-4.
        """
        self.r_tol = r_tol

        # get model parameters and observables
        self.retrieve_variables()
        # and update with default parameters
        self.parameters.update(default_parameters)

    @abc.abstractmethod
    def retrieve_variables(self):
        """
        Retrieve model parameters and observables.
        """
        self.parameters: Dict[str, float]
        self.observables: List[str]

    @abc.abstractmethod
    def simulate(
        self,
        t0: float,
        t_end: float,
        stepsize: float,
        parameters: Dict[str, float] = {},
        observe: List[str] | None = None,
        replicate_ID: str | None = SINGLE_ID,
    ) -> Simulation:
        """
        Simulate the model.

        Parameters
        ----------
        t0 : float
            Start time of the simulation.
        t_end : float
            End time of the simulation.
        stepsize : float
            Time step size for the simulation.
        parameters : dict
            Parameters for the simulation.
        observe : list
            List of observables to record during the simulation.
        replicate_ID : str, optional
            Replicate ID for the simulation, by default SINGLE_ID.

        Returns
        -------
        Simulation
            The simulation results.
        """


class FmuModel(Estim8Model):
    def __init__(
        self,
        path: str,
        fmi_type: Literal["ModelExchange", "CoSimulation"] = "ModelExchange",
        default_parameters: dict = {},
        r_tol=1e-4,
    ):
        """
        Initialize the FmuModel.

        Parameters
        ----------
        path : str
            Path to the FMU file.
        fmi_type : Literal["ModelExchange", "CoSimulation"], optional
            Type of FMI, by default "ModelExchange".
        default_parameters : dict, optional
            Default parameters for the model, by default {}.
        r_tol : float, optional
            Relative tolerance for the model, by default 1e-4.
        """
        self.path = Path(path).resolve()

        # set fmi type and thereby load fmu
        self.fmi_type = fmi_type
        super().__init__(default_parameters=default_parameters, r_tol=r_tol)

    def cleanup(self):
        """
        Clean up temporary files and directories.
        """
        try:
            self.freeInstance()
        except:
            pass

        # Clean up unzip directory
        if hasattr(self, "_unzip_dir") and self._unzip_dir:
            try:
                shutil.rmtree(self._unzip_dir)
            except:
                pass

    def __getstate__(self):
        """
        Return state for pickling.

        Returns
        -------
        dict
            State dictionary for pickling.
        """
        state = self.__dict__.copy()
        # Remove unpicklable entries
        state.pop("_fmu", None)
        state.pop("_model_description", None)
        state.pop("_unzip_dir", None)
        return state

    def __setstate__(self, state):
        """
        Restore state when unpickling.

        Parameters
        ----------
        state : dict
            State dictionary for unpickling.
        """
        self.__dict__.update(state)
        # Reinstantiate the FMU
        self.instantiate_fmu()

    def instantiate_fmu(self):
        """
        Instantiate the FMU from the copied file.
        """
        # Create a copy of the FMU file in a temporary directory
        self._model_description = fmpy.read_model_description(self.path)
        self._unzip_dir = fmpy.extract(self.path)
        self._fmu = fmpy.instantiate_fmu(
            self._unzip_dir, self._model_description, self._fmi_type
        )

    def __del__(self):
        """
        Destructor to ensure cleanup.
        """
        self.cleanup()

    @property
    def fmi_type(self):
        """
        Get the FMI type.

        Returns
        -------
        str
            The FMI type.
        """
        return self._fmi_type

    @fmi_type.setter
    def fmi_type(self, value: Literal["ModelExchange", "CoSimulation"]):
        """
        Set the FMI type.

        Parameters
        ----------
        value : Literal["ModelExchange", "CoSimulation"]
            The FMI type.

        Raises
        ------
        ValueError
            If the value is not a valid FMI type.
        """
        if value not in ["ModelExchange", "CoSimulation"]:
            raise ValueError(
                f"{value} is not a valid FMI type. Choose one of 'CoSimulation' or 'Modelexchange'"
            )
        self._fmi_type = value
        self.instantiate_fmu()

    def freeInstance(self):
        """
        Free the FMU instance.
        """
        self._fmu.freeInstance()

    def retrieve_variables(self):
        """
        Retrieve model parameters and observables.
        """
        self.parameters, self.observables = {}, []
        for variable in self._model_description.modelVariables:
            if variable.initial == "exact":
                self.parameters[variable.name] = float(variable.start)
            else:
                self.observables.append(variable.name)

    def simulate(
        self,
        t0: float,
        t_end: float,
        stepsize: float,
        parameters: dict[str, float] = {},
        observe: list[str] | None = None,
        replicate_ID: str | None = SINGLE_ID,
        r_tol: float | None = None,
        solver: Literal["CVode", "Euler"] = "CVode",
    ) -> Simulation:
        """
        Simulate the FMU model.

        Parameters
        ----------
        t0 : float
            Start time of the simulation.
        t_end : float
            End time of the simulation.
        stepsize : float
            Time step size for the simulation.
        parameters : dict, optional
            Parameters for the simulation, by default {}.
        observe : list, optional
            List of observables to record during the simulation, by default None.
        r_tol : float, optional
            Relative tolerance for the simulation, by default None.
        solver : Literal["CVode", "Euler"], optional
            Solver to use for the simulation, by default "CVode".
        replicate_ID : str, optional
            Replicate ID for the simulation, by default SINGLE_ID.

        Returns
        -------
        Simulation
            The simulation results.
        """
        self._fmu.reset()

        params = self.parameters.copy()
        params.update(parameters)

        if not r_tol:
            r_tol = self.r_tol

        if not observe:
            observe = self.observables

        sim_raw = fmpy.simulate_fmu(
            filename=self._unzip_dir,
            start_time=t0,
            stop_time=t_end,
            output_interval=stepsize,
            relative_tolerance=r_tol,
            output=observe,
            start_values=params,
            fmi_type=self._fmi_type,
            validate=True,
            solver=solver,
            debug_logging=True,
            model_description=self._model_description,
            fmu_instance=self._fmu,
        )

        sim = {"time": sim_raw["time"]}
        for obs in observe:
            sim[obs] = np.array(sim_raw[obs])

        return Simulation(sim, replicate_ID=replicate_ID)
