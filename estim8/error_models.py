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
This module defines the blueprint for modeling measurement noise.
"""

import abc
from typing import Any, Dict, List

import numpy as np
import scipy.stats


class BaseErrorModel(abc.ABC):
    """
    Abstract base class for error modeling.
    """

    def __init__(
        self,
        error_distribution: scipy.stats.rv_continuous = scipy.stats.norm,
        error_distribution_kwargs: dict = {},
    ):
        """
        Creates an ErrorModel object

        Parameters
        ----------
        error_distribution : scipy.stats.rv_continuous
            Defines the distribution of the measurement noise. Is used for calculating the Likelihood, as well as generating random samples of data.
        error_distribution_kwargs : dict
            Keyword arguments for calling error_distribution.rvs()
        """
        self.error_distribution = error_distribution
        self.error_distribution_kwargs = error_distribution_kwargs

    @abc.abstractmethod
    def generate_error_data(self, values: np.array) -> np.array:
        """
        Abstract class method for calculating errors of experimental data given the datapoints.

        Parameters
        ----------
        values : np.array
            Values of the data on which to apply error model

        Returns
        -------
        errors : np.array
            Error calculated according to specified model
        """
        return

    def get_sampling(
        self, values: np.array, errors: np.array, n_samples: int
    ) -> List[np.array]:
        """
        Resamples values of data given the class instance error_distribution.

        Parameters
        ----------
        values : np.array
            The values to resample.
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        resampling : List[np.array]
            The generated Monte Carlo samples of values.
        """

        resampling = []
        # apply distribution
        for _ in range(n_samples):
            resampling.append(
                self.error_distribution.rvs(
                    loc=values,
                    scale=errors,
                    **self.error_distribution_kwargs,
                    size=values.shape,
                )
            )

        return resampling


class LinearErrorModel(BaseErrorModel):
    """
    An ErrorModel with linear relationship between measurement value and noise given by;

    .. math:: \sigma = slope \cdot y + offset

    """

    def __init__(
        self,
        slope: float = 0,
        offset: float = 0,
        error_distribution: scipy.stats.rv_continuous = scipy.stats.norm,
        error_distribution_kwargs: dict = {},
    ):
        """
        Creates a LinearErrorModel object.

        Parameters
        ----------
        slope : float, optional
            The relative noise of measurements, by default 0.02
        offset : float, optional
            The absolute noise of measurements, by default 1e-6
        error_distribution : scipy.stats.rv_continuous, optional
            Defines the distribution of the measurement noise. Is used for calculating the Likelihood, as well as generating random samples of data.
            By default scipy.stats.norm.
        error_distribution_kwargs : dict, optional
            Keyword arguments for calling error_distribution.rvs(), by default {}
        """
        self.error_model_params = {"slope": slope, "offset": offset}
        super().__init__(error_distribution, error_distribution_kwargs)

    @property
    def error_model_params(self) -> dict:
        """Error model parameters given by slope and offset."""
        return self._error_model_params

    @error_model_params.setter
    def error_model_params(self, value: Dict[str, float]):
        """
        Setter method for Error model parameters

        Parameters
        ----------
        value : Dict[str, float]
            A dictionary with slope and offset parameters.

        Raises
        ------
        IOError
            If the dictionary keys are not 'slope' and 'offset'.
        """
        if set(value.keys()) != set(["slope", "offset"]):
            raise IOError(
                """"Please pass a dict with keys slope and offset, such that they fit the equation [values * slope + offset] or choose another error model.
                """
            )
        self._error_model_params = value

    def generate_error_data(self, values: np.array) -> np.array:
        """
        Generates error data based on the linear error model.

        Parameters
        ----------
        values : np.array
            Values of the data on which to apply error model.

        Returns
        -------
        errors : np.array
            Error calculated according to the linear model.
        """
        return (
            self.error_model_params["slope"] * values
            + self.error_model_params["offset"]
        )
