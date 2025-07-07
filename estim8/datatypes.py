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
This model defines the datatypes used to store simulated and experimental data in the `estim8` package.
Further it implements methods for calculation of discrepancy measures or statistical likelhoods.
"""
from typing import Dict, List, Literal, get_args

import numpy as np
import pandas as pd
from numpy.core.multiarray import array as array

from . import error_models


class Constants:
    SINGLE_ID = None
    VALID_METRICS = Literal["SS", "WSS", "negLL"]


class TimeSeries:
    """
    A class to represent a time series.

    Attributes
    ----------
    name : str
        The name of the time series.
    timepoints : np.array
        The time points of the time series.
    values : np.array
        The values of the time series.
    replicate_ID : str
        The replicate ID of the time series.

    Methods
    -------
    _equal_shapes(array1, array2)
        Checks if two arrays have equal shapes.
    _get_mask()
        Creates a mask for non-NaN values.
    drop_nans()
        Drops NaN values from timepoints and values.
    """

    def __init__(
        self,
        name: str,
        timepoints: np.array,
        values: np.array,
        replicate_ID: str | None = Constants.SINGLE_ID,
    ):
        """
        Constructs all the necessary attributes for the TimeSeries object.

        Parameters
        ----------
        name : str
            The name of the time series.
        timepoints : np.array
            The time points of the time series.
        values : np.array
            The values of the time series.
        replicate_ID : str, optional
            The replicate ID of the time series (default is Constants.SINGLE_ID).
        """
        self.name = name
        self.replicate_ID = replicate_ID
        self.timepoints = timepoints
        self.values = values
        self._equal_shapes(timepoints, values)
        self.drop_nans()

    def _equal_shapes(self, array1: np.ndarray, array2: np.ndarray) -> None:
        """
        Checks if two arrays have equal shapes.

        Parameters
        ----------
        array1 : np.array
            The first array.
        array2 : np.array
            The second array.

        Raises
        ------
        ValueError
            If the shapes of the arrays are not equal.
        """
        if not array1.shape == array2.shape:
            raise ValueError(
                f"Arrays in a {self.__class__} must have equal shapes, got {array1.shape} and {array2.shape}."
            )

    def _get_mask(self):
        """
        Creates a mask for non-NaN values.
        """
        mask_timpoints = ~np.isnan(self.timepoints)
        mask_values = ~np.isnan(self.values)
        self._joint_mask = mask_timpoints * mask_values

    def drop_nans(self):
        """
        Drops NaN values from timepoints and values.
        """
        if not hasattr(self, "joint_mask"):
            self._get_mask()
        self.timepoints = self.timepoints[self._joint_mask]
        self.values = self.values[self._joint_mask]

    def __repr__(self):
        """
        Returns a string representation of the TimeSeries object.

        Returns
        -------
        str
            A string representation of the TimeSeries object.
        """
        return f"{self.__class__.__name__}(name={self.name}, replicate_ID={self.replicate_ID})"


class ModelPrediction(TimeSeries):
    """
    A class to represent model predictions, inheriting from TimeSeries.

    Methods
    -------
    interpolate(measurement_timepoints)
        Interpolates the model predictions to the given measurement timepoints.
    """

    def interpolate(self, measurement_timepoints: np.array) -> TimeSeries:
        """
        Interpolates the model predictions to the given measurement timepoints.

        Parameters
        ----------
        measurement_timepoints : np.array
            The time points at which to interpolate the model predictions.

        Returns
        -------
        TimeSeries
            The interpolated model predictions.
        """
        return np.interp(measurement_timepoints, self.timepoints, self.values)


class Simulation:
    """
    A class to represent a simulation.

    Attributes
    ----------
    model_predictions : list of ModelPrediction
        The model predictions for the simulation.
    replicate_ID : str
        The replicate ID of the simulation.

    Methods
    -------
    __getitem__(name)
        Gets the model prediction with the given name.
    """

    def __init__(
        self,
        simulation: Dict[str, np.array],
        replicate_ID: str | None = Constants.SINGLE_ID,
    ) -> None:
        """
        Constructs all the necessary attributes for the Simulation object.

        Parameters
        ----------
        simulation : dict of str to np.array
            The simulation data.
        replicate_ID : str, optional
            The replicate ID of the simulation (default is Constants.SINGLE_ID).
        """
        self.model_predictions = [
            ModelPrediction(
                observable, simulation["time"], simulation[observable], replicate_ID
            )
            for observable in simulation
            if not observable == "time"
        ]
        self.replicate_ID = replicate_ID

    def __getitem__(self, name):
        """
        Gets the model prediction with the given name.

        Parameters
        ----------
        name : str
            The name of the model prediction.

        Returns
        -------
        ModelPrediction
            The model prediction with the given name.

        Raises
        ------
        KeyError
            If no model prediction with the given name is found.
        """
        for model_predictions in self.model_predictions:
            if model_predictions.name == name:
                return model_predictions
        raise KeyError(f"No ModelPrediction with name {name} contained in Simulation")


class Measurement(TimeSeries):
    """
    A class to represent a measurement, inheriting from TimeSeries.

    Attributes
    ----------
    errors : np.array
        The errors of the measurement.
    error_model : error_models.BaseErrorModel
        The error model used for the measurement.

    Methods
    -------
    get_loss(model_prediction, metric)
        Calculates the loss between the measurement and the model prediction.
    get_sampling(n_samples)
        Generates samples from the measurement.
    """

    def __init__(
        self,
        name,
        timepoints: np.array,
        values: np.array,
        replicate_ID: str | None = Constants.SINGLE_ID,
        errors: np.array = None,
        error_model: error_models.BaseErrorModel = error_models.LinearErrorModel(),
    ):
        """
        Constructs all the necessary attributes for the Measurement object.

        Parameters
        ----------
        name : str
            The name of the measurement.
        timepoints : np.array
            The time points of the measurement.
        values : np.array
            The values of the measurement.
        replicate_ID : str, optional
            The replicate ID of the measurement (default is Constants.SINGLE_ID).
        errors : np.array, optional
            The errors of the measurement (default is None).
        error_model : error_models.BaseErrorModel, optional
            The error model used for the measurement (default is error_models.LinearErrorModel()).
        """
        super().__init__(name, timepoints, values, replicate_ID)

        self.error_model = error_model
        if errors is None:
            self.errors = self.error_model.generate_error_data(self.values)
        else:
            self.errors = errors[self._joint_mask]

    @property
    def errors(self):
        """
        Gets the errors of the measurement.

        Returns
        -------
        np.array
            The errors of the measurement.
        """
        return self._errors

    @errors.setter
    def errors(self, values):
        """
        Sets the errors of the measurement.

        Parameters
        ----------
        values : np.array
            The errors of the measurement.
        """
        self._equal_shapes(self.values, values)
        self._errors = values

    def get_loss(
        self, model_prediction: ModelPrediction, metric: Constants.VALID_METRICS = "SS"
    ) -> float:
        """
        Calculates the loss between the measurement and the model prediction.

        Parameters
        ----------
        model_prediction : ModelPrediction
            The model prediction.
        metric : str, optional
            The metric to use for calculating the loss (default is "SS").

        Returns
        -------
        float
            The loss between the measurement and the model prediction.

        Raises
        ------
        NotImplementedError
            If the metric is not supported.
        """
        if metric not in get_args(Constants.VALID_METRICS):
            raise NotImplementedError(
                f"{metric} is not supported. Choose one of the following \n 'SS', 'WSS', 'negLL'"
            )

        # interpolate prediction to timepoints of measurement
        y_pred = model_prediction.interpolate(self.timepoints)

        if metric == "SS":
            loss = np.square(y_pred - self.values)

        elif metric == "WSS":
            loss = np.square((y_pred - self.values) / self.errors)

        elif metric == "negLL":
            loss = -1 * self.error_model.error_distribution.logpdf(
                x=y_pred,
                loc=self.values,
                scale=self.errors,
                **self.error_model.error_distribution_kwargs,
            )

        return np.sum(loss)

    def get_sampling(self, n_samples) -> List["Measurement"]:
        """
        Generates samples from the measurement.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        list of Measurement
            The generated samples.
        """
        return [
            Measurement(
                self.name,
                self.timepoints,
                sampled_values,
                self.replicate_ID,
                self.errors,
                self.error_model,
            )
            for sampled_values in self.error_model.get_sampling(
                self.values, self.errors, n_samples
            )
        ]


class Experiment:
    """
    A class to represent an experiment.

    Attributes
    ----------
    measurements : list of Measurement
        The measurements of the experiment.
    replicate_ID : str
        The replicate ID of the experiment.
    observation_mapping : dict of str to str
        The mapping of observation names to measurement names.

    Methods
    -------
    __getitem__(name)
        Gets the measurement with the given name.
    getbysimkey(name)
        Gets the measurement by the simulation key.
    calculate_loss(simulation, metric)
        Calculates the loss between the experiment and the simulation.
    generate_mc_samples(n_samples)
        Generates Monte Carlo samples from the experiment.
    """

    def __init__(
        self,
        measurements: List[Measurement] | pd.DataFrame,
        replicate_ID=Constants.SINGLE_ID,
        errors: pd.DataFrame = None,
        error_model: error_models.BaseErrorModel = error_models.LinearErrorModel(),
        observation_mapping: Dict[str, str] | None = None,
    ):
        """
        Constructs all the necessary attributes for the Experiment object.

        Parameters
        ----------
        measurements : list of Measurement or pd.DataFrame
            The measurements of the experiment.
        replicate_ID : str, optional
            The replicate ID of the experiment (default is Constants.SINGLE_ID).
        errors : pd.DataFrame, optional
            The errors of the measurements (default is None).
        error_model : error_models.BaseErrorModel, optional
            The error model used for the measurements (default is error_models.LinearErrorModel()).
        observation_mapping : dict of str to str, optional
            The mapping of observation names to measurement names (default is None).
        """
        self.replicate_ID = replicate_ID
        if isinstance(measurements, pd.DataFrame):
            if isinstance(errors, pd.DataFrame):
                if measurements.shape != errors.shape:
                    raise ValueError(
                        f"Measurements shape {measurements.shape} does not match shape of errors dataframe {errors.shape}"
                    )

                self.measurements = [
                    Measurement(
                        name=column,
                        timepoints=measurements.index,
                        values=measurements[column].to_numpy(),
                        replicate_ID=replicate_ID,
                        errors=errors[column].to_numpy(),
                        error_model=error_model,
                    )
                    for column in measurements.columns
                ]
            else:
                self.measurements = [
                    Measurement(
                        name=column,
                        timepoints=measurements.index,
                        values=measurements[column].to_numpy(),
                        replicate_ID=replicate_ID,
                        errors=None,
                        error_model=error_model,
                    )
                    for column in measurements.columns
                ]

        elif all(
            [isinstance(measurement, Measurement) for measurement in measurements]
        ):
            self.measurements = measurements
        else:
            raise ValueError(
                f"Measurements must be List[Measurement] or pd.DataFrame, not {type(measurements)}."
            )

        self.observation_mapping = {
            measurement.name: measurement.name for measurement in self.measurements
        }

        if observation_mapping is not None:
            self.observation_mapping.update(observation_mapping)
        if not all(
            [
                key in [measurement.name for measurement in self.measurements]
                for key in self.observation_mapping.keys()
            ]
        ):
            raise ValueError(
                f"Keys {self.observation_mapping.keys()} of observation mapping dont match names {[measurement.name for measurement in self.measurements]} of measurements!"
            )

    def __getitem__(self, name):
        """
        Gets the measurement with the given name.

        Parameters
        ----------
        name : str
            The name of the measurement.

        Returns
        -------
        Measurement
            The measurement with the given name.

        Raises
        ------
        KeyError
            If no measurement with the given name is found.
        """
        for measurement in self.measurements:
            if measurement.name == name:
                return measurement
        raise KeyError(f"No Measurement with name {name} contained in Experiment")

    def getbysimkey(self, name):
        """
        Gets the measurement by the simulation key.

        Parameters
        ----------
        name : str
            The simulation key.

        Returns
        -------
        Measurement
            The measurement with the given simulation key.
        """
        for obs_data, obs_sim in self.observation_mapping.items():
            if name == obs_sim:
                return self.__getitem__(obs_data)

    def calculate_loss(
        self, simulation: Simulation, metric: Constants.VALID_METRICS = "SS"
    ) -> float:
        """
        Calculates the loss between the experiment and the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation.
        metric : str, optional
            The metric to use for calculating the loss (default is "SS").

        Returns
        -------
        float
            The loss between the experiment and the simulation.
        """
        return np.sum(
            [
                measurement.get_loss(
                    simulation[self.observation_mapping[measurement.name]], metric
                )
                for measurement in self.measurements
            ]
        )

    def generate_mc_samples(self, n_samples: int) -> List["Experiment"]:
        """
        Generates Monte Carlo samples from the experiment.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        list of Experiment
            The generated samples.
        """
        resampling = []
        for measurement in self.measurements:
            resampling.append(measurement.get_sampling(n_samples))

        return [
            Experiment(
                _measurements,
                self.replicate_ID,
                observation_mapping=self.observation_mapping,
            )
            for _measurements in zip(*resampling)
        ]
