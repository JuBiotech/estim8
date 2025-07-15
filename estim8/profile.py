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
"""This module implements functionality for profiling likelihoods and likelihood based confidence intervals.
"""
import re
from typing import List, Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chi2

from .objective import Objective
from .optimizers import Optimization


def calculate_negll_thresshold(alpha: float = 0.05, df: int = 1, mle_negll: float = 0):
    """Calculate the negative log-likelihood threshold for profile likelihood.

    Parameters
    ----------
    alpha : float, optional
        Significance level, by default 0.05
    df : int, optional
        Degrees of freedom, by default 1
    mle_negll : float, optional
        Maximum likelihood estimate of negative log-likelihood, by default 0

    Returns
    -------
    float
        The threshold value for the negative log-likelihood
    """
    quantile = chi2.ppf(1 - alpha, df)
    return mle_negll + quantile / 2


class ProfileSampler:
    """Sample points along a profile likelihood curve.

    Parameters
    ----------
    parameter : str
        Name of the parameter to profile
    mle : float
        Maximum likelihood estimate of the parameter
    mle_negll : float
        Negative log-likelihood at the MLE
    negll_threshold : float
        Threshold value for the negative log-likelihood
    optimizer : Optimization
        Optimizer instance to use for likelihood calculations
    bounds : List[float]
        Parameter bounds [lower, upper]
    direction : Literal[-1, 1]
        Direction to walk the profile (1 for positive, -1 for negative)
    stepsize : float, optional
        Relative step size for parameter updates, by default 0.02
    max_steps : int, optional
        Maximum number of steps to take, by default None, in which case the sampler proceeds until the negll threshold is reached.
    """

    def __init__(
        self,
        parameter,
        mle: float,
        mle_negll: float,
        negll_threshold: float,
        optimizer: Optimization,
        bounds: List[float],
        direction: Literal[-1, 1],
        stepsize=0.02,
        max_steps: int | None = None,
    ):
        self.parameter = parameter
        self.mle = mle
        self.direction = direction
        self.stepsize = stepsize
        self.optimizer = optimizer
        self.bounds = bounds
        self.negll_threshold = negll_threshold
        self.max_steps = max_steps

        # initialize the samples
        self.samples = np.ndarray(shape=(0, 2))
        self.samples = np.vstack((self.samples, np.array([mle, mle_negll])))

        self.finished = False

    def next_step(self):
        """Take a next fixed step along the profile.

        Returns
        -------
        float
            The next value of the parameter to evaluate
        """
        next_step = self.samples[-1][0] + self.direction * self.stepsize * (
            self.mle if self.mle != 0 else 1
        )

        # check if bounds are violated
        if not self.bounds[0] <= next_step <= self.bounds[1]:
            self.finished = True
            next_step = self.bounds[-self.direction]

        # check if max_steps is reached
        if self.max_steps is not None:
            if len(self.samples[0]) >= self.max_steps - 1:
                self.finished = True

        return next_step

    def update_optimizer_objective(self, value: float):
        """Update the optimizer's objective function with a new parameter value and set a new task_id.

        Parameters
        ----------
        value : float
            New value for the profiled parameter
        """
        # update the objective functions parameter mapping with the new value of the parameter
        if not isinstance(self.optimizer.objective, Objective):
            raise TypeError(
                "Optimizer's objective must be an instance of Objective to update parameter mapping."
            )
        self.optimizer.objective.parameter_mapping.set_parameter(self.parameter, value)
        # Parse current task ID using regex
        current_id = self.optimizer.task_id
        match = re.match(r"pl_job_(\d+)_(\d+)", current_id)

        if match is None:
            raise ValueError(
                f"Optimizer task_id is not set properly. Cannot get task_id from {current_id}."
            )
        else:
            job_num = match.group(1)
            step_num = int(match.group(2)) + 1

        # Update task ID with new step number
        self.optimizer.task_id = f"pl_job_{job_num}_{step_num}"

    def next_pl_sample(self):
        """Calculate the next profile likelihood sample point.

        Updates the internal samples array with the new point and checks if
        the profile likelihood threshold has been exceeded.
        """
        # get the next sample point
        next_value = self.next_step()

        # update the optimizer
        self.update_optimizer_objective(next_value)

        # calculate pl
        _, info = self.optimizer.optimize()

        fun = getattr(info, "fun")

        # add result to samples
        self.samples = np.vstack((self.samples, np.array([next_value, fun])))

        if fun > self.negll_threshold:
            self.finished = True

        # del info object with eventally contains archipelago object
        del info

    def walk_profile(self):
        """Walk the complete profile likelihood curve.

        Returns
        -------
        tuple
            Tuple containing:
            - numpy.ndarray: Array of sample points [(parameter_value, negll), ...]
            - str: Name of the profiled parameter
        """
        while not self.finished:
            self.next_pl_sample()

        return self.samples, self.parameter


def approximate_confidence_interval(xvalues, negll_values, threshold):
    """Approximate the confidence interval from profile likelihood results.

    Parameters
    ----------
    xvalues : numpy.ndarray
        The parameter values of the profile likelihood curve
    negll_values : numpy.ndarray
        The negative log-likelihood values of the profile likelihood curve
    threshold : float
        The threshold value for the confidence interval

    Returns
    -------
    tuple
        Tuple containing:
        - float: Lower bound of the confidence interval
        - float: Upper bound of the confidence interval

    Raises
    ------
    ValueError
        If confidence interval bounds cannot be found
    """
    # Interpolate to find more precise crossing points
    f = interp1d(xvalues, negll_values - threshold, kind="cubic")

    # Create finer grid for interpolation
    x_fine = np.linspace(xvalues.min(), xvalues.max(), 1000)
    y_fine = f(x_fine)

    # Find zero crossings
    zero_crossings = np.where(np.diff(np.signbit(y_fine)))[0]
    del f

    if len(zero_crossings) >= 2:
        lower = x_fine[zero_crossings[0]]
        upper = x_fine[zero_crossings[-1]]
        return (lower, upper)
    else:
        raise ValueError(
            "Could not find confidence interval bounds - profile may be too flat or not enough points"
        )
