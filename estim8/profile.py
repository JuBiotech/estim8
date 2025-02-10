import re
from typing import List, Literal

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import chi2

from .optimizers import Optimization


def calculate_negll_thresshold(alpha: float = 0.05, df: int = 1, mle_negll: float = 0):
    quantile = chi2.ppf(1 - alpha, df)
    return mle_negll + quantile / 2


class ProfileSampler:
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
        max_steps: int = None,
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
        """Take a next ficed step along the profile

        Returns
        -------
        float
            The next value of the parameter
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
        # update the objective functions parameter mapping with the new value of the parameter
        self.optimizer.objective
        self.optimizer.objective.parameter_mapping.set_parameter(self.parameter, value)
        # Parse current task ID using regex
        current_id = self.optimizer.task_id
        match = re.match(r"pl_job_(\d+)_(\d+)", current_id)
        job_num = match.group(1)
        step_num = int(match.group(2)) + 1

        # Update task ID with new step number
        self.optimizer.task_id = f"pl_job_{job_num}_{step_num}"

    def next_pl_sample(self):
        # get the next sample point
        next_value = self.next_step()

        # update the optimizer
        self.update_optimizer_objective(next_value)

        # calculate pl
        _, info = self.optimizer.optimize()

        # add result to samples
        self.samples = np.vstack((self.samples, np.array([next_value, info["fun"]])))

        if info["fun"] > self.negll_threshold:
            self.finished = True

    def walk_profile(self):
        while not self.finished:
            self.next_pl_sample()

        return self.samples, self.parameter


def approximate_confidence_interval(xvalues, negll_values, threshold):
    """
    Approximate the confidence interval from the profile likelihood results.

    Parameters
    ----------
    xvalues : np.ndarray
        The x values of the profile likelihood.
    negll_values : np.ndarray
        The negative log likelihood values of the profile likelihood.
    threshold : float
        The threshold of the profile likelihood.

    Returns
    -------
    float
        The lower bound of the confidence interval.
    float
        The upper bound of the confidence interval.
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
