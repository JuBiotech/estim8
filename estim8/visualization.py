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
"""This module implements visualization functions for the estim8 package.
"""
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn

from .datatypes import Constants, Experiment, Measurement, ModelPrediction, Simulation
from .estimator import Estimator
from .models import Estim8Model
from .profile import approximate_confidence_interval, calculate_negll_thresshold
from .utils import ModelHelpers

plt.style.use("seaborn-v0_8-colorblind")
from itertools import cycle
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import scipy

SINGLE_ID = Constants.SINGLE_ID

axis_fontsize = 15
rel_figure_width = 6
rel_fig_height = 4


def plot_simulation(
    simulation: Simulation,
    observe: List[str] | None = None,
    experiment: Experiment | None = None,
):
    """
    Plot the simulation results.

    Parameters
    ----------
    simulation : Simulation
        The simulation results to plot.
    observe : List[str], optional
        List of observables to plot, by default None.
    experiment : Experiment, optional
        The experiment data to plot alongside the simulation, by default None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    if observe is None:
        obs_list = [
            model_prediction.name for model_prediction in simulation.model_predictions
        ]
    else:
        obs_list = observe

    # Define subplot grid:
    ncols = min(3, len(obs_list))
    nrows = int(len(obs_list) / ncols) + bool(len(obs_list) % ncols)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(rel_figure_width * ncols, rel_fig_height * nrows),
    )

    # ensure axes is array
    axes = np.asarray(axes)

    # turn off empty axes by default
    for ax in axes.flat:
        ax.set_axis_off()

    for ax, observable, color in zip(
        axes.flat, obs_list, cycle(seaborn.color_palette("colorblind", len(obs_list)))
    ):
        # turn on used axis
        ax.set_axis_on()

        if experiment is not None:
            if observable in experiment.observation_mapping.values():
                _measurement = experiment.getbysimkey(observable)
                plot_measurement(ax=ax, measurement=_measurement, color=color)

        plot_model_prediction(
            ax=ax,
            model_prediction=simulation[observable],
            color=color,
        )

        ax.set_xlabel("time", fontsize=axis_fontsize)
        ax.legend()
    if simulation.replicate_ID is not None:
        fig.suptitle(f"replicate ID: {simulation.replicate_ID}")

    fig.tight_layout()
    return fig


def plot_model_prediction(ax, model_prediction: ModelPrediction, **kwargs):
    """
    Plot the model prediction.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    model_prediction : ModelPrediction
        The model prediction to plot.
    **kwargs
        Additional keyword arguments for the plot.
    """
    plotting_kwargs = dict(color="black", linestyle="-", label=str(model_prediction))
    plotting_kwargs.update(kwargs)
    ax.plot(model_prediction.timepoints, model_prediction.values, **plotting_kwargs)


def plot_measurement(ax, measurement: Measurement, **kwargs):
    """
    Plot the measurement.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    measurement : Measurement
        The measurement to plot.
    **kwargs
        Additional keyword arguments for the plot.
    """
    plotting_kwargs = dict(
        color="black",
        fmt=".",
        ecolor="black",
        markeredgecolor="black",
        label=str(measurement),
    )
    plotting_kwargs.update(kwargs)

    ax.errorbar(
        x=measurement.timepoints,
        y=measurement.values,
        yerr=measurement.errors,
        **plotting_kwargs,
    )


def plot_estimates(estimates: dict, estimator: Estimator, only_measured: bool = False):
    """
    Plot the estimates.

    Parameters
    ----------
    estimates : dict
        The parameter estimates.
    estimator : Estimator
        The estimator object.
    only_measured : bool, optional
        Whether to plot only the measured observables, by default False.

    Returns
    -------
    matplotlib.figure.Figure or list of matplotlib.figure.Figure
        The figure(s) containing the plots.
    """
    t0, t_end, stepsize = estimator.t

    figures = []

    for rid, data in estimator.data.items():
        replicate_parameters = estimator.parameter_mapping.replicate_handling(
            parameters=estimates, replicate_ID=rid
        )

        simulation = estimator.model.simulate(
            t0=t0,
            t_end=t_end,
            stepsize=stepsize,
            replicate_ID=rid,
            parameters=replicate_parameters,
        )

        if only_measured:
            observables = list(data.observation_mapping.values())
        else:
            observables = None

        figures.append(
            plot_simulation(simulation=simulation, observe=observables, experiment=data)
        )

    if len(figures) > 1:
        return figures
    else:
        return figures[0]


def plot_heatmap(
    mc_samples: pd.DataFrame, thresholds: int = 5, show_vals: bool = False
):
    """
    Plotting the correlations for parameter results of a Monte Carlo Sampling as a heatmap. The threshold defines
    the resolution of values, i.e. it defines a color map with discrete increments. The larger the thresholds
    argument is set, the finer is the resolution. Set show_vals True to write the exact correlation values into the
    plot.

    Parameters
    ----------
    mc_samples : pd.DataFrame
        Parameter information from Monte Carlo Sampling.
    thresholds : int, optional
        Number of increments for color map, by default 5.
    show_vals : bool, optional
        Displays exact values into the plot, by default False.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the heatmap.
    """
    # Check input
    if not isinstance(mc_samples, pd.DataFrame):
        try:
            mc_samples = pd.DataFrame(mc_samples)
        except:
            raise ValueError(
                f"data could not be converted into DataFrame. Please try passing a DataFrame instead"
            )

    # Compute Correlations
    corrs = mc_samples.corr()
    corrs.astype("float")

    fig, ax = plt.subplots(figsize=(22, 22))

    # Setup colors
    colors = []
    th = np.linspace(-1, 1, thresholds)
    for i in range(thresholds):
        if i < (thresholds / 2):
            colors.append(tuple([0.0, 0.0, 0.8, abs(th[i])]))
        else:
            colors.append(tuple([0.8, 0.0, 0.0, abs(th[i])]))

        cmap = mcolors.LinearSegmentedColormap.from_list("Custom", colors, len(colors))

    # Plot Heatmap
    fig = seaborn.heatmap(
        data=corrs,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=show_vals,
        cmap=cmap,
        linecolor="black",
        linewidths=0.1,
        ax=ax,
    )

    return fig


def plot_distributions(mc_samples: pd.DataFrame, ci_level: float = 0.95, kde=True):
    """
    Plot the distributions of the Monte Carlo samples.

    Parameters
    ----------
    mc_samples : pd.DataFrame
        The Monte Carlo samples.
    ci_level : int, optional
        The confidence interval level, by default 0.95.
    kde : bool, optional
        Whether to plot the kernel density estimate, by default True.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots.
    """
    # Define subplot grid:
    ncols = min(4, len(mc_samples.columns))
    nrows = int(len(mc_samples.columns) / ncols) + bool(len(mc_samples.columns) % ncols)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(rel_figure_width * ncols, rel_fig_height * nrows),
    )

    # ensure axes is array
    axes = np.asarray(axes)

    # turn off empty axes by default
    for ax in axes.flat:
        ax.set_axis_off()

    for ax, parameter, color in zip(
        axes.flat,
        mc_samples.columns,
        cycle(seaborn.color_palette("colorblind", len(mc_samples.columns))),
    ):
        ax.set_axis_on()
        values = mc_samples[parameter]
        seaborn.histplot(data=values, color=color, ax=ax, kde=kde, stat="density")

        # calculate Confidence interval
        mean_estimate = np.mean(values)
        std_estimate = np.std(values)
        z_score = scipy.stats.norm.ppf(1 - (1 - ci_level) / 2)

        ci_lower = mean_estimate - z_score * std_estimate
        ci_upper = mean_estimate + z_score * std_estimate

        ax.annotate(
            f"sampling mean: \n {mean_estimate:.3f}",
            xy=(0.3, 0.9),
            xycoords="axes fraction",
            fontweight="bold",
        )

        ax.hlines(
            y=ax.get_ylim()[1] * 0.015,
            xmin=ci_lower,
            xmax=ci_upper,
            color="black",
            linestyle="-",
            lw=2,
        )
        ax.text(
            ci_lower,
            ax.get_ylim()[1] * 0.025,
            f"{ci_lower:.3f}",
            color="black",
            ha="center",
            fontsize=12,
        )
        ax.text(
            ci_upper,
            ax.get_ylim()[1] * 0.025,
            f"{ci_upper:.3f}",
            color="black",
            ha="center",
            fontsize=12,
        )

    fig.tight_layout()
    return fig


def plot_pairs(
    mc_samples: pd.DataFrame, kind: Literal["scatter", "kde", "hist", "reg"] = "kde"
):
    """
    Plot pairwise relationships in the Monte Carlo samples.

    Parameters
    ----------
    mc_samples : pd.DataFrame
        The Monte Carlo samples.
    kind : str, optional
        The kind of plot to draw, by default "kde".

    Returns
    -------
    seaborn.axisgrid.PairGrid
        The pair grid containing the plots.
    """
    fig = seaborn.pairplot(mc_samples, kind=kind, corner=True)
    return fig


def plot_estimates_many(
    mc_samples: pd.DataFrame, estimator: Estimator, only_measured: bool = False
):
    """
    Plot the estimates for many Monte Carlo samples.

    Parameters
    ----------
    mc_samples : pd.DataFrame
        The Monte Carlo samples.
    estimator : Estimator
        The estimator object.
    only_measured : bool, optional
        Whether to plot only the measured observables, by default False.

    Returns
    -------
    matplotlib.figure.Figure or list of matplotlib.figure.Figure
        The figure(s) containing the plots.
    """
    if only_measured:
        observables = {
            rid: rep_data.observation_mapping.values()
            for rid, rep_data in estimator.data.items()
        }
    else:
        observables = {
            rid: estimator.model.observables for rid in estimator.replicate_IDs
        }

    t0, t_end, stepsize = estimator.t

    # run the simulations
    replicate_simulations: Dict[str | None, List[Simulation]] = {
        rid: [] for rid in estimator.replicate_IDs
    }

    for i in range(len(mc_samples)):
        sample = mc_samples.iloc[i].to_dict()
        replicate_parameters = {
            rID: estimator.parameter_mapping.replicate_handling(
                parameters=sample, replicate_ID=rID
            )
            for rID in estimator.replicate_IDs
        }

        for rID, parameters in replicate_parameters.items():
            replicate_simulations[rID].append(
                estimator.model.simulate(
                    t0=t0,
                    t_end=t_end,
                    stepsize=stepsize,
                    parameters=parameters,
                    replicate_ID=rID,
                    observe=observables[rID],
                )
            )

    figures = []

    for rID, simulations in replicate_simulations.items():
        ncols = min(3, len(observables[rID]))
        nrows = int(len(observables[rID]) / ncols) + bool(len(observables[rID]) % ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(rel_figure_width * ncols, rel_fig_height * nrows),
        )
        # ensure axes is array
        axes = np.asarray(axes)

        # turn off empty axes by default
        for ax in axes.flat:
            ax.set_axis_off()

        experiment = estimator.data[rID]

        for ax, observable, color in zip(
            axes.flat,
            observables[rID],
            seaborn.color_palette("colorblind", len(observables[rID])),
        ):
            ax.set_axis_on()

            # choose timepoints as they may have inhomogenous shapes to to event records in simulations
            timepoints = simulations[0][observable].timepoints
            sim_data = np.array(
                [
                    simulation[observable].interpolate(timepoints)
                    for simulation in simulations
                ]
            )

            plot_predictives_many(
                ax=ax, timepoints=timepoints, trajectories=sim_data, color=color
            )

            ax.set_xlabel("time", fontsize=axis_fontsize)
            ax.set_ylabel(observable, fontsize=axis_fontsize)

            if observable in experiment.observation_mapping.values():
                _measurement = experiment.getbysimkey(observable)
                plot_measurement(ax=ax, measurement=_measurement, color=color)

        if rID is not Constants.SINGLE_ID:
            fig.suptitle(rID)

        fig.tight_layout()
        figures.append(fig)

    if len(figures) > 1:
        return figures
    else:
        return figures[0]


def plot_predictives_many(ax, timepoints: np.ndarray, trajectories: np.ndarray, color):
    """
    Plot the predictive distributions for many trajectories.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    timepoints : np.ndarray
        The time points.
    trajectories : np.ndarray
        The trajectories.
    color : str
        The color for the plot.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list("_", ["white", color])
    percentiles = np.linspace(51, 99, 40)
    colors = (percentiles - np.min(percentiles)) / (
        np.max(percentiles) - np.min(percentiles)
    )
    trajectories = trajectories.T

    timepoints = timepoints.flatten()

    for i, p in enumerate(percentiles[::-1]):
        upper = np.percentile(trajectories, p, axis=1)
        lower = np.percentile(trajectories, 100 - p, axis=1)

        color_val = colors[i]
        ax.fill_between(timepoints, upper, lower, color=cmap(color_val), alpha=0.8)

    return ax


def plot_profile_likelihood(pl_results, p_opt, alpha=0.05, show_coi=True):
    """
    Plot the profile likelihood results.

    Parameters
    ----------
    pl_results : dict
        The profile likelihood results.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plots.
    """

    ncols = min(3, len(pl_results))
    nrows = int(len(pl_results) / ncols) + bool(len(pl_results) % ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(rel_figure_width * ncols, rel_fig_height * nrows),
        sharey=True,
    )

    # get negLL minimum
    mle_negll = np.hstack([elem[:, 1] for elem in pl_results.values()]).min()

    # define negLL threshhold as stopping criterion
    threshold = calculate_negll_thresshold(
        alpha=alpha, df=len(p_opt), mle_negll=mle_negll
    )

    # ensure axes is array
    axes = np.asarray(axes)

    # turn off empty axes by default
    for ax in axes.flat:
        ax.set_axis_off()

    for ax, (param, results) in zip(axes.flat, pl_results.items()):
        xvals = results[:, 0]
        neglls = results[:, 1]

        # show point estimate
        ax.axvline(
            p_opt[param],
            color="red",
            linestyle="-",
            label=f"estimate: {p_opt[param]:.3f}",
        )

        # show the thresshold for identifiability
        ax.axhline(threshold - mle_negll, linestyle="--", color="grey")

        if show_coi:
            coi = approximate_confidence_interval(xvals, neglls, threshold)

            ax.axvline(coi[0], linestyle="--", color="green")
            ax.axvline(coi[1], linestyle="--", color="green")

            # Add confidence interval to legend
            ax.plot(
                [],
                [],
                color="green",
                linestyle="--",
                label=f"{1-alpha} % CoI: {coi[0]:.3f}, {coi[1]:.3f}",
            )

        ax.legend(loc="best", ncol=2, fancybox=True)

        ax.set_axis_on()

        ax.plot(xvals, neglls - mle_negll, label="Profile Likelihood")
        ax.set_ylabel(r"$\Delta \;  negLL$")
        ax.set_xlabel(param)

    fig.tight_layout()
    return fig
