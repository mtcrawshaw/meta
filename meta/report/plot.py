""" Plotting for performance metrics. """

from typing import Dict, List, Union, Any

import numpy as np
import matplotlib.pyplot as plt

from meta.utils.metrics import Metrics


def plot(
    metrics: Union[Metrics, Dict[str, Metrics]],
    plot_path: str,
    summary: Dict[str, Any] = None,
) -> None:
    """
    Plot the metrics from `metrics`, store image at `plot_path`. `metrics` should be a
    `Metrics` object in the case that the metrics from a single training run should be
    plotted. Otherwise, `metrics` should be a dictionary whose keys are names of methods
    and whose values are `Metrics` objects holding metric values from training with the
    corresponding methods. These multiple `Metrics` objects should hold values for the
    same metrics and each of these metrics should have the same history length. In this
    case, the results from different methods will be plotted on the same chart. Summary
    should be provided (as computed by `experiment()`) only in the case that `metrics`
    is a `dict`.
    """

    # Count the number of plots. We create a plot for each unique basename among all
    # basenames from the given metrics with non-empty history.
    basenames = []
    if isinstance(metrics, Metrics):
        basenames = get_basenames(metrics)
    elif isinstance(metrics, dict):
        first_metrics = list(metrics.values())[0]
        basenames = get_basenames(first_metrics)
        for method_metrics in metrics.values():
            assert basenames == get_basenames(method_metrics)
    else:
        raise NotImplementedError

    # Add an extra plot which will just contain a table.
    num_plots = len(basenames) + 1

    # Create subplots.
    fig_width = 12.8
    plot_height = 4.8
    table_height_ratio = 0.3
    fig, axs = plt.subplots(
        num_plots,
        figsize=(fig_width, plot_height * (num_plots - (1.0 - table_height_ratio))),
        gridspec_kw={"height_ratios": [1.0] * (num_plots - 1) + [table_height_ratio]},
    )

    # Wrap a single axis in a list to make sure that axs is iterable.
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # Create figure title.
    fig.suptitle("Training Metrics")

    # Find bound of x-axis for each plot.
    if isinstance(metrics, Metrics):
        max_metric_len = get_max_metric_len(metrics)
    elif isinstance(metrics, dict):
        first_metrics = list(metrics.values())[0]
        max_metric_len = get_max_metric_len(first_metrics)
        for method_metrics in metrics.values():
            assert max_metric_len == get_max_metric_len(method_metrics)
    else:
        raise NotImplementedError

    # Plot each metric set on a separate plot.
    plotted_metrics = []
    for i, basename in enumerate(basenames):

        # Construct metric names and find max metric length (scale of x-axis).
        legend = []
        if isinstance(metrics, Metrics):
            metric_names = get_metric_names(metrics, basename)
        elif isinstance(metrics, dict):
            first_metrics = list(metrics.values())[0]
            metric_names = get_metric_names(first_metrics, basename)
            for method_metrics in metrics.values():
                assert metric_names == get_metric_names(method_metrics, basename)
        else:
            raise NotImplementedError

        for metric_name in metric_names:

            # Get metric state values.
            mean_arrs = []
            stdev_arrs = []
            method_names = []
            if isinstance(metrics, Metrics):
                metric = metrics.metric_dict[metric_name]
                mean_arrs = [np.array(metric.mean)]
                stdev_arrs = [np.array(metric.stdev)]
                assert len(mean_arrs[0]) == len(stdev_arrs[0])

            elif isinstance(metrics, Dict):
                for j, (method, method_metrics) in enumerate(metrics.items()):
                    metric = method_metrics.metric_dict[metric_name]
                    mean_arr = np.array(metric.mean)
                    stdev_arr = np.array(metric.stdev)
                    assert len(mean_arr) == len(stdev_arr)
                    if j != 0:
                        assert len(mean_arr) == len(mean_arrs[0])
                    mean_arrs.append(mean_arr)
                    stdev_arrs.append(stdev_arr)
                    method_names.append(method)

            else:
                raise NotImplementedError

            # Assign x-axis values to each data point.
            num_points = len(mean_arrs[0])
            num_intervals = num_points if num_points > 1 else 1
            x_axis = [(j * max_metric_len) // num_intervals for j in range(num_points)]

            # Plot mean.
            for j in range(len(mean_arrs)):
                axs[i].plot(x_axis, mean_arrs[j])
                if isinstance(metrics, Metrics):
                    legend.append(metric_name)
                elif isinstance(metrics, dict):
                    legend.append(f"{method_names[j]}_{metric_name}")
                else:
                    raise NotImplementedError

                # Fill space between mean and upper/lower std devs.
                upper_dev_array = mean_arrs[j] + stdev_arrs[j]
                lower_dev_array = mean_arrs[j] - stdev_arrs[j]
                axs[i].fill_between(x_axis, lower_dev_array, upper_dev_array, alpha=0.2)

            # Add x-axis label.
            axs[i].set_xlabel("Training episodes")

            plotted_metrics.append(metric_name)

        # Add legend to subplot.
        axs[i].legend(legend, loc="upper left")

    # Helper function for constructing cell text.
    possibly_none = lambda val: "%.5f" % val if val is not None else "None"

    # Write out table of final metrics.
    axs[-1].axis("off")
    cell_text = []
    if isinstance(metrics, Metrics):
        row_labels = list(plotted_metrics)
        col_labels = ["Best", "Final"]

        for metric_name in plotted_metrics:
            metric = metrics.metric_dict[metric_name]
            row_text = []
            row_text.append(possibly_none(metric.best))
            if len(metric.mean) == 0:
                row_text.append("None")
            else:
                row_text.append(possibly_none(metric.mean[-1]))
            cell_text.append(list(row_text))

    elif isinstance(metrics, dict):
        row_labels = list(summary.keys())
        col_labels = list(metrics.keys())

        for metric_name in summary:
            row_text = []
            for method, method_metrics in metrics.items():
                method_results = summary[metric_name][method]
                mean = method_results["mean_performance"]
                CI = method_results["CI"]
                rad = (CI[1] - CI[0]) / 2
                if mean is None:
                    row_text.append(None)
                else:
                    row_text.append(f"{mean:.5f} += {rad:.5f}")
            cell_text.append(list(row_text))

    else:
        raise NotImplementedError

    axs[-1].table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        colWidths=[0.2] * len(col_labels),
        loc="upper center",
    )

    # Save out plot.
    plt.savefig(plot_path)
    plt.close()


def get_basenames(metrics: Metrics) -> List[str]:
    """ Utility function to aggregate basenames from a `Metrics` objects. """
    basenames = []
    for metric in metrics.metric_dict.values():
        if len(metric.history) > 0 and metric.basename not in basenames:
            basenames.append(metric.basename)
    return basenames


def get_max_metric_len(metrics: Metrics) -> List[str]:
    """ Utility function to get maximum length of a metric from `metrics`. """
    return max([len(metric.mean) for metric in metrics.metric_dict.values()])


def get_metric_names(metrics: Metrics, basename: str) -> List[str]:
    """
    Utility function to aggregate metric names with a given basename from a `Metrics`
    object.
    """
    return [
        metric_name
        for metric_name, metric in metrics.metric_dict.items()
        if metric.basename == basename
    ]
