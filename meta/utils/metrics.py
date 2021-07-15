"""
Object definition for Metrics class, which stores and updates training performance metrics.
"""

from typing import Dict, List, Tuple, Any

import numpy as np


class Metrics:
    """
    Metrics object, which stores and updates training performance metrics.
    """

    def __init__(self, metric_set: List[Dict[str, Any]]) -> None:
        """ Init function for Metrics object. """

        # Set metrics.
        for metric_info in metric_set:
            metric_kwargs = dict(metric_info)
            del metric_kwargs["name"]
            setattr(
                self, metric_info["name"], Metric(**metric_kwargs),
            )

        self.state_vars = [metric_info["name"] for metric_info in metric_set]
        self.metric_dict = {
            state_var: getattr(self, state_var) for state_var in self.state_vars
        }

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        message = ""
        printed_vars = 0
        for i, state_var in enumerate(self.state_vars):
            metric = getattr(self, state_var)
            if not metric.show:
                continue
            if printed_vars != 0:
                message += " | "
            message += "%s %s" % (state_var, getattr(self, state_var))
            printed_vars += 1

        return message

    def update(self, update_values: Dict[str, List[float]]) -> None:
        """
        Update performance metrics with a sequence of the most recent metric values.
        """
        for metric_name, metric_values in update_values.items():
            getattr(self, metric_name).update(metric_values)

    def current_values(self) -> Dict[str, float]:
        """
        Return a dictionary of the most recent values for each performance metric.
        """
        return {
            state_var: getattr(self, state_var).mean[-1]
            for state_var in self.state_vars
        }

    def history(self) -> Dict[str, List[float]]:
        """
        Return a dictionary of the history of values for each performance metric.
        """
        return {
            state_var: getattr(self, state_var).history for state_var in self.state_vars
        }

    def state(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dictionary with the value of all state variables.
        """
        return {
            state_var: getattr(self, state_var).state() for state_var in self.state_vars
        }

    @staticmethod
    def mean(metrics_list: List["Metrics"]) -> "Metrics":
        """
        Compute the mean of a given list of metrics from multiple trials, each trial
        with the same metrics.

        This method requires `metrics_list` is non-empty, and that each `Metrics`
        objects in `metrics_list` has the same set of state variables (stored in
        `self.state_vars`). Further, for each state variable, each corresponding
        `Metric` object must have the same values of the following attributes:
        `['len(history)', 'basename', 'window', 'point_avg', 'maximize', 'show']`.
        """

        assert len(metrics_list) > 0

        # Check that state variables and attributes of underlying `Metric` objects of
        # each element of `metrics_list` match those of the first element of
        # `metric_list`.
        first = metrics_list[0]
        state_vars = list(first.state_vars)
        for metrics in metrics_list:
            assert metrics.state_vars == state_vars
            for state_var in state_vars:
                exp_metric = first.metric_dict[state_var]
                metric = metrics.metric_dict[state_var]
                assert len(metric.history) == len(exp_metric.history)
                assert metric.basename == exp_metric.basename
                assert metric.window == exp_metric.window
                assert metric.point_avg == exp_metric.point_avg
                assert metric.maximize == exp_metric.maximize
                assert metric.show == exp_metric.show

        # Instantiate mean metrics object. Notice that we set `point_avg` to False for
        # `mean_metrics`. If `point_avg` is True for any of the metrics that we are
        # taking the mean over, then the original values given to `Metric.update()` will
        # have already been averaged before being stored in the metric history. So there
        # is no need to average again.
        metric_attrs = ["basename", "window", "maximize", "show"]
        mean_metrics = Metrics(
            [
                {
                    "name": state_var,
                    "point_avg": False,
                    **{
                        attr: getattr(first.metric_dict[state_var], attr)
                        for attr in metric_attrs
                    },
                }
                for state_var in state_vars
            ]
        )

        # Compute mean of each metric at each timestep.
        mean_histories = {}
        for state_var in state_vars:
            var_history = np.array(
                [metrics.metric_dict[state_var].history for metrics in metrics_list]
            )
            mean_histories[state_var] = np.mean(var_history, axis=0)

        # Fill mean metrics and return.
        mean_metrics.update(mean_histories)
        return mean_metrics


class Metric:
    """ Class to store values for a single metric. """

    def __init__(
        self,
        basename: str,
        window: int = 50,
        point_avg: bool = False,
        maximize: bool = True,
        show: bool = True,
    ) -> None:
        """
        Init function for Metric. We keep track of the total history of metric values, a
        moving average of the past `window` values, a moving standard deviation of
        the past `window` values, and a maximum average so far. If `point_avg` is
        True, then `update` will condense the list of given values into their average
        and treat it as a single update for the metric.
        """

        self.basename = basename
        self.window = window
        self.point_avg = point_avg
        self.maximize = maximize
        self.show = show

        # Metric values.
        self.history: List[float] = []
        self.mean: List[float] = []
        self.stdev: List[float] = []
        self.best: float = None

        self.state_vars = ["history", "mean", "stdev", "best"]

    def __repr__(self) -> str:
        """ String representation of ``self``. """

        if self.maximize is None:
            if len(self.history) > 0:
                mean = self.mean[-1]
                message = "mean: %.3f" % mean
            else:
                message = "mean: %f" % None
        else:
            if len(self.history) > 0:
                mean = self.mean[-1]
                best = self.best
                message = "mean, best: %.3f, %.3f" % (mean, best)
            else:
                message = "mean, best: %r, %r" % (None, None)

        return message

    def update(self, values: List[float]) -> None:
        """ Update history, mean, and stdev with new values. """

        # Replace `values` with average of `values` if self.point_avg, using recent
        # history to fill in if `values` is empty.
        if self.point_avg:

            if len(values) > 0:
                new_val = np.mean(values)
            elif len(self.mean) > 0:
                new_val = self.mean[-1]
            else:
                # If `values` and history are both empty, do nothing.
                return

            values = [new_val]

        # Add each new value to history and update running estimates.
        for value in values:
            self.history.append(value)

            # Update moving average and standard deviation.
            self.mean.append(np.mean(self.history[-self.window :]))
            self.stdev.append(np.std(self.history[-self.window :]))

            # Compute new best.
            if self.maximize is None:
                continue

            if self.best is None:
                self.best = self.mean[-1]
            else:
                if self.maximize:
                    new_best = self.mean[-1] > self.best
                else:
                    new_best = self.mean[-1] < self.best
                if new_best:
                    self.best = self.mean[-1]

    def state(self) -> Dict[str, Any]:
        """ Return a dictionary with the value of all state variables. """
        return {state_var: getattr(self, state_var) for state_var in self.state_vars}
