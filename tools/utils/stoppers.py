from tools.utils.typing import Status, MetricType, MetricEntry, Dict, List
import numpy as np


class Stopper:
    def __init__(self, config: Dict, tasks: List = None):
        """Create a stopper instance with metric fields. This fields should cover
        all feasible attributes from rollout/training results.

        :param Dict config: Configuration to control the stopping.
        :param List tasks: A list of sub task identifications. Default to None
        """
        self._config = config
        self.tasks_status = dict.fromkeys(tasks, Status.NORMAL) if tasks else {}

    def __call__(self, results: Dict[str, MetricEntry], global_step: int) -> bool:
        """Parse results and determine whether we should terminate tasks."""

        raise NotImplementedError

    @property
    def info(self):
        """Return statistics for analysis"""

        raise NotImplementedError

    def set_terminate(self, task_id: str) -> None:
        """Terminate sub task tagged with task_id, and set status to terminate."""

        assert task_id in self.tasks_status, (task_id, self.tasks_status)
        self.tasks_status[task_id] = Status.TERMINATE

    def all(self):
        """Judge whether all tasks have been terminated

        :return: a bool value indicates terminated or not
        """

        terminate = len(self.tasks_status) > 0
        for status in self.tasks_status.values():
            if status == Status.NORMAL:
                terminate = False
                break

        return terminate


class SimpleRolloutStopper(Stopper):
    """SimpleRolloutStopper will check the equivalence between evaluate results and"""

    def __init__(self, config, tasks: List = None):
        super(SimpleRolloutStopper, self).__init__(config, tasks)
        self._config["max_step"] = self._config.get("max_step", 100)
        self._info = {MetricType.REACH_MAX_STEP: False}

    @property
    def max_iteration(self):
        return self._config["max_step"]

    def __call__(self, results: Dict[str, MetricEntry], global_step):
        """Default rollout stopper will return true when global_step reaches to an oracle"""
        if global_step == self._config["max_step"]:
            self._info[MetricType.REACH_MAX_STEP] = True
            return True
        return False

    @property
    def info(self):
        raise self._info


class GFootRolloutStopper(Stopper):
    def __init__(self, config: Dict, tasks: List = None):
        super().__init__(config, tasks=tasks)

        self._max_step = self._config.get("max_step", 1000)
        self._min_step = self._config.get("min_step", 200)
        self._threshold = self._config.get("threshold", 0.03)
        self._min_win_rate = self._config.get("min_win_rate", 0.03)
        self._stop_when_reach_min_win = self._config.get(
            "stop_when_reach_min_win", False
        )
        # assert (
        #     self._min_step >= 100
        # ), f"config min_step must >= 100 but now {self._min_step}."

        self._past_win_rate = []

    def __call__(
        self, results: Dict[str, Dict], global_step: int
    ) -> bool:  # we need win stats
        if global_step == 0:
            return False

        if global_step < self._min_step:
            return False
        elif global_step >= self._max_step:
            return True
        else:
            current_win = results["custom_metrics/team_0/win_mean"]  # mean win rate
            self._past_win_rate.append(current_win)

            past_100_wins = self._past_win_rate[-100:]
            smooth_wins = [np.mean(past_100_wins[i : i + 10]) for i in range(90)]
            smooth_2_wins = [np.mean(smooth_wins[i : i + 10]) for i in range(80)]
            if self._stop_when_reach_min_win:
                if smooth_2_wins[-1] > self._min_win_rate:
                    print(
                        f">>>>>>epoch:{global_step}, "
                        "heuristic stop in rollout for reaching minimum win rate <<<<<<"
                    )
                    return True

            if (
                smooth_2_wins[-1] > 0.5
                and (smooth_2_wins[-1] - smooth_2_wins[-10]) < self._threshold
            ):
                # and (smooth_2_wins[-1] - smooth_2_wins[-10]) > self._threshold:
                print(
                    f">>>>>>epoch:{global_step}, "
                    "heuristic stop in rollout for non increasing win rate <<<<<<"
                )
                return True
            else:
                return False


class NonStopper(Stopper):
    """NonStopper always return false"""

    def __init__(self, config, tasks=None):
        super(NonStopper, self).__init__(config, tasks)

    def __call__(self, *args, **kwargs):
        return False

    @property
    def info(self):
        return {}


class SimpleTrainingStopper(Stopper):
    """SimpleRolloutStopper will check the equivalence between evaluate results and"""

    def __init__(self, config: Dict, tasks: List = None):
        super(SimpleTrainingStopper, self).__init__(config, tasks)
        self._config["max_step"] = self._config.get("max_step", 100)
        self._info = {
            MetricType.REACH_MAX_STEP: False,
        }

    def __call__(self, results: Dict[str, MetricEntry], global_step):
        """Ignore training loss, use global step."""

        if global_step == self._config["max_step"]:
            self._info[MetricType.REACH_MAX_STEP] = True
            return True
        return False

    @property
    def info(self):
        return self._info


def get_stopper(name: str):
    """Return a stopper class with given type name.

    :param str name: Stopper name, choices {simple_rollout, simple_training}.
    :return: A stopper type.
    """

    return {
        "none": NonStopper,
        "simple_rollout": SimpleRolloutStopper,
        "simple_training": SimpleTrainingStopper,
        "grf_rollout": GFootRolloutStopper,
    }[name]
