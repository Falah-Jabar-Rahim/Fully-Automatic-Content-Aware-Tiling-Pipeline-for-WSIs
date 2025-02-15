import abc
from typing import Dict, List, Union

import torchmetrics
import torch

from . import Logger, RegistrantFactory


class Metric(RegistrantFactory):
    """
    Wrapper around torchmetrics MetricCollection to incorporate
    logging and multiple MetricCollection objects

    This class expects the user to provide with get_metrics method

    """

    SETTINGS = ["train", "val", "test"]
    subclasses = {}
    mode = "train"

    def __init__(self, logger: Logger, device: torch.device, **kwargs) -> None:
        """
        Wraps list of metric collections to make it perform like
        singular metric collections. Usefull for multioutput cases.
        Please ensure these objects have seperate prefix so that it could be differentiated
        This class will automatically take care of train/val metrics,
        hence do not clone seperately for train/test/val logs
        Parameters:
            metrics: Single or list of metriccollections
            logger: Logger object for logging purposes
        """
        self.logger = logger
        self.kwargs = kwargs
        self.metrics = self.get_metrics()

        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]

        self.metrics_dict = {}
        for name in Metric.SETTINGS:
            temp_metric = []
            for mets in self.metrics:
                temp_metric.append(mets.clone(prefix=name + "_").to(device))
            self.metrics_dict[name] = temp_metric

    @abc.abstractmethod
    def get_metrics(
        self,
    ) -> Union[torchmetrics.MetricCollection, List[torchmetrics.MetricCollection]]:
        """
        Define the metrics in this method
        This method is supposed to return metrics
        Returns:
            metrics:  Union[torchmetrics.MetricCollection, List[torchmetrics.MetricCollection]]
        """
        pass

    def compute(self) -> None:
        """
        Calculate the aggregate for the epoch and reset
        The results are stored in metrics_calc attributes
        """
        self.metrics_calc = []
        for i in range(len(self.metrics_dict[self.mode])):
            self.metrics_calc.append(self.metrics_dict[self.mode][i].compute())
        self.reset()

    def reset(self) -> None:
        """
        Resets all torchmetrics compose objects
        """
        for i in range(len(self.metrics_dict[self.mode])):
            self.metrics_dict[self.mode][i].reset()

    def __call__(self, *args) -> None:
        """
        For multiple outputs, the argument should be passed as a
        list of tuples, for instance metric([(tensor,tensor),(tensor,tenosr)])
        For single output, the arguments can be two tensors,
        for instance metric(tensor,tensor)
        """
        if not isinstance(args, list):
            args = [args]

        assert len(self.metrics_dict[self.mode]) == len(
            args
        ), "Number of arguments doesn't match , number of MetricCollection objects provided"

        for i, argument in enumerate(args):
            self.metrics_dict[self.mode][i](*argument)

    @property
    def results(self) -> Dict:
        metric_results = {}
        for results in self.metrics_calc:
            diff_results = results.keys()
            for key in diff_results:
                metric_results[key] = results[key].cpu().numpy()
        return metric_results

    def log(self) -> None:
        """
        Logs the metrics using logger object
        """
        results = self.results
        self.logger.log(results)
