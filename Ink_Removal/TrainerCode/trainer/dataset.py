"""
This script is a wrapper around the data loader. The class defined returns data loaders
"""
import abc
from typing import Any, List, Optional, Union, Tuple, Dict

from torch.utils.data import Dataset, DataLoader

from . import RegistrantFactory


class Dataset(RegistrantFactory):
    """
    Class for handling data input output operations. The class wraps around augmentation, torch Dataset class and
    torch DataLoader

    The class expects user to provide with get_transforms and get_loaders methods
    """

    subclasses = {}

    def __init__(
        self, path: str, train_batch_size: int, test_batch_size: int, **kwargs
    ) -> None:

        self.path = path
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.kwargs = kwargs

        # Empty Variables
        self.train_loader = None
        self.test_loader = None
        self.train_set = None
        self.test_set = None

        self.train_transform, self.test_transform = self.get_transforms()
        (
            self.train_set,
            self.train_loader,
            self.test_set,
            self.test_loader,
        ) = self.get_loaders()

    def _inject_args(self,args:Dict) -> None:
        """
        Injects all the hyperparameters into class variable
        """
        self.system_params = args
    
    
    @abc.abstractmethod
    def get_transforms(self) -> Tuple[Any, Any]:
        """
        In this method, you are supposed to define transforms for augmentation purposes
        Please ensure they are returned in the same order
        Returns:
            train transforms
            test transforms
        """

    @abc.abstractmethod
    def get_loaders(self) -> Tuple[Dataset, DataLoader, Dataset, DataLoader]:
        """
        This method is supposed to overwrite and get the train and test dataset and dataloaders ready
        Please ensure they are returned in the same order
        Returns:
            train_set: training dataset of Dataset class
            train_loader: training data loader of DataLoader class
            test_set: testing/val dataset of Dataset class
            test_loader: testing/val dataloader od DataLoader class
        """

    @property
    def trainloader(self) -> DataLoader:
        return self.train_loader

    @property
    def testloader(self) -> DataLoader:
        return self.test_loader

    @property
    def trainset(self) -> Dataset:
        return self.train_set

    @property
    def testset(self) -> Dataset:
        return self.test_set
