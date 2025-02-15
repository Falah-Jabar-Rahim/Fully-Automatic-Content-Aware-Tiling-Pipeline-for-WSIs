"""
Trainer Class, Written for general purpose training. Aim is to have a better code base, learn how to write proper object oriented programming
and while shifting from experiment to experiment, dont have to change main code again and again
Author: Vishwesh Ramanathan
Created: June 3, 2022
"""
import sys
import os
import abc
from typing import Tuple, Union, Dict, Any
import time
import inspect
import random
import importlib

from pathlib import Path
import torch
import numpy as np
import yaml
import ruamel.yaml

from . import Dataset, Model, Metric, Logger


class Trainer:
    """
    Main class for training purposes.
    For running training:
        trainer = Trainer(config_pth="path.yml")
        trainer.run()
    The class expects user to provide with train and val method along with a config file in yml format

    Attributes:
        args: Dictionary from config file provided in yaml format
        device: torch device where data will reside
        dataset: trainer.Dataset class for handling data
        logger: trainer.Logger class for logging purposes
        model: trainer.Model class for deep learning model
        metrics: trainer.Metric class for obtaining metrics while training/val
        loss_fun: lossfunction as specified in the config file for training
        optimizer: optimizer as specified in the config file for optimizing
        scheduler: scheduler as specified in the config file for learning rate scheduling
    Methods:
        save_checkpoint: For saving all the states of model, optimizer, scheduler, metric and epoch
        load_checkpoint: For loading checkpoint
        train: Abstract method to be specified by the user for training
        val: Abstract method to be specified by the user for validation
        run: For running the training loop based on the config file given
    """

    # mode should be one of either train/val/test. Useful for self.metric attribute
    mode = "train"

    def __init__(self, config_pth: str, sweep_config: str = None, **kwargs) -> None:
        """
        Initializes the trainer with given configurations as mentioned in config path.
        Additionally, kwargs can be used to edit some variables in the config path in cases of automatic
        parameters

        Parameters:
            config_path: The path to config file. It is assumed the file is in YAML format. For sweep mode,
                         please provide path to the sweep configuration file
            sweep: Set True if the user is running hyperparameter sweeps
            kwargs: Mention variables you want to change in the config file via code. Useful for variables
                    given via argparse. Please ensure the name of kwarg variables matches key in the config
                    file
        """
        #Determine if sweep is activated or not
        sweep = sweep_config is not None
        
        with open(config_pth,"r") as file:
            self.args = yaml.safe_load(file)

        # Edit variables using kwargs
        self._editargs(kwargs)

        # Initializes for training, gets folders ready
        self._initialize_engine()

        self.device = torch.device(
            f'cuda:{self.args["ENGINE"]["gpu_devices"][0]}'
            if torch.cuda.is_available()
            else "cpu"
        )

        if sweep:
            with open(sweep_config,"r") as file:
                self.sweep_args = yaml.safe_load(file)
            wandb_config = None
        else:
            self._display(config_pth)
            wandb_config = self.args

        # Build classes
        if self.args["LOGGER"]["subclass_name"] is None:
            self.logger = Logger(**{k: self.args["LOGGER"][k] for k in set(list(self.args["LOGGER"].keys())) - set(["subclass_name","watch_gradients"])},configs=wandb_config)
        else:
            self.logger = Logger.create(**{k: self.args["LOGGER"][k] for k in set(list(self.args["LOGGER"].keys())) - set(["watch_gradients"])},configs=wandb_config)
        
        if sweep:
            #After initialization of wandb run, edit the arguments based on sweep controller
            self._sweepargs()
        
        self.dataset = Dataset.create(**self.args["DATASET"])
        self.model = Model.create(**self.args["MODEL"])
        self.metrics = Metric.create(
            **self.args["METRIC"], logger=self.logger, device=self.device
        )

        #Inject system parameters so that it could be used everywhere in the classes
        self.dataset._inject_args(self.args)
        self.model._inject_args(self.args)


        # Build loss function, optimizer and scheduler for training operations
        self.loss_fun = self._build_from_name(**self.args["LOSS"]).to(self.device)
        self.optimizer = self._build_from_name(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.args["OPTIMIZER"],
        )   
        self.scheduler = self._build_from_name(
            optimizer=self.optimizer, **{k: self.args["SCHEDULER"][k] for k in set(list(self.args["SCHEDULER"].keys())) - set(["epoch_wise"])}
        )

        # For resuming training from checkpoint
        if self.args["ENGINE"]["resume_loc"] is not None:
            self.load_checkpoint()
        # For transfer learning
        elif self.args["ENGINE"]["transfer_loc"] is not None:
            self.model.load_model_weights(
                model_path=self.args["ENGINE"]["transfer_loc"], device=self.device
            )

        # For switching to dataparallel
        if self.args["ENGINE"]["use_dataparallel"]:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.args["ENGINE"]["gpu_devices"]
            )

        # Switch model to gpu
        self.model.to(self.device)

    def _editargs(self, new_values: dict) -> None:
        """
        Edit variables stored in self.args based on given new_values
        """
        module_keys = self.args.keys()
        for key, value in new_values.items():
            flag = 0
            for module_key in module_keys:
                if key in self.args[module_key].keys():
                    self.args[module_key][key] = value
                    flag = 1
                    break
            if flag == 0:
                raise ValueError(f"{key} not found in the given config file")

    def _sweepargs(self) -> None:
        """
        Edit args based on the hyperparameters decided by the sweep controller which is present in the logger object
        """
        #go through all parameters and change values in self.args
        for key in self.sweep_args["parameters"].keys():
            dict_heirarchy = key.split(".")
            #Hyperparameter val
            arg_val = self.logger.get_logger.config[key]
            print(key)
            print(arg_val)
            #Know only a very primitive way of returning multilevels
            if len(dict_heirarchy)==1:
                self.args[dict_heirarchy[0]] = arg_val
            elif len(dict_heirarchy)==2:
                self.args[dict_heirarchy[0]][dict_heirarchy[1]] = arg_val
            elif len(dict_heirarchy)==3:
                self.args[dict_heirarchy[0]][dict_heirarchy[1]][dict_heirarchy[2]] = arg_val
            else:
                raise ValueError("Too many levels in config file")
        print(self.args)

    @staticmethod
    def _display(file_path) -> None:
        """
        Display selected hyperparameters
        """
        print("-"*50)
        print("Selected Hyperparameters")
        print("-"*50)
        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        with open(file_path, "r") as file:
            data = yaml.load(file)
        yaml.dump(data, sys.stdout)
        print("-"*50)
    
    def _initialize_engine(self) -> None:
        """
        Makes new directory for saving model if applicable and initializes seed
        """
        # Specifically for computecanada
        if self.args["ENGINE"]["location_mod"] is not None:
            self.args["DATASET"]["path"] = str(
                Path(self.args["ENGINE"]["location_mod"]) / self.args["DATASET"]["path"]
            )

        if self.args["ENGINE"]["save_loc"] is not None:
            self.is_save = True
            parent_name = Path(self.args["ENGINE"]["save_loc"])
            folder_name = parent_name / Path("Results")
            self.model_save = (
                folder_name / Path(self.args["LOGGER"]["run_name"]) / "saved_models"
            )

            if not folder_name.is_dir():
                os.mkdir(folder_name)
                os.mkdir(self.model_save.parent)
                os.mkdir(self.model_save)
            elif not self.model_save.parent.is_dir():
                os.mkdir(self.model_save.parent)
                os.mkdir(self.model_save)
            else:
                pass
        else:
            self.is_save = False

        random_seed = self.args["ENGINE"]["random_seed"]
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    def save_checkpoint(self, epoch: int, metric: float) -> None:
        """
        Saves checkpoint for the model, optimizer, scheduler. Additionally saves the best metric score and epoch value
        """
        print("Saving Checkpoint ...")
        #Saving bug for cyclic lr
        scheduler_statedict = self.scheduler.state_dict()
        if "_scale_fn_ref" in scheduler_statedict.keys():
            scheduler_statedict.pop("_scale_fn_ref")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": scheduler_statedict,
                "metric": metric,
            },
            self.model_save
            / Path(
                "Checkpoint_{}_{:.2f}.pt".format(
                    time.strftime("%d%b%H_%M_%S", time.gmtime()), metric.item()
                )
            ),
        )
        torch.cuda.empty_cache()

    def load_checkpoint(self) -> Tuple[int, float]:
        """
        Loads checkpoint for the model, optimizer, scheduler, epoch and best_metric
        """
        checkpoint = torch.load(
            self.args["ENGINE"]["resume_loc"], map_location=self.device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        best_metric = checkpoint["metric"]
        del checkpoint
        torch.cuda.empty_cache()
        return epoch, best_metric

    @staticmethod
    def _build_from_name(module_name: str, subclass_name: str, **kwargs) -> Any:
        """
        Given module, builds object based on the given name and extra parameters
        """
        mymodule = importlib.import_module(module_name)
        return getattr(mymodule, subclass_name)(**kwargs)

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """
        Method for training loop
        """
        ...

    @abc.abstractmethod
    def val(self, *args, **kwargs) -> Tuple[float, float]:
        """
        Method for validation loop.
        Returns:
            metrics: Metric value for the epoch. Please make sure better performance indicates higher metric score
            loss: Loss value, mainly used for step function if valid
        """
        ...

    def run(self) -> None:
        """
        Run training and validation for the number of epochs provided
        """
        if self.args["LOGGER"]["use_wandb"] and (self.args["LOGGER"]["watch_gradients"] is not None):
            if self.args["LOGGER"]["watch_gradients"]:
                self.logger.watch(models=self.model,
                                log="all",
                                log_graph=True
                                )
        best_metric_val = -10000
        if self.args["ENGINE"]["resume_loc"] is not None:
            epoch_start, best_metric_val = self.load_checkpoint()
        else:
            epoch_start = 0

        for epoch in range(epoch_start, self.args["ENGINE"]["epochs"]):
            self.current_epoch = epoch
            print("EPOCH: {}, METRIC: {}".format(epoch, best_metric_val))
            # For logging purpose, we need to define the modes
            self.metrics.mode = "train"
            self.train()
            self.metrics.mode = "val"
            metric_val, val_loss = self.val()
            if self.is_save and (
                ((epoch + 1) % self.args["ENGINE"]["save_freq"] == 0)
                or (metric_val > best_metric_val)
            ):
                self.save_checkpoint(epoch, metric_val)
            
            if metric_val > best_metric_val:
                best_metric_val = metric_val

            if self.args["SCHEDULER"]["epoch_wise"]:
                if "metrics" in inspect.getfullargspec(self.scheduler.step).args:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            self.logger.log({"Learning Rate": self.optimizer.param_groups[0]["lr"]})
        print("Training Done ...")

    def sweep(self, sweep_counts:int, sweep_id = None,) -> str:
        """
        For running sweeps given sweep configuration. Returns sweep id which can be used for parallel agents
        Paramters:
            sweep_counts: Number of experiments
            sweep_id: Incase of already initialized sweep experiments
        """
        if sweep_id is None:
            sweep_id = self.logger.get_logger.sweep(sweep=self.sweep_args, project=self.args["LOGGER"]["project_name"])
        
        self.logger.get_logger.agent(sweep_id, function=self.run, count=sweep_counts)
