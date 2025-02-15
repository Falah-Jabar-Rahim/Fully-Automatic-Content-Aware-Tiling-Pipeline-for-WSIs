from typing import Union, Tuple, Any

import numpy as np
import torch
import torchmetrics
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

import trainer

class TrainEngine(trainer.Trainer):
    def train(self):
        self.model.train()
        for data in tqdm(self.dataset.trainloader):
            image, label = data
            image, label = image.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(image)
            loss = self.loss_fun(outputs, label)
            loss.backward()
            self.optimizer.step()
            # Track loss
            self.logger.track(loss_value=loss.item())
            # metric calculation
            self.metrics(outputs, label)
            # Logging loss
            self.logger.log({"Epoch Train loss": loss.item()})
        self.metrics.compute()
        self.metrics.log()
        print(
            "Total Train loss: {}".format(
                np.mean(self.logger.get_tracked("loss_value"))
            )
        )

    def val(self):
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self.dataset.testloader):
                image, label = data
                image, label = image.to(self.device), label.to(self.device)
                outputs = self.model(image)
                loss = self.loss_fun(outputs, label)
                # Track loss
                self.logger.track(loss_value=loss.item())
                # metric calculation
                self.metrics(outputs, label)
                # Logging loss
                self.logger.log({"Epoch Train loss": loss.item()})
        self.metrics.compute()
        self.metrics.log()
        if self.current_epoch % 5 == 0:
            self.logger.log_table(image, outputs, label, self.current_epoch,self.metrics.results)

        mean_loss = np.sum(self.logger.get_tracked("loss_value")) / len(
            self.dataset.testloader
        )
        print("Total Val loss: {}".format(mean_loss))

        return self.metrics.results["val_Accuracy"], mean_loss
