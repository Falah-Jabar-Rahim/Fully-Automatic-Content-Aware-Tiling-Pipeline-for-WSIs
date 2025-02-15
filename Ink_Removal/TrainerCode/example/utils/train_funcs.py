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


@trainer.Metric.register("mnist")
class Test_Metric(trainer.Metric):
    def get_metrics(self):
        metricfun = torchmetrics.MetricCollection(
            [torchmetrics.Accuracy(), torchmetrics.ConfusionMatrix(num_classes=10)]
        )
        return metricfun


@trainer.Dataset.register("mnist")
class Mnist_Dataset(trainer.Dataset):
    def get_transforms(self) -> Tuple[Any, Any]:
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return transform, transform

    def get_loaders(self):
        trainset = torchvision.datasets.MNIST(
            "../data", train=True, download=True, transform=self.train_transform
        )
        testset = torchvision.datasets.MNIST(
            "../data", train=False, transform=self.train_transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.train_batch_size
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.test_batch_size
        )
        return trainset, trainloader, testset, testloader


@trainer.Logger.register("mnist")
class Mnist_logger(trainer.Logger):
    def log_table(self, input, output, label, epoch):
        columns = ["id", "image", "real class", "calculated class"]
        table = wandb.Table(columns=columns)
        _, preds = torch.max(output.data, 1)
        for i in range(10):
            idx = f"{epoch}_{i}"
            image = wandb.Image(input[i].permute(1, 2, 0).cpu().numpy())
            table.add_data(idx, image, preds[i], label[i])
        self.log({"table_key": table})


@trainer.Model.register("mnist")
class Mnist_model(trainer.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


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
            self.logger.log_table(image, outputs, label, self.current_epoch)

        mean_loss = np.mean(self.logger.get_tracked("loss_value")) / len(
            self.dataset.testloader
        )
        print("Total Val loss: {}".format(mean_loss))

        return self.metrics.results["val_Accuracy"], mean_loss
