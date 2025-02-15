# Trainer: Modularized code for training deep learning algorithms
This repository is for making the training code more modularized and easy to run and track multiple experiments, without much changes in the main code.

## Introduction
Yet to write documentation. Please look at `examples` directory meanwhile
## Getting Started
### Installation
This repository currently can be installed as a package using the following command
Recommended:
```
pip install git+https://github.com/Vishwesh4/TrainerCode.git
``` 
Optional:
```
pip install -e git+https://github.com/Vishwesh4/TrainerCode.git
```
### Requirements
```
numpy
torch
torchmetrics
wandb
pyyaml
```
### How to use Trainer
#### Step 1
For the repository to work, you need to specify your classes and register it with appropriate module using any name, as specified in the `example`. This
way you can directly call your classes by just mentioning the registered name in the config file.  
You need to create your own classes for `dataset`,`metric`,`model` and register it. Optionally you can also create your own class for `logger` and register it. Your code should look like this
```python
from typing import Tuple, Union, Any

from torch.utils.data import Dataset, DataLoader
import trainer


#For registering to dataset module
@trainer.Dataset.register("custom_name")
class YOUR_CLASS(trainer.Dataset):
    def get_transforms(self) -> Tuple[Any, Any]:
        #REQUIRED ABSTRACT METHOD, IMPLEMENT HERE
        return train_transform, test_transform
    def get_loaders(self) -> Tuple[Dataset, DataLoader, Dataset, DataLoader]:
        #REQUIRED ABSTRACT METHOD, IMPLEMENT HERE
        return trainset, trainloader, testset, testloader

#For registering to dataset module
@trainer.Metric.register("custom_name")
class YOUR_CLASS(trainer.Metric):
    def get_metrics(self) -> Union[torchmetrics.MetricCollection, List[torchmetrics.MetricCollection]]:
        #REQUIRED ABSTRACT METHOD, IMPLEMENT HERE
        return metricfunction

#For registering to dataset module
@trainer.Model.register("custom_name")
class YOUR_CLASS(trainer.Model):
    pass

#For registering to dataset module
@trainer.Logger.register("custom_name")
class YOUR_CLASS(trainer.Logger):
    pass

```
#### Step 2
For the package to detect your classes, you need to import it. This can be easily done by putting all your custom classes in a folder and creating a `__init__.py` file with the following code for automatically importing.

```python
'''
 file_name : __init__.py
 For automatically importing your classes in the folder where the file __init__.py is situated
'''
import os
from inspect import isclass
from pathlib import Path
from importlib import import_module

# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for name in os.listdir(package_dir):
    if (name.endswith(".py")) and (name!=Path(__file__).name):
        # import the module and iterate through its attributes
        module_name = name[:-3]
        module = import_module(f"{__name__}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isclass(attribute):
                # Add the class to this package's variables
                globals()[attribute_name] = attribute
```
#### Step 3 - Final
For the training to happen, you need to create a class inheriting `Trainer` module. For beginning the training, you can pass `config.yml` file to the 
class and use `run` method to begin the training. Note that all your custom classes should be loaded in globals. You can use the given `__init__.py` for it.

```python
import trainer
import your_custom_class_folder

class YOUR_TRAINER(trainer.Trainer):
    def train(self):
        #REQUIRED ABSTRACT METHOD, IMPLEMENT HERE
        pass
    def val(self) -> Tuple[float, float]:
        #REQUIRED ABSTRACT METHOD, IMPLEMENT HERE
        pass

your_trainer = YOUR_TRAINER(config_pth=config_path)
your_trainer.run()
```
### Creating Config file
