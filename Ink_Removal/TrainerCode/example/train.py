#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Wed Jun 22 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script shows example of how to run training on MNIST using the modules
# Modifications (date, what was modified):
#   1.
# --------------------------------------------------------------------------------------------------------------------------
#
import sys

from pathlib import Path
import yaml

import utils
import trainer

config_path = "mnist.yml"

mnist_trainer = utils.TrainEngine(config_pth=config_path)
mnist_trainer.run()
