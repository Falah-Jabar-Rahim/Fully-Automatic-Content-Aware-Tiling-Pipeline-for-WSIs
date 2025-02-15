#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Wed Jun 23 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is to run training on ink filter using the modules
# --------------------------------------------------------------------------------------------------------------------------
#
import sys

from pathlib import Path
import yaml
import argparse

import utils
import trainer

parser = argparse.ArgumentParser()
parser.add_argument("-c", help="config location",required=True)
#parser.add_argument("-l", help="location modifier of dataset incase such as for compute canada",required=True)
args = parser.parse_args()

config_path = args.c
#location_mod = args.l

mnist_trainer = utils.TrainEngine(config_pth=config_path)
mnist_trainer.run()
