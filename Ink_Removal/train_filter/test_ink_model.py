import os
import sys
from pathlib import Path
import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torchvision
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchmetrics
import random
import pickle as pkl

from modules.patch_extraction import ExtractAnnotations
import trainer
import utils

random_seed = 2022
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

#Get paths
parser = argparse.ArgumentParser()
parser.add_argument("-d", help="csv file location with slide names",required=True)
parser.add_argument("-i", help="ink slide location path",required=True)
parser.add_argument("-c", help="clean slide location path",required=True)
args = parser.parse_args()

df = pd.read_excel(args.d)
ink_slide_path = args.i
clean_path = args.c

#Assuming csv file has two headers ink slide and clean slide
slide_path = [str( Path(ink_slide_path) / (str(df["Ink Slides"][i])+".svs" ) ) for i in range(len(df))]
#Sedeen automatically saves all annotations in a folder named sedeen in the location where all slides are saved
annotation_dir = str( Path(ink_slide_path) / Path("sedeen") )

TILE_H = 256
TILE_W = 256
TILE_STRIDE_FACTOR_H = 1
TILE_STRIDE_FACTOR_W = 1
LWST_LEVEL_IDX = 0
TRANSFORM = torchvision.transforms.ToTensor()
# SPACING = 0.2526
SPACING = None

ink_labelset = {"clean":"#00ff00ff","ink":"#ff0000ff"}

all_dataset = ExtractAnnotations(
        image_pth=slide_path,
        annotation_dir=annotation_dir,
        renamed_label=ink_labelset,
        tile_h=TILE_H,
        tile_w=TILE_W,
        tile_stride_factor_h=TILE_STRIDE_FACTOR_H,
        tile_stride_factor_w=TILE_STRIDE_FACTOR_W,
        spacing=None,
        lwst_level_idx=LWST_LEVEL_IDX,
        mode="train",
        train_split=1,
        transform=TRANSFORM,
        threshold=0.7,
        sample_threshold=50
        )

print(f"Total Length of the dataset: {len(all_dataset)}")

#Ink filter model
model_path = str(Path(__file__).parent.parent / "Ink_removal_weights/filter_weights.pt")
device = torch.device("cuda:3")
model = trainer.Model.create("ink")
model.load_model_weights(model_path,torch.device("cpu"))
model.to(device)

cm = torchmetrics.ConfusionMatrix(num_classes=2)
auroc = torchmetrics.AUROC(pos_label=1)

model.eval()
dataloader = torch.utils.data.DataLoader(all_dataset,batch_size=64)
index = []
ink_index = []
with torch.no_grad():
    for i,data in enumerate(tqdm(dataloader)):
        img, label = data
        img = img.to(device)
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        preds = torch.squeeze(predicted.cpu())
        ink_index.extend(64*i + torch.where(preds==1)[0].numpy())
        index.extend(64*i + torch.where(torch.abs(preds-label)==1)[0].numpy())
        soft = torch.nn.functional.softmax(outputs,dim=1).cpu()
        cm(preds,label)
        auroc(soft[:,1],label)
        # break

confusion = cm.compute()
tn = confusion[0,0]
fp = confusion[0,1]
fn = confusion[1,0]
tp = confusion[1,1]

precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1 = 2*(precision*recall)/(precision+recall)
accuracy = (tp+tn)/(tp+fp+fn+tn)

auc = auroc.compute()

print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nAUCROC: {auc}\nConfusion matrix : {confusion}")