import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
import torch.utils.data as data
import os, math, torch
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import pandas as pd

import modules
from modules.patch_extraction import Pairwise_ExtractAnnot
from data.base_dataset import BaseDataset, get_transform


class PairinkDataset(BaseDataset, Pairwise_ExtractAnnot):
    """A template dataset class for you to implement custom datasets."""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.add_argument('--mode',type=str,default="train",help="Train/Test")
        parser.add_argument('--stride_h',type=float,default=5,help="Stride factor with tile size 256 in y direction")
        parser.add_argument('--stride_w',type=float,default=5,help="Stride factor with tile size 256 in x direction")
        parser.add_argument('--pair_csv',type=str,default="test.xlsx",help="csv path for pair of data")
        parser.add_argument('--ink_slide_pth',default="images", type=str,help="path for all ink slides")
        parser.add_argument('--clean_slide_pth',default="images", type=str,help="path for all clean slides")
        if is_train==False:
            parser.set_defaults(mode="test")  # specify dataset-specific default values

        return parser
    
    def __init__(self, 
                 opt, 
                 tile_h=256, 
                 tile_w=256, 
                 lwst_level_idx=0, 
                 mode="train", 
                 train_split=1, 
                 threshold=0.7, 
                 transform=transforms.ToTensor()
                 ):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        df = pd.read_excel(opt.pair_csv)
        ink_slide_path = opt.ink_slide_pth
        clean_path = opt.clean_slide_pth

        pair_list = [(str( Path(clean_path) / (str(df["Clean Slides"][i])+".svs" ) ),str( Path(ink_slide_path) / (str(df["Ink Slides"][i])+".svs" ) ))
                for i in range(len(df))]
        annotation_dir = str( Path(ink_slide_path) / Path("sedeen") )
        #For the experiments I have provided annotations for faster processing
        ink_labelset = {"clean":"#00ff00ff","ink":"#ff0000ff"}

        BaseDataset.__init__(self, opt)

        self.do_norm = opt.do_norm
        
        Pairwise_ExtractAnnot.__init__(self,
                                pair_pths=pair_list,
                                annotation_dir=annotation_dir,
                                renamed_label=ink_labelset,
                                tile_h=tile_h,
                                tile_w=tile_w,
                                tile_stride_factor_h=opt.stride_h, 
                                tile_stride_factor_w=opt.stride_w, 
                                lwst_level_idx=lwst_level_idx, 
                                mode=mode, 
                                train_split=train_split, 
                                transform=transform,
                                threshold=threshold,
                                sample_threshold=50
                                )
        
        all_labels = np.array(self.all_labels)
        all_labels_1 = np.where(all_labels==1)[0]
        all_labels_0 = np.where(all_labels==0)[0]
        np.random.shuffle(all_labels_1)
        np.random.shuffle(all_labels_0)

        #Get equal number of postive and negative samples for testing
        self.all_lab_shuff_idx = np.concatenate((all_labels_1[:opt.num_test//2],all_labels_0[:opt.num_test//2]))
        np.random.shuffle(self.all_lab_shuff_idx)
        print(f"Length of dataset: {len(self.all_lab_shuff_idx)}")

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        index = self.all_lab_shuff_idx[index]
        ink_img, clean_img = self.all_image_tiles_hr[index]
        label = self.all_labels[index]
        #Get images and transform
        data_A =  self.transform(Image.fromarray(ink_img))
        data_B =  self.transform(Image.fromarray(clean_img))
        
        return {'A': self.normalize(data_A), 'B': self.normalize(data_B), 'A_paths': "tiger_dataset_{}_{}".format(label,index), 'B_paths':  "tiger_dataset_{}_{}".format(label,index), 'label': label}

    def add_inkstain(self,img):
        """
        For adding artificial ink stains on a given image
        """
        #For classification
        p = torch.rand(1).item()
        if p<0.3: #30% chance for clean and ink stained data
            label = 0
            noise_img = img.copy()/255
        else:
            _,_,noise_img,_,_,_ = self.ink_generator.generate(img)
            label = 1
        
        return noise_img, label
    
    def __len__(self):
        """Return the total number of images."""
        return len(self.all_lab_shuff_idx)

    def normalize(self,img):
        if self.do_norm:
            return 2*img - 1
        else:
            return img

