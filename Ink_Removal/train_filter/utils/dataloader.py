from pathlib import Path
import sys
import random
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
# import openslide
from tqdm import tqdm
import torch.utils.data as data
# from skimage import io
from PIL import Image
import cv2
import random

from modules.patch_extraction import ExtractPatches, ExtractAnnotations
from .inkgeneration import InkGenerator


class Handwritten(data.Dataset):
    ''' Gets the handwritten dataset, but can be used for any random simple image dataset, whose data can be loaded into the memory'''
    def __init__(self,path,n=10000,transform=None):
        super(Handwritten,self).__init__()
        self.path = path
        self.master_path = list(Path(self.path).glob("**/*.png"))
        self.transform = transform
        random.shuffle(self.master_path)
        self.master_path = self.master_path[:n]
        self.data = self.read_data()

    def __len__(self):
        return len(self.master_path)
    
    def __getitem__(self,index):
        if self.transform!=None:
            return self.transform(Image.fromarray(self.data[index]))
        else:
            return self.data[index]
    
    def read_data(self):
        print("Loading the data...")
        data = []
        for i in tqdm(range(len(self.master_path))):
            path = str(self.master_path[i])
            data.append(cv2.imread(path,cv2.IMREAD_GRAYSCALE))
        return data


class Vectorize_WSIs(ExtractPatches):
    """ WSI dataset preparation for ink filter for TIGER"""
    def __init__(self, 
                 image_pth,
                 handwritten_obj, 
                 tile_h, 
                 tile_w, 
                 tile_stride_factor_h, 
                 tile_stride_factor_w,
                 colors, 
                 spacing=None, 
                 mask_pth=None, 
                 output_pth=None, 
                 lwst_level_idx=0, 
                 mode="train", 
                 train_split=0.8, 
                 threshold=0.7, 
                 transform=None):
        """
        Args:
            image_pth (str): path to wsi/folder of wsi.
            mask_pth(str): path to mask folder
            handwritten_obj(Handwritten): handwritten object
            tile_h (int): tile height
            tile_w (int): tile width
            tile_stride_factor_h (int): stride height factor, height will be tile_height * factor
            tile_stride_factor_w (int): stride width factor, width will be tile_width * factor
            colors (List[Tuple(str,str),]): List of tuples consisting of two colors, giving a range to color
            lwst_level_idx (int): lowest level for patch indexing
            mode (str): train or val, split the slides into trainset and val set
            train_split(float): Between 0-1, ratio of split between train and val set
        """
        self.colors = colors
        #For ink stains
        self.ink_templates = handwritten_obj
        self.n_templ = len(self.ink_templates)
        self.ink_generator = InkGenerator(ink_template=self.ink_templates,
                                          colors=self.colors
                                         )

        super().__init__(image_pth, tile_h, tile_w, tile_stride_factor_h, tile_stride_factor_w, spacing, mask_pth, output_pth, lwst_level_idx, mode, train_split, threshold, transform)

    def __getitem__(self, index):
        img =  self.all_image_tiles_hr[index]
        #Get fake images
        gen_image,label = self.add_inkstain(img)
        if self.transform is not None:
            return self.transform(Image.fromarray((gen_image*255).astype(np.uint8))), label
        else:
            return gen_image, label

    def add_inkstain(self,img):
        """
        For adding artificial ink stains on a given image
        """
        #For classification
        p = torch.rand(1).item()
        if p<0.5: #50% chance for clean and ink stained data
            label = 0
            noise_img = img.copy()/255
        else:
            _,_,noise_img,_,_,_ = self.ink_generator.generate(img)
            label = 1
        
        return noise_img, label
