#Adapted from Lukas' code

import numpy as np
import cv2
import openslide
from skimage import color
from scipy.ndimage.morphology import binary_fill_holes

def extract_mask(wsi_path, threshold=0.1, kernel_size=9):
    scan = openslide.OpenSlide(wsi_path)
    #get levels and the dimensions
    level_dimensions = scan.level_dimensions
    for i,levels in enumerate(level_dimensions[::-1]):
        #Select level with dimension around 1000 otherwise it becomes too big
        if levels[0]>1000 or levels[1]>1000:
            break
    img = scan.read_region((0,0),len(level_dimensions)-1,levels).convert('RGB')
    #Apply threshold on LAB
    img=np.array(img).astype('uint8')
    #get mask
    lab = color.rgb2lab(img)
    mu = np.mean(lab[..., 1])
    lab = lab[..., 1] > (1 + threshold ) * mu
    mask = lab.astype(np.uint8)
    mask = binary_fill_holes(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = mask.astype(np.uint8)
    return mask
