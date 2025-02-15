import glob
import random
from os.path import exists
from typing import Tuple

import cv2
import torchvision
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ..register import Pairwise_Extractor
from .process_wsi import ExtractPatches


class Pairwise_ExtractPatches(ExtractPatches):
    def __init__(self,
                 pair_pths,
                 tile_h,
                 tile_w,
                 tile_stride_factor_h, 
                 tile_stride_factor_w, 
                 spacing=None, 
                 mask_pth=None, 
                 output_pth=None, 
                 lwst_level_idx=0, 
                 mode="train", 
                 train_split=0.8, 
                 threshold=0.7, 
                 transform=None,
                 get_template=False):
        """
        Based on given pair of paths, registers the pairs, and extracts patchs at both the slides in the same location
        Parameters:
            pair_pths (List[Tuple[str,str]]): List of pairs of tuples containing paths of src and destination slide
        """
        super().__init__(pair_pths, 
                         tile_h, 
                         tile_w, 
                         tile_stride_factor_h, 
                         tile_stride_factor_w, 
                         spacing, 
                         mask_pth, 
                         output_pth, 
                         lwst_level_idx, 
                         mode, 
                         train_split, 
                         threshold, 
                         transform,
                         get_template)

    def __getitem__(self, index):
        dest_img, src_img = self.all_image_tiles_hr[index]
        if self.transform is not None:
            return self.transform(dest_img), self.transform(src_img)
        else:
            return dest_img, src_img

    def tiles_array(self):
        # Check image
        if isinstance(self.image_path,tuple):
            all_wsipaths = [self.image_path]
        elif isinstance(self.image_path,list):
            all_wsipaths = self.image_path
        else:
            raise ValueError("Pass pair of WholeSlideImages as list of tuples or single tuple")

        #Select subset of slides for training/val setup
        if len(all_wsipaths)>5:
            if self.mode=="train":
                wsipaths = all_wsipaths[:int(self.train_split*len(all_wsipaths))]
            else:
                wsipaths = all_wsipaths[int(self.train_split*len(all_wsipaths)):]
        else:
            wsipaths = all_wsipaths

        with tqdm(enumerate(sorted(wsipaths))) as t:

            all_image_tiles_hr = []

            for wj, wsipath in t:
                t.set_description(
                    "Loading wsis.. {:d}/{:d}".format(1 + wj, len(wsipaths))
                )
                
                "generate tiles for this wsi"
                image_tiles_hr, template = self.get_wsi_patches(wsipath)

                # Check if patches are generated or not for a wsi
                if len(image_tiles_hr) == 0:
                    print("bad wsi, no patches are generated for", str(wsipath))
                    continue
                else:
                    all_image_tiles_hr.append(image_tiles_hr)

            # Stack all patches across images
            all_image_tiles_hr = np.concatenate(all_image_tiles_hr)

        if self.get_template and (self.output_path is not None):
            cv2.imwrite(str(Path(self.output_path) / "template.png"), 255 * (template > 0))
        
        return all_image_tiles_hr, template

    def get_wsi_patches(self, wsipth:Tuple[str,str])->Tuple[np.array,np.array]:
        """
        For given set of src and destination slide, this function registers and extract patches
        according to the tissue map of destination slide.
        Parameters:
            wsipth (Tuple[str,str]): Tuple of source and destination slide paths
        Returns:
            image_tiles_hr (np.array N*2*H*W*3): All the patches extracted as numpy array as pairs of destination patch
                                                 and source patches
            template (np.array): track of extracted patches in a 2d format. Usefull for plotting predictions later on
        """
        
        "read the wsi scan"
        src_slide_pth, dest_slide_pth = wsipth
        #Perform registration
        patch_extractor = Pairwise_Extractor.from_path(src_path=src_slide_pth, dest_path=dest_slide_pth)       
        
        #Get the mask from dest_slide
        mask = self._get_mask(dest_slide_pth)
        
        #Get the mask from src slide and warp it to dest_slide
        # mask = self._get_mask(src_slide_pth)
        
        "downsample multiplier"
        """
        due to the way pyramid images are stored,
        it's best to use the lower resolution to
        specify the coordinates then pick high res.
        from that (because low. res. pts will always
        be on high res image but when high res coords
        are downsampled, you might lose that (x,y) point)
        """

        iw, ih = patch_extractor.dest_slide.dimensions
        sh, sw = self.tile_stride_h, self.tile_stride_w
        ph, pw = self.tile_h, self.tile_w

        self.mask_factor = np.array(patch_extractor.dest_slide.dimensions) / np.array(mask.dimensions)

        patch_id = 0
        image_tiles_hr = []
        
        if self.get_template:
            template = np.zeros(shape=((ih-1-ph-sh)//sh + 1, (iw-1-pw-sw)//sw + 1), dtype=np.float32)
        else:
            template = None

        for y,ypos in enumerate(range(sh, ih - 1 - ph, sh)):
            for x,xpos in enumerate(range(sw, iw - 1 - pw, sw)):
                if self._isforeground((xpos, ypos), mask):  # Select valid foreground patch
                    # coords.append((xpos,ypos))
                    image_tile_dest,_,image_tile_src = patch_extractor.extract(xpos,ypos,(pw,ph))

                    if image_tile_dest is None:
                        continue
                    
                    image_tiles_hr.append(np.stack((image_tile_dest,image_tile_src)))

                    patch_id = patch_id + 1
                    
                    if self.get_template:
                        #Template filling
                        template[y,x] =  patch_id
        
        # Concatenate
        if len(image_tiles_hr) == 0:
            image_tiles_hr == []
        else:
            image_tiles_hr = np.stack(image_tiles_hr, axis=0).astype("uint8")

        return image_tiles_hr, template