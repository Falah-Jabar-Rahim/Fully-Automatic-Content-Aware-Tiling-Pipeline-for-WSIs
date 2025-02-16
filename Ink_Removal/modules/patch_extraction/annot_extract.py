#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Mon Jul 18 2022 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is about extraction of patches given sedeen annotations
# Modifications (date, what was modified):
#   1.
# --------------------------------------------------------------------------------------------------------------------------
#
import os
import glob
from os.path import exists
from typing import Tuple, List
from pathlib import Path
import random

import numpy as np
from shapely.geometry import Point
import cv2
from tqdm import tqdm

from .process_wsi import ExtractPatches
from .utils.sedeen_helpers import Annotation
from .utils import SedeenAnnotationParser

class ExtractAnnotations(ExtractPatches):
    """
    Extract patches from annotations with labels as given in sedeen
    Paramters:
        sample_threshold: Buffer threshold for sampling points from annotations
    """
    def __init__(self, 
                 image_pth,
                 annotation_dir,
                 renamed_label:dict,
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
                 sample_threshold:int=80,
                 get_template=False,
                 get_coordinates=False):
        
        self.annotation_parser = SedeenAnnotationParser(renamed_label)

        self.all_xmls = list(Path(annotation_dir).glob("*.xml"))
        self.sample_threshold = sample_threshold

        super().__init__(image_pth,
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
                         get_template,
                         get_coordinates)

    def __getitem__(self, index):
        img = self.all_image_tiles_hr[index]
        label = self.all_labels[index]
        if self.transform is not None:
            return self.transform(img), label
        else:
            return img, label

    def tiles_array(self):
        
        all_wsipaths = []
        if isinstance(self.image_path,list):
            #Check if images in list exist
            for i in range(len(self.image_path)):
                if not(exists(self.image_path[i])):
                    raise Exception("WSI file does not exist in: %s" % str(self.image_path[i]))
            all_wsipaths = self.image_path.copy()
        else:
            # Check image
            if not exists(self.image_path):
                raise Exception("WSI file does not exist in: %s" % str(self.image_path))
            if Path(self.image_path).suffix[1:] in ["tif","svs"]:
                all_wsipaths.append(self.image_path)
            for file_ext in ['tif', 'svs']:
                all_wsipaths = all_wsipaths + glob.glob('{}/*.{}'.format(self.image_path, file_ext))
        random.shuffle(all_wsipaths)
        
        #Select subset of slides for training/val setup
        if len(all_wsipaths)>5:
            if self.mode=="train":
                wsipaths = all_wsipaths[:int(self.train_split*len(all_wsipaths))]
            else:
                wsipaths = all_wsipaths[int(self.train_split*len(all_wsipaths)):]
        else:
            wsipaths = all_wsipaths

        self.all_coordinates = []
        
        with tqdm(enumerate(sorted(wsipaths))) as t:

            all_image_tiles_hr = []
            all_labels = []

            for wj, wsipath in t:
                t.set_description(
                    "Loading wsis.. {:d}/{:d}".format(1 + wj, len(wsipaths))
                )

                "generate tiles for this wsi"
                image_tiles_hr, template, labels, coordinates = self.get_wsi_patches(wsipath)
                
                if self.get_template and (self.output_path is not None):
                    cv2.imwrite(str(self.output_path / Path("templates") /f"{Path(wsipath).stem}_template.png"), 255 * (template > 0))

                # Check if patches are generated or not for a wsi
                if len(image_tiles_hr) == 0:
                    print("bad wsi, no patches are generated for", str(wsipath))
                    continue
                else:
                    all_image_tiles_hr.append(image_tiles_hr)
                    all_labels.extend(labels)

                if self.get_coordinates:
                    self.all_coordinates.append(np.array(coordinates))


            # Stack all patches across images
            all_image_tiles_hr = np.concatenate(all_image_tiles_hr)

        self.all_labels = np.array(all_labels)
        
        return all_image_tiles_hr, template
    
    
    def _get_annotations(self, wsipth)->List[Annotation]:
        """
        Gets annotations in xml format based on the slide. Assumes the xml file shares the same name as the name 
        in wsipth
        """
        filename, file_extension = os.path.splitext(Path(wsipth).name)
        indv_annot_pth = list(filter(lambda x: filename in str(x),self.all_xmls))
        annoations = self.annotation_parser.parse(str(indv_annot_pth[0]))
        return annoations
        
    def _in_annotation(self,coords,annotations)->Tuple[bool,Annotation]:
        """
        Determines if a point lies inside any of the annotations
        """
        temp_point = Point(*coords)
        for annots in annotations:
            if annots.geometry.buffer(-self.sample_threshold).contains(temp_point):
                return True, annots
        return False, None

    def get_wsi_patches(self, wsipth):
        "read the wsi scan"
        scan = self._get_slide(wsipth)
        # mask = self._get_mask(wsipth)
        annotations = self._get_annotations(wsipth)

        "downsample multiplier"
        """
        due to the way pyramid images are stored,
        it's best to use the lower resolution to
        specify the coordinates then pick high res.
        from that (because low. res. pts will always
        be on high res image but when high res coords
        are downsampled, you might lose that (x,y) point)
        """

        iw, ih = scan.dimensions
        sh, sw = self.tile_stride_h, self.tile_stride_w
        ph, pw = self.tile_h, self.tile_w

        # self.mask_factor = np.array(scan.dimensions) / np.array(mask.dimensions)

        patch_id = 0
        image_tiles_hr = []
        labels = []
        if self.get_template:
            template = np.zeros(shape=((ih-1-ph-sh)//sh + 1, (iw-1-pw-sw)//sw + 1), dtype=np.float32)
        else:
            template = None

        if self.get_coordinates:
            coordinates = []
        else:
            coordinates = None

        for y,ypos in enumerate(range(sh, ih - 1 - ph, sh)):
            for x,xpos in enumerate(range(sw, iw - 1 - pw, sw)):
                inside, annot = self._in_annotation((xpos,ypos),annotations)
                # if inside and self._isforeground((xpos, ypos), mask):  # Select valid foreground patch and from valid annotation
                if inside:
                    # coords.append((xpos,ypos))
                    image_tile_hr = self._getpatch(scan, xpos, ypos)

                    image_tiles_hr.append(image_tile_hr)
                    labels.append(annot.label["value"] - 1)
                    
                    patch_id = patch_id + 1
                    
                    if self.get_template:
                        #Template filling
                        template[y,x] =  patch_id

                    if self.get_coordinates:
                        coordinates.append((xpos,ypos))

        
        # Concatenate
        if len(image_tiles_hr) == 0:
            image_tiles_hr == []
        else:
            image_tiles_hr = np.stack(image_tiles_hr, axis=0).astype("uint8")

        return image_tiles_hr, template, labels, coordinates