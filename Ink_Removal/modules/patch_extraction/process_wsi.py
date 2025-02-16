import sys
import os
from os.path import exists
import glob
import random
import warnings

import cv2
import numpy as np
import openslide
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
from pathlib import Path

from .extract_mask import extract_mask

class ExtractPatches(Dataset):
    """
    WSI dataset, This class based on given image path,
    extracts tissue mask at a lower resolution. The image is loaded and converted to the desired spacing as given in the parameter.
    Based on this , points are extracted at a uniform stride and ensured that the patch belongs inside the whole slide
    """

    def __init__(
        self,
        image_pth,
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
        get_template=False,
        get_coordinates=False,
        **kwargs
    ):

        """
        Args:
            image_pth (str): path to wsi.
            tile_h (int): tile height
            tile_w (int): tile width
            tile_stride_factor_h (int): stride height factor, height will be tile_height * factor
            tile_stride_factor_w (int): stride width factor, width will be tile_width * factor
            spacing(float): Specify this value if you want to extract patches at a given spacing
            mask_pth(str): Directory where all masks are stored, if none is given then masks are extracted automatically
            output_pth(str): Directory where all the masks and template if calculated are stored
            mode (str): train or val, split the slides into trainset and val set
            train_split(float): Between 0-1, ratio of split between train and val set
            lwst_level_idx (int): lowest level for patch indexing
            threshold(float): For filtering from mask
        """

        self.image_path = image_pth
        self.output_path = output_pth
        self.spacing = spacing
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_stride_h = int(tile_h*tile_stride_factor_h)
        self.tile_stride_w = int(tile_w*tile_stride_factor_w)
        self.hr_level = lwst_level_idx
        self.mask_path = mask_pth
        self.transform = transform
        self.mode = mode
        self.train_split = train_split
        self.threshold = threshold
        self.get_template = get_template
        self.get_coordinates = get_coordinates

        for key,value in kwargs.items():
            setattr(self,key,value)

        #Get all mask paths if applicable
        if self.mask_path is not None:
            temppth = Path(self.mask_path)
            if temppth.is_dir():
                self.all_masks = list(temppth.glob("*"))
            else:
                print(f"Found {len(self.all_masks)} masks")
                self.all_masks = list(self.mask_path)

        if self.output_path is not None:
            self.output_path = Path(self.output_path)
            if not (self.output_path).is_dir():
                os.mkdir(self.output_path)
            if not (self.output_path/"masks").is_dir():
                os.mkdir(self.output_path/"masks")
            if not (self.output_path/"templates").is_dir():
                os.mkdir(self.output_path/"templates")

        #Load all extracted patches into RAM
        self.all_image_tiles_hr, self.template = self.tiles_array()

        print(f"Extracted {len(self.all_image_tiles_hr)} patches")

    def __len__(self):
        return len(self.all_image_tiles_hr)

    def __getitem__(self, index):
        img = self.all_image_tiles_hr[index]
        if self.transform is not None:
            return self.transform(img)
        else:
            return img

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

            for wj, wsipath in t:
                t.set_description(
                    "Loading wsis.. {:d}/{:d}".format(1 + wj, len(wsipaths))
                )

                "generate tiles for this wsi"
                image_tiles_hr, template, coordinates = self.get_wsi_patches(wsipath)

                if self.get_template and (self.output_path is not None):
                    cv2.imwrite(str(self.output_path / Path("templates") /f"{Path(wsipath).stem}_template.png"), 255 * (template > 0))

                # Check if patches are generated or not for a wsi
                if len(image_tiles_hr) == 0:
                    print("bad wsi, no patches are generated for", str(wsipath))
                    continue
                else:
                    all_image_tiles_hr.append(image_tiles_hr)

                if self.get_coordinates:
                    self.all_coordinates.append(np.array(coordinates))

            # Stack all patches across images
            all_image_tiles_hr = np.concatenate(all_image_tiles_hr)
        
        return all_image_tiles_hr, template
    
    def _get_mask(self, wsipth):
        #If mask path not available, calulates the mask
        if self.mask_path is None:
            tissue_mask = extract_mask(wsipth,threshold=0.1,kernel_size=9)
            mask_pil = Image.fromarray(255 * tissue_mask)
            if self.output_path is not None:
                mask_pil.save(str(str(self.output_path / Path("masks") / f"{Path(wsipth).stem}_mask_image.png")))
            mask = openslide.ImageSlide(mask_pil)
        else:
            filename, file_extension = os.path.splitext(Path(wsipth).name)
            indv_mask_pth = list(filter(lambda x: filename in str(x),self.all_masks))
            mask = io.imread(str(indv_mask_pth[0]))
            # mask = Image.fromarray(mask).convert("RGB")
            # mask_processed = openslide.ImageSlide(mask)
            # del mask

        return mask

    def _getpatch(self, scan, x, y):

        'read low res. image'
        'hr patch'
        image_tile_hr = scan.read_region((x, y), self.hr_level, (self.tile_w, self.tile_h)).convert('RGB')
        image_tile_hr = np.array(image_tile_hr).astype('uint8')

        return image_tile_hr

    def _get_slide(self, wsipth):
        """
        Returns openslide object based on the provided scaling in self.spacing
        """
        scan = openslide.OpenSlide(wsipth)
        # Getting the original slide spacing, different slides have different fields populated
        if self.spacing is None:
            #if user doesnt want to resize the slides
            return scan
        else:
            if "tiff.XResolution" in scan.properties.keys():
                slide_spacing = 1 / (float(scan.properties["tiff.XResolution"]) / 10000)
            elif "openslide.mpp-x" in scan.properties.keys():
                slide_spacing = np.float(scan.properties["openslide.mpp-x"])
            else:
                raise ValueError("Not able to find spacing")
            # Rescaling the slide
            resize_factor = slide_spacing / self.spacing

            if (resize_factor - 1) <= 0.2:
                # Spacing almost same
                return scan
            else:
                print("Resizing the slide...")
                img_temp = io.imread(wsipth)
                img_temp = Image.fromarray(img_temp).convert("RGB")
                new_size = (np.array((img_temp.size)) * resize_factor).astype(int)
                img_resize = img_temp.resize(new_size, Image.ANTIALIAS)
                scan = openslide.ImageSlide(img_resize)
                del img_temp, img_resize
                return scan

    def get_wsi_patches(self, wsipth):
        "read the wsi scan"
        scan = self._get_slide(wsipth)
        mask = self._get_mask(wsipth)
        # scan = openslide.OpenSlide(wsipth)

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

        if self.mask_path is None:
            self.mask_factor = np.array(scan.dimensions) / np.array(mask.dimensions)

        patch_id = 0
        image_tiles_hr = []
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
                if self._isforeground((xpos, ypos), mask):  # Select valid foreground patch
                    # coords.append((xpos,ypos))
                    image_tile_hr = self._getpatch(scan, xpos, ypos)

                    image_tiles_hr.append(image_tile_hr)
                    
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
        
        return image_tiles_hr, template, coordinates

    def _isforeground(self, coords, mask):
        if self.mask_path is None:
            coords_resize = (coords / self.mask_factor).astype(int)
            dim_resize = tuple(([self.tile_w, self.tile_h] / self.mask_factor).astype(int))
            patch = np.array(mask.read_region(coords_resize, self.hr_level, dim_resize).convert("L"))
            if np.max(patch)>1:
                max_val = 255
            else:
                max_val = 1
            perc = np.sum((np.array(patch) / max_val)) / (dim_resize[0] * dim_resize[1])
            return perc >= self.threshold
        else:
            x,y = coords
            patch = mask[y:y+self.tile_w,x:x+self.tile_w]
            return (np.sum(patch)/float(self.tile_w*self.tile_h))>=self.threshold
