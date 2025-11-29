
import os
import sys
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from Ink_Removal.modules.metrics import ssim, psnr, mse, pbvif
sys.path.append(str(Path(__file__).resolve().parent.parent))


def calc_imagemmetrics(img_ref, img_src):
    ssim_calc = ssim(img_ref, img_src)
    psnr_calc = psnr(img_ref, img_src)
    vif_calc = pbvif(img_ref, img_src)
    return (ssim_calc, psnr_calc, vif_calc)


def create_my_dataset(args, image_tile_path):
    image_tiles_hr = []
    labels = []
    template = None
    img_names = []
    img_number = 0

    for filename in os.listdir(image_tile_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Add more extensions if needed
            image_path = os.path.join(image_tile_path, filename)
            image_tile_dest = Image.open(image_path)
            image_tile_dest = image_tile_dest.resize((args.load_size, args.load_size))
            image_tile_dest = np.array(image_tile_dest)
            if image_tile_dest.shape[2] == 4:
                image_tile_dest = image_tile_dest[:, :, :3]
            if image_tile_dest is None:
                continue
            image_tiles_hr.append(np.stack((image_tile_dest, image_tile_dest)))
            img_names.append(filename)
            img_number = img_number + 1

    if len(image_tiles_hr) == 0:
        image_tiles_hr == []
    else:
        image_tiles_hr = np.stack(image_tiles_hr, axis=0).astype("uint8")

    return image_tiles_hr, template, labels, img_names, img_number



def calculate_color_difference(img1, img2):
    # Convert images to LAB color space
    img1_lab = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    img2_lab = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    # Calculate Euclidean distance between corresponding pixels in LAB color space
    color_diff = np.linalg.norm(img1_lab - img2_lab, axis=2)  # Compute along the channels
    # Calculate the mean of all color differences
    average_color_diff = np.mean(color_diff)
    # Normalize the average color difference to the range [0, 1]
    max_possible_diff = np.sqrt(3) * 255  # Max Euclidean distance in LAB color space
    normalized_average_diff = average_color_diff / max_possible_diff

    return normalized_average_diff



