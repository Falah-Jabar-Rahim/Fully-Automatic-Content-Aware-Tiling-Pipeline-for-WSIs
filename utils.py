import math
import shutil
import cv2
import torch
from matplotlib import pyplot as plt
from medpy import metric
import torch.nn as nn
from PIL import Image
import os
from collections import Counter
from skimage.morphology import remove_small_objects
from torch.autograd import Variable
import time
from scipy.ndimage import binary_fill_holes, binary_dilation, label, find_objects
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.multiprocessing as mp
import pyvips as vips
from openslide import open_slide
import multiprocessing
import random
from datasets.dataset import DHUnet_dataset
import logging
import numpy as np
from inti_penmarker import create_my_dataset

from Ink_Removal.wsi_tile_cleanup.utils.pyvips import split_rgb, split_rgb_torch
from Ink_Removal.wsi_tile_cleanup.filters.pens import pen_percent, pen_percent_torch
from Ink_Removal.wsi_tile_cleanup.filters import blackish_percent
import pyvips
from multiprocessing import Pool
import concurrent.futures


def TimeOnCuda():
    torch.cuda.synchronize()
    return time.time()


def get_dataloader(args, fold_no=0, total_fold=5, split="train", batch_size=1, shuffle=False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    db_data = DHUnet_dataset(list_dir=args.list_dir, split=split, fold_no=fold_no, total_fold=total_fold,
                             img_size=args.img_size)
    logging.info("The length of {} {} set is: {}".format(args.dataset, split, len(db_data)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    dataloader = DataLoader(db_data, batch_size=batch_size, shuffle=shuffle, num_workers=8, pin_memory=True,
                            worker_init_fn=worker_init_fn)
    return dataloader


class DiceLoss(nn.Module):
    def __init__(self, n_classes, weight=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weight

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        # print(score.shape, target.shape)
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if self.weight is None:
            self.weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weight[i]
        return loss / self.n_classes


def calculate_IoU_binary(y_pred, y_true):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = 1e-9
    iou = (intersection + smooth) / (union + smooth)
    return iou


def calculate_Dice_binary(y_pred, y_true):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true))
    mask_sum = np.sum(np.abs(y_true)) + np.sum(np.abs(y_pred))

    smooth = 1e-9
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice


def calculate_F1_binary(pred, true):
    """
    F1 score:
        Accuracy =(TP+TN)/(TP+TN+FP+FN)
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1 = 2*(Precision*Recall)/(Precision+Recall)
    """
    epsilon = 1e-9
    TP = true * pred
    FP = pred ^ TP
    FN = true ^ TP
    precision = TP.sum() / (TP.sum() + FP.sum() + epsilon)
    recall = TP.sum() / (TP.sum() + FN.sum() + epsilon)
    F1 = (2 * precision * recall) / (precision + recall + epsilon)
    return F1


def calculate_Acc_binary(y_pred, y_true):
    """
    compute accuracy for binary segmentation map via numpy
    """
    w, h = y_pred.shape
    smooth = 1e-9
    acc = (np.sum(y_true == y_pred) + smooth) / (h * w + smooth)
    return acc


import numpy as np
import cv2




def fill_holes_wsi_seg(mask):
    # Create a copy of the mask to fill the holes
    filled_mask = mask.copy()
    # Get the height and width of the mask
    height, width, _ = mask.shape
    # Define the 8-neighbor directions (up, down, left, right, and 4 diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            # Check if the current pixel is a hole (value [0, 0, 0])
            if np.array_equal(mask[y, x], [0, 0, 0]):
                neighbor_values = []
                # Check all 8 neighbors
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_value = tuple(mask[ny, nx])
                        if neighbor_value != (0, 0, 0):  # Avoid including holes in the neighbors
                            neighbor_values.append(neighbor_value)

                # If there are valid neighbors, fill the current pixel
                if neighbor_values:
                    most_common_value = Counter(neighbor_values).most_common(1)[0][0]
                    filled_mask[y, x] = most_common_value

    return filled_mask


def calculate_metric_perpatch(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = calculate_Dice_binary(pred, gt)
        yc = metric.binary.jc(pred, gt)  # calculate_IoU_binary(pred, gt) # metric.binary.jc(pred, gt) # jaccard == iou
        acc = calculate_Acc_binary(pred, gt)
        M = [dice, yc, acc]
        return M
    elif pred.sum() == 0 and gt.sum() == 0:
        M = [1, 1, 1]
        return M
    elif pred.sum() == 0 and gt.sum() > 0:
        M = [0, 0, 0]
        return M
    else:  # pred.sum() > 0 and gt.sum() == 0:
        M = [np.nan, np.nan, np.nan]
        return M


def make_cm(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    return cm


def validate_single_patch(image, label, net, classes, test_save_path=None, case=None, network="DHUnet"):
    label = label.squeeze(0).cpu().detach().numpy()
    image = image.cuda()
    net.eval()
    with torch.no_grad():
        if network == "DHUnet":
            out = torch.argmax(torch.softmax(net(image, image)[0], dim=1), dim=1).squeeze(0)
        else:
            out = torch.argmax(torch.softmax(net(image), dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()

    if test_save_path is not None:
        save_pred_path = test_save_path + '/' + case[0].split('.')[0] + '.png'
        if not os.path.exists(os.path.dirname(save_pred_path)):
            os.makedirs(os.path.dirname(save_pred_path))
        print(save_pred_path)
        mask = Image.fromarray(np.uint32(prediction))
        mask.save(save_pred_path)

    metric = []
    for i in range(1, classes):
        if (label == i).sum() > 0:
            metric.append(calculate_metric_perpatch(prediction == i, label == i))
        else:
            metric.append([np.NaN, np.NaN, np.NaN])
    return metric


def find_Tissue_regions(wsi_path, thumbnail_size, tile_size,level,  plot=False):
    # load wsi image
    wsi = open_slide(wsi_path)
    #wsi_width, wsi_height = wsi.dimensions
    wsi_width, wsi_height = wsi.level_dimensions[level]
    # get wsi thumbnail
    thumbnail = wsi.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_h = thumbnail.height
    thumbnail_w = thumbnail.width
    # compute scaling factor
    sf_h = wsi_height / thumbnail_h
    sf_w = wsi_width / thumbnail_w
    thumbnail_np = np.array(thumbnail)
    # apply Otsu thresholding to detect tissue regions
    gray_image = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    _, otsu_threshold = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu_threshold == 0
    # remove small objects
    cleaned_mask = remove_small_objects(mask, min_size=25)
    structure_8connectivity = np.ones((4, 4), dtype=bool)
    binary_mask = binary_dilation(cleaned_mask, structure=structure_8connectivity)
    # Find the indices of the non-zero elements
    non_zero_indices = np.argwhere(binary_mask)
    # Find the minimum and maximum x and y coordinates on thumbnail image
    ymin, xmin = non_zero_indices.min(axis=0)
    ymax, xmax = non_zero_indices.max(axis=0)
    image_with_bbox = thumbnail_np.copy()
    cv2.rectangle(image_with_bbox, (xmin, ymin), (xmax, ymax), (0, 255, 0), 25)  # Blue box with thickness 2

    if plot:
        plt.imshow(mask)
        plt.show()
        plt.imshow(cleaned_mask)
        plt.show()
        plt.imshow(binary_mask)
        plt.show()
        plt.imshow(image_with_bbox)
        plt.show()

    # Find the minimum and maximum x and y coordinates on original wsi image
    xmin = int(xmin * sf_w)
    xmax = int(xmax * sf_w)
    ymin = int(ymin * sf_h)
    ymax = int(ymax * sf_h)

    if 0 <= xmin <= tile_size:
        xmin_indx = 0
    else:
        xmin_indx = (xmin // tile_size)
    if 0 <= ymin <= tile_size:
        ymin_indx = 0
    else:
        ymin_indx = (ymin // tile_size)
    if xmax + tile_size < wsi_width:
        xmax_indx = (xmax // tile_size)
    else:
        xmax_indx = (xmax // tile_size)
    if ymax + tile_size < wsi_height:
        ymax_indx = (ymax // tile_size)
    else:
        ymax_indx = (ymax // tile_size)

    return thumbnail_np, binary_mask, image_with_bbox, xmin_indx, ymin_indx, xmax_indx, ymax_indx, sf_w, sf_h


def create_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # If it exists, remove the folder and its contents
        shutil.rmtree(folder_path)
        print(f'Deleted existing folder: {folder_path}')
    # Create a new folder
    os.makedirs(folder_path)
    print(f'Created new folder: {folder_path}')


def extract_small_patches(patch, patch_size):
    """
    Extract smaller patches from a larger patch.

    Parameters:
    - patch: The larger patch (H x W x C) as a numpy array.
    - patch_size: Tuple (patch_width, patch_height) specifying the size of the smaller patches.

    Returns:
    - small_patches: NumPy array of smaller patches (shape: (num_patches, patch_size[0], patch_size[1], channels)).
    """
    patch_w, patch_h = patch_size
    height, width, _ = patch.shape

    # Calculate number of smaller patches
    num_patches_x = (width - patch_w) // patch_w + 1
    num_patches_y = (height - patch_h) // patch_h + 1

    # Create an array to hold all small patches
    small_patches = np.zeros((num_patches_x * num_patches_y, patch_h, patch_w, patch.shape[2]), dtype=np.uint8)

    index = 0
    for y in range(0, height - patch_h + 1, patch_h):
        for x in range(0, width - patch_w + 1, patch_w):
            small_patches[index] = patch[y:y + patch_h, x:x + patch_w]
            index += 1

    return small_patches, num_patches_x * num_patches_y


def crop(region, patch_size, x, y, level):
    return region.read_region((x, y), level, (patch_size, patch_size))


def extract_and_save_patch(y_cord, file_path, file_name, patch_folder, patch_size, xmin_indx, xmax_indx, overlap,
                           level):
    # load wsi image
    slide = open_slide(file_path)
    f_name = file_name.split(".")[0]
    # compute overlapping thr for tiling
    ovr_th = int(((100-overlap) / 100) * patch_size)
    # iterate over columns direction (x)
    for x_cord in range(xmin_indx, xmax_indx):
        # handle borders
        if x_cord > 0 and y_cord > 0:
            x1 = patch_size * x_cord
            y1 = patch_size * y_cord
            # center patch
            patch_c = crop(slide, patch_size, x1, y1, level)
            base_name = f"{f_name}_{x1}_{y1}_center.png"
            patch_c_rgb = patch_c.convert('RGB')
            patch_c_rgb.save(os.path.join(patch_folder, base_name))
            ## left patch
            patch_l = crop(slide, patch_size, x1 - ovr_th, y1, level)
            base_name = f"{f_name}_{x1 - ovr_th}_{y1}_left.png"
            patch_l_rgb = patch_l.convert('RGB')
            patch_l_rgb.save(os.path.join(patch_folder, base_name))
            ## right patch
            patch_r = crop(slide, patch_size, x1 + ovr_th, y1, level)
            base_name = f"{f_name}_{x1 + ovr_th}_{y1}_right.png"
            patch_r_rgb = patch_r.convert('RGB')
            patch_r_rgb.save(os.path.join(patch_folder, base_name))
            ## top patch
            patch_u = crop(slide, patch_size, x1, y1 - ovr_th, level)
            base_name = f"{f_name}_{x1}_{y1 - ovr_th}_top.png"
            patch_u_rgb = patch_u.convert('RGB')
            patch_u_rgb.save(os.path.join(patch_folder, base_name))
            ## down patch
            patch_d = crop(slide, patch_size, x1, y1 + ovr_th, level)
            base_name = f"{f_name}_{x1}_{y1 + ovr_th}_down.png"
            patch_d_rgb = patch_d.convert('RGB')
            patch_d_rgb.save(os.path.join(patch_folder, base_name))


def calculate_percentages(bands, colors):
    perc = []

    # Prepare arguments for multiprocessing
    args = [(bands, color) for color in colors]

    # Create a Pool object to manage multiple processes
    with Pool() as pool:
        # Map the pen_percent function to the arguments
        results = pool.map(pen_percent, args)

        # Process results
        for result in results:
            color, percentage = result
            perc.append(percentage)

    return perc


def pen_marker(args_artifact, args, exel_path, model_penmarker, img_path, cleaned_img_path, max_marker_thr, min_marker_gb_thr,
               clean_penmarker,exel_name):
    # get image tiles list
    image_tiles_hr, template, labels, img_names, img_number = create_my_dataset(args, img_path)
    # get the number of batches
    batch_size = args.batch_size
    num_batches = math.ceil(img_number / batch_size)  # Ensure we cover all images
    pen_stats = [["tile", "classification"]]

    # iterate over the batches: do pen marker detection and removal (use GPU)
    for batch_num in range(num_batches):
        # Calculate the start and end indices for this batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, img_number)  # Ensure we don't go out of bounds
        # Get the current batch of image and their names
        current_batch = image_tiles_hr[start_idx:end_idx]
        current_batch_names = img_names[start_idx:end_idx]
        # unpack data from data loader and fed to the model
        model_penmarker.set_input(current_batch)
        model_penmarker.test()  # run inference for pen marker removal
        visuals = model_penmarker.get_current_visuals()
        # pen marker detection
        real_ink = visuals["real_A"] * 255
        ink_removed = visuals["fake_B"]* 255
        # Clamp the values to ensure they are within the range [0, 255] and convert to uint8
        real_ink = real_ink.clamp(0, 255).to(torch.uint8)
        ink_removed = ink_removed.clamp(0, 255).to(torch.uint8)

        bands = split_rgb_torch(real_ink)
        colors = ["red", "green", "blue", "black"]
        results = {color: pen_percent_torch(bands, color) for color in colors}
        red_per=results["red"].cpu().numpy()
        green_per=results["green"].cpu().numpy()
        blue_per=results["blue"].cpu().numpy()
        black_per=results["black"].cpu().numpy()

        # classify tiles to low, medium, and high pen marker
        for i in range(len(red_per)): # iterate over images in current batch
            perc = np.array([green_per[i], blue_per[i], black_per[i]]) # ignor red pen marker
            # compute max, dominated color
            perc_max = perc.max()
            name = current_batch_names[i]
            # apply thresholding
            if clean_penmarker:
                if perc_max >= max_marker_thr:  # strong pen marker, remove this tile
                    real_ink_np = np.asarray(real_ink[i].permute(1, 2, 0).cpu().numpy(), np.uint8)
                    real_ink_pl = Image.fromarray(real_ink_np)
                    real_ink_pl_rgb = real_ink_pl.convert('RGB')
                    real_ink_pl_rgb = real_ink_pl_rgb.resize((args_artifact.img_size, args_artifact.img_size), Image.BILINEAR)
                    real_ink_pl_rgb.save(os.path.join(cleaned_img_path, name))
                    pen_stats.append([name, "remove"])
                elif perc[0] <= min_marker_gb_thr and perc[1] <= min_marker_gb_thr and perc[
                    2] <= min_marker_gb_thr + 0.05:  # no sign of pen marker, keep original tile
                    real_ink_np = np.asarray(real_ink[i].permute(1, 2, 0).cpu().numpy(), np.uint8)
                    real_ink_pl = Image.fromarray(real_ink_np)
                    real_ink_pl_rgb = real_ink_pl.convert('RGB')
                    real_ink_pl_rgb = real_ink_pl_rgb.resize((args_artifact.img_size, args_artifact.img_size), Image.BILINEAR)
                    real_ink_pl_rgb.save(os.path.join(cleaned_img_path, name))
                    pen_stats.append([name, "keep"])
                else:  # has pen marker keep the cleaned tile
                    ink_removed_np = np.asarray(ink_removed[i].permute(1, 2, 0).cpu().numpy(), np.uint8)
                    ink_removed_pl = Image.fromarray(ink_removed_np)
                    ink_removed_pl_rgb = ink_removed_pl.convert('RGB')
                    ink_removed_pl_rgb = ink_removed_pl_rgb.resize((args_artifact.img_size, args_artifact.img_size), Image.BILINEAR)
                    ink_removed_pl_rgb.save(os.path.join(cleaned_img_path, name))
                    pen_stats.append([name, "keep"])
            else:
                if perc_max >= (max_marker_thr / 2):  # strong pen marker, remove this tile
                    real_ink_np = np.asarray(real_ink[i].permute(1, 2, 0).cpu().numpy(), np.uint8)
                    real_ink_pl = Image.fromarray(real_ink_np)
                    real_ink_pl_rgb = real_ink_pl.convert('RGB')
                    real_ink_pl_rgb = real_ink_pl_rgb.resize((args_artifact.img_size, args_artifact.img_size), Image.BILINEAR)
                    real_ink_pl_rgb.save(os.path.join(cleaned_img_path, name))
                    pen_stats.append([name, "remove"])
                else:  # no sign of pen marker, keep original tile
                    real_ink_np = np.asarray(real_ink[i].permute(1, 2, 0).cpu().numpy(), np.uint8)
                    real_ink_pl = Image.fromarray(real_ink_np)
                    real_ink_pl_rgb = real_ink_pl.convert('RGB')
                    real_ink_pl_rgb = real_ink_pl_rgb.resize((args_artifact.img_size, args_artifact.img_size), Image.BILINEAR)
                    # real_ink_pl_rgb.save(os.path.join(cleaned_img_path, name + "_keep.png"))
                    real_ink_pl_rgb.save(os.path.join(cleaned_img_path, name))
                    pen_stats.append([name, "keep"])

    df = pd.DataFrame(pen_stats[1:], columns=pen_stats[0])
    excel_file_path = os.path.join(exel_path, exel_name)
    df.to_excel(excel_file_path, index=False)


def filter_penmarker(penmarker_st: pd.DataFrame, qury: str) -> str:
    name_list = penmarker_st['tile'].tolist()
    classif = penmarker_st['classification'].tolist()
    try:
        index = name_list.index(qury)
    except ValueError:
        raise ValueError(f"Image '{qury}' not found in the list.")
    qury_class = classif[index]
    return qury_class

def tile_scores(seg_stats_sort, penmarker_stats_sort, bc, fo, bl, qury):

    name_list_seg = seg_stats_sort['tile'].tolist()
    name_list_pen = penmarker_stats_sort['tile'].tolist()

    try:
        index_seg = name_list_seg.index(qury)
        index_pen = name_list_pen.index(qury)

    except ValueError:
        raise ValueError(f"Image '{qury}' not found in the list.")
    bc_pct = seg_stats_sort["%background"][index_seg]
    # ts_pct = seg_stats_sort["%tissue"][index]
    fo_pct = seg_stats_sort["%fold"][index_seg]
    bl_pct = seg_stats_sort["%blur"][index_seg]
    classifi = seg_stats_sort["classification"][index_seg]

    pen_marker = penmarker_stats_sort["classification"][index_pen]

    return bc*(bc_pct/100) + fo*(fo_pct/100) + bl*(bl_pct/100), pen_marker, classifi


def tiles_selection(thumbnail,  sf_w, sf_h,overlap, seg_stats_name, penmarker_stats_name, Tile_folder_seg,
                    Tile_folder_clean,Selected, exel_path,  bc=1, fo=1, bl=1, tilesize=270):

    thumbnail_h, thumbnail_w, _ = thumbnail.shape
    ovr_th = int(((100-overlap) / 100) * tilesize)

    wsi_seg = np.zeros((thumbnail_h, thumbnail_w, 3), dtype=np.uint8)
    wsi_seg_qt = np.zeros((thumbnail_h, thumbnail_w, 3), dtype=np.uint8)
    wsi_clean_tiles = thumbnail.copy()

    seg_stats = pd.read_excel(os.path.join(exel_path, seg_stats_name))
    penmarker_stats = pd.read_excel(os.path.join(exel_path, penmarker_stats_name))

    seg_stats_sort = seg_stats
    penmarker_stats_sort = penmarker_stats
    seg_stats_list = seg_stats_sort['tile'].tolist()
    center_images = [img for img in seg_stats_list if 'center' in img]
    results = [["center_name", "top_name", "left_name", "down_name", "right_name", "selected_name"]]

    tile_cords = []
    for center_img_name in center_images:
        ## get center-tile x1 and y1 coordinates (top left corner)
        x_c = int(center_img_name.split(".")[0].split("_")[-3])
        y_c = int(center_img_name.split(".")[0].split("_")[-2])

        ## generate wsi-thumbnail segmentation mask
        tile_img_seg = np.array(Image.open(os.path.join(Tile_folder_seg, center_img_name)))
        tile_img_resized = cv2.resize(tile_img_seg, (int(tilesize / sf_w), int(tilesize / sf_h)),
                                      interpolation=cv2.INTER_NEAREST)
        x_min_seg = int(x_c / sf_w)
        ymin_seg = int(y_c / sf_h)
        x_max_seg = int(x_min_seg + tile_img_resized.shape[0])
        y_max_seg = int(ymin_seg + tile_img_resized.shape[1])
        wsi_seg[ymin_seg:y_max_seg, x_min_seg:x_max_seg, :] = tile_img_resized

        ## do tile selection
        top_img_name = center_img_name.replace("center", "top").replace(str(x_c)+"_"+str(y_c), str(x_c)+"_"+str(y_c-ovr_th))
        left_img_name = center_img_name.replace("center", "left").replace(str(x_c)+"_"+str(y_c), str(x_c-ovr_th)+"_"+str(y_c))
        down_img_name = center_img_name.replace("center", "down").replace(str(x_c)+"_"+str(y_c), str(x_c)+"_"+str(y_c+ovr_th))
        right_img_name = center_img_name.replace("center", "right").replace(str(x_c)+"_"+str(y_c), str(x_c+ovr_th)+"_"+str(y_c))

        center_score, center_pen, center_artf = tile_scores(seg_stats_sort, penmarker_stats_sort, bc, fo, bl, center_img_name)
        top_score, top_pen, top_artf = tile_scores(seg_stats_sort, penmarker_stats_sort,  bc, fo, bl, top_img_name)
        left_score, left_pen, left_artf = tile_scores(seg_stats_sort, penmarker_stats_sort, bc, fo, bl, left_img_name)
        down_score, down_pen, down_artf = tile_scores(seg_stats_sort, penmarker_stats_sort, bc, fo, bl, down_img_name)
        right_score, right_pen, right_artf = tile_scores(seg_stats_sort, penmarker_stats_sort, bc, fo, bl, right_img_name)

        matrix = np.array([[center_img_name, center_score, center_pen, center_artf], [top_img_name, top_score, top_pen, top_artf],
                 [left_img_name, left_score, left_pen, left_artf], [down_img_name, down_score, down_pen, down_artf],
                 [right_img_name, right_score, right_pen, right_artf]])
        # Filter the numbers based on the "keep" values
        filtered_matrix = matrix[matrix[:, 2] == "keep", :]

        ## generate pen marker-free wsi-thumbnail
        tile_img_clean = np.array(Image.open(os.path.join(Tile_folder_clean, center_img_name)))
        tile_img_clean_resized = cv2.resize(tile_img_clean, (int(tilesize / sf_w), int(tilesize / sf_h)),
                                                interpolation=cv2.INTER_NEAREST)
        if center_artf == "qualified":
            wsi_clean_tiles[ymin_seg:y_max_seg, x_min_seg:x_max_seg, :] = tile_img_clean_resized

        # Select the minimum value from the filtered numbers
        if filtered_matrix.size > 0:
            min_value = np.min(filtered_matrix[:, 1].astype(float))
            min_index = np.where(filtered_matrix[:, 1].astype(float) == min_value)[0][0]

            if center_pen == "keep" and center_artf == "qualified":
                tile_sel = center_img_name
                xmin_sel = int(tile_sel.split(".")[0].split("_")[-3])
                ymin_sel = int(tile_sel.split(".")[0].split("_")[-2])
                xmax_sel = xmin_sel + tilesize
                ymax_sel = ymin_sel + tilesize
                xmin_sel_sc = int(xmin_sel / sf_w)
                ymin_sel_sc = int(ymin_sel / sf_h)
                xmax_sel_sc = int(xmax_sel / sf_w)
                ymax_sel_sc = int(ymax_sel / sf_h)
                results.append([center_img_name, top_img_name, left_img_name, down_img_name, right_img_name, center_img_name])
                wsi_clean_tiles = cv2.rectangle(wsi_clean_tiles, (xmin_sel_sc, ymin_sel_sc), (xmax_sel_sc, ymax_sel_sc),
                              (0, 255, 0), 3)  # Blue box with thickness 2
                wsi_seg_qt[ymin_sel_sc:ymax_sel_sc, xmin_sel_sc:xmax_sel_sc,:]=255

                tile_cords.append([xmin_sel, ymin_sel])

                # copy qualified tile
                source_path = os.path.join(Tile_folder_clean, tile_sel)
                destination_path = os.path.join(Selected, tile_sel)
                shutil.copy(source_path, destination_path)

            else:
                tile_sel = filtered_matrix[min_index, 0]
                xmin_sel = int(tile_sel.split(".")[0].split("_")[-3])
                ymin_sel = int(tile_sel.split(".")[0].split("_")[-2])
                xmax_sel = xmin_sel + tilesize
                ymax_sel = ymin_sel + tilesize

                xmin_sel_sc = int(xmin_sel / sf_w)
                ymin_sel_sc = int(ymin_sel / sf_h)
                xmax_sel_sc = int(xmax_sel / sf_w)
                ymax_sel_sc = int(ymax_sel / sf_h)

                if filtered_matrix[min_index, 3] =="qualified":
                    # ## generate pen marker-free wsi-thumbnail
                    # tile_img_clean = np.array(Image.open(os.path.join(Tile_folder_clean, tile_sel)))
                    # tile_img_clean_resized = cv2.resize(tile_img_clean, (int(tilesize / sf_w), int(tilesize / sf_h)),
                    #                                     interpolation=cv2.INTER_NEAREST)
                    # wsi_clean[ymin_sel_sc:ymax_sel_sc, xmin_sel_sc:xmax_sel_sc, :] = tile_img_clean_resized
                    # wsi_clean_tiles[ymin_sel_sc:ymax_sel_sc, xmin_sel_sc:xmax_sel_sc, :] = tile_img_clean_resized


                    results.append([center_img_name, top_img_name, left_img_name, down_img_name, right_img_name, filtered_matrix[min_index, 0]])
                    wsi_clean_tiles = cv2.rectangle(wsi_clean_tiles, (xmin_sel_sc, ymin_sel_sc), (xmax_sel_sc, ymax_sel_sc),
                                  (0, 255, 0), 3)  # Green box with thickness 2
                    wsi_seg_qt[ymin_sel_sc:ymax_sel_sc, xmin_sel_sc:xmax_sel_sc, :] = 255
                    tile_cords.append([xmin_sel, ymin_sel])

                    # copy qualified tile
                    source_path = os.path.join(Tile_folder_clean, tile_sel)
                    destination_path = os.path.join(Selected, tile_sel)
                    shutil.copy(source_path, destination_path)


                else:
                    results.append([center_img_name, top_img_name, left_img_name, down_img_name, right_img_name, "Non"])

    data_list = [wsi_seg,wsi_clean_tiles, wsi_seg_qt]
    with Pool() as pool:
        results = pool.map(fill_holes_wsi_seg, data_list)
    # Unpack results to individual variables (order will be preserved)
    wsi_seg_fill, wsi_clean_tiles_fill, wsi_seg_qt_fill= results

    return wsi_seg_fill, wsi_clean_tiles_fill, wsi_seg_qt_fill,  results, tile_cords

def post_proces(prediction, obj_size, args, back_thr, blur_fold_thr):
    # resize segmentation to original size
    prediction = cv2.resize(prediction, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)

    class_colors = {
        0: (0, 0, 0),  # Class 0: Black for background
        1: (0, 255, 0),  # Class 1: green for tissue
        2: (255, 65, 90),  # Class 2: yellow for fold
        3: (255, 165, 0),  # Class 3: orange for blur
    }

    ### fill small holes in the background with tissue
    structure_8connectivity = np.ones((3, 3), dtype=bool)
    binary_mask = prediction == 1
    binary_mask = binary_dilation(binary_mask, structure=structure_8connectivity)
    # Fill holes in the binary mask
    filled_binary_mask = binary_fill_holes(binary_mask)
    # Create a new mask to store the result
    filled_mask = prediction.copy()
    # Set all regions that were holes to the nearest non-zero label
    filled_mask[filled_binary_mask & (prediction == 0)] = 1  # fill with tissue
    prediction = filled_mask

    ### remove small regions that have blur of fold
    class_mask = prediction > 1
    # Label connected components in the class mask
    labeled_array, num_features = label(class_mask)
    # Find slices of labeled objects
    object_slices = find_objects(labeled_array)
    # Create a copy of the original mask to modify
    modified_mask = prediction.copy()
    # Iterate over each detected object
    for i, slice_tuple in enumerate(object_slices):
        # Calculate the size of the object
        object_size = np.sum(labeled_array[slice_tuple] == (i + 1))
        # Replace object with replacement class if its size is less than min_size
        if object_size < obj_size:
            modified_mask[labeled_array == (i + 1)] = 1

    prediction = modified_mask

    ### Assign colors to each pixel based on the class color map
    height, width = prediction.shape
    output_image = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            class_label = prediction[y, x]
            color = class_colors[class_label]
            output_image.putpixel((x, y), color)

    output_image = np.array(output_image)

    ### compute artifact statistics
    total_pixels = prediction.size
    num_classes = args.num_classes
    tile_stats = []
    for class_value in range(0, num_classes):
        class_pixel_count = np.sum(prediction == class_value)
        percentage = (class_pixel_count / total_pixels) * 100
        tile_stats.append(round(percentage, 2))

    ### tile classification
    if tile_stats[0] >= back_thr:  # check for white background
        classification = "unqualified"
    elif tile_stats[2] >= blur_fold_thr or tile_stats[3] >= blur_fold_thr:  # check for fold or blur
        classification = "unqualified"
    else:  # artifact free
        classification = "qualified"

    tile_stats.append(classification)
    tile_stats = np.array(tile_stats)

    return output_image, tile_stats


def data_generator(patch_folder, test_transform, batch_size=32, worker=1):
    print(f"\nLoading patches...........")
    # load tile images
    test_images = custom_data_loader(patch_folder, test_transform)
    test_loader = DataLoader(dataset=test_images, batch_size=batch_size, shuffle=False, num_workers=worker,
                             pin_memory=True)
    total_patches = len(test_images)
    print(f"total number of patches are {total_patches}")
    return test_loader, total_patches


class custom_data_loader(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_dir = img_path
        self.transform = transform
        self.data_path = []
        file_list = os.listdir(self.img_dir)
        for img in file_list:
            self.data_path.append(os.path.join(self.img_dir, img))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image = Image.open(self.data_path[idx]).convert('RGB')
        img_name = os.path.basename(self.data_path[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, img_name


def create_patches(wsi_path, wsi_name, patch_folder, workers, patch_size, xmin_indx, ymin_indx, xmax_indx, ymax_indx,
                   overlap, level):
    # get parameters for parallel processing in row directions (y)
    params = [(y, wsi_path, wsi_name, patch_folder, patch_size, xmin_indx, xmax_indx, overlap, level)
              for y in range(ymin_indx, ymax_indx)]
    # extract_and_save_patch(*params[5])
    with mp.Pool(processes=workers) as p:
        p.starmap(extract_and_save_patch, params)


def count_color(img, target_color):
    # Create a mask for the target color
    target_color_np = np.array(target_color)
    mask = cv2.inRange(img, target_color_np, target_color_np)
    count = cv2.countNonZero(mask)
    return count


def create_patches_seg(n, wsi_seg_path, x_min_wsi_seg, ymin_wsi_seg, wsi_seg_tile_h, wsi_seg_tile_w, args):
    wsi_seg = Image.open(wsi_seg_path)
    wsi_seg_np = np.array(wsi_seg)
    tile_seg = wsi_seg_np[ymin_wsi_seg:ymin_wsi_seg + wsi_seg_tile_h, x_min_wsi_seg:x_min_wsi_seg + wsi_seg_tile_w]

    back_thr = args.back_thr
    blur_fold_thr = args.blur_fold_thr

    # class_colors = {
    #     0: (0, 0, 0),  # Class 0: Black for background
    #     1: (0, 255, 0),  # Class 1: green for tissue
    #     2: (255, 65, 90),  # Class 2: yellow for fold
    #     3: (255, 165, 0),  # Class 3: orange for blur
    # }
    total_pixels = tile_seg.shape[0] * tile_seg.shape[1]
    tile_stats = []

    background = count_color(tile_seg, [0, 0, 0])
    percentage_bk = (background / total_pixels) * 100
    tile_stats.append(round(percentage_bk, 2))

    tissue = count_color(tile_seg, [0, 255, 0])
    percentage_ts = (tissue / total_pixels) * 100
    tile_stats.append(round(percentage_ts, 2))

    fold = count_color(tile_seg, [255, 65, 90])
    percentage_fo = (fold / total_pixels) * 100
    tile_stats.append(round(percentage_fo, 2))

    blur = count_color(tile_seg, [255, 165, 0])
    percentage_bl = (blur / total_pixels) * 100
    tile_stats.append(round(percentage_bl, 2))

    ### tile classification
    if tile_stats[0] >= back_thr:  # check for white background
        classification = "unqualified"
    elif tile_stats[2] >= blur_fold_thr or tile_stats[3] >= blur_fold_thr:  # check for fold or blur
        classification = "unqualified"
    else:  # artifact free
        classification = "qualified"

    tile_stats.append(classification)
    tile_stats = np.array(tile_stats)

    tile_seg_res = cv2.resize(tile_seg, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
    return tile_seg_res, tile_stats


def tile_seg(tile_name, wsi_tile_size, args, cpu_workers, wsi_seg_path, sf_w, sf_h):
    # Open wsi segmetation
    mask_image = Image.open(wsi_seg_path)
    wsi_seg_tile_h = int(wsi_tile_size / sf_h)
    wsi_seg_tile_w = int(wsi_tile_size / sf_w)

    x_min_wsi_seg = []
    ymin_wsi_seg = []
    for n in tile_name:
        x = int(n.split(".")[0].split("_")[-2])
        y = int(n.split(".")[0].split("_")[-1])
        x_scale = int(x / sf_w)
        y_scale = int(y / sf_h)
        x_min_wsi_seg.append(x_scale)
        ymin_wsi_seg.append(y_scale)
        # tile_seg_res, tile_stats = create_patches_seg(n, wsi_seg_path, x_scale,y_scale, wsi_seg_tile_h, wsi_seg_tile_w, args )
    x_min_wsi_seg = np.array(x_min_wsi_seg)
    ymin_wsi_seg = np.array(ymin_wsi_seg)

    ### post-processing using parallel processing
    params = [(tile_name[i], wsi_seg_path, x_min_wsi_seg[i], ymin_wsi_seg[i], wsi_seg_tile_h, wsi_seg_tile_w, args) for
              i in range(len(tile_name))]
    with mp.Pool(processes=cpu_workers) as p:
        result = p.starmap(create_patches_seg, params)

    batch_tile_stat = []
    batch_tile = []
    for idx, (output_image, tile_stats) in enumerate(result):
        batch_tile.append(output_image)
        batch_tile_stat.append(tile_stats)

    return batch_tile, batch_tile_stat


def tile_segmentation(args, exel_path, model, test_loader, cpu_workers, Tile_folder_seg, exel_name, save_fla=True):
    # to store segmentation statistics
    stats = [["tile", "%background", "%tissue", "%fold", "%blur", "classification"]]

    for data, names in test_loader:
        data = data.cuda()
        batch_output_seg, batch_tile_sta = test_single_patch(args, data, model, cpu_workers, network=args.network)
        ## save segmentation and statistics
        for i in range(len(batch_output_seg)):
            img = batch_output_seg[i]
            st = batch_tile_sta[i]
            n = names[i]
            stats.append([n, st[0], st[1], st[2], st[3], st[4]])
            if save_fla:
                pil_img = Image.fromarray(img)
                img_rgb = pil_img.convert('RGB')
                img_rgb.save(os.path.join(Tile_folder_seg, n))

    df = pd.DataFrame(stats[1:], columns=stats[0])
    excel_file_path = os.path.join(exel_path, exel_name)
    df.to_excel(excel_file_path, index=False)


def test_single_patch(args, image, net, num_processes, network="DHUnet", obj_size=500):
    back_thr = args.back_thr
    blur_fold_thr = args.blur_fold_thr
    # send images to segmentation model
    image = image.cuda()
    net.eval()
    with torch.no_grad():
        if network == "DHUnet":
            net_out = net(image, image)[0]
            out = torch.argmax(torch.softmax(net_out, dim=1), dim=1).squeeze(0)
        else:
            net_out = net(image)
            out = torch.argmax(torch.softmax(net_out, dim=1), dim=1).squeeze(0)
        predictions = out.cpu().detach().numpy()

    ## post-processing step with parallel processing
    params = [(y, obj_size, args, back_thr, blur_fold_thr) for y in predictions]

    with mp.Pool(processes=num_processes) as p:
        result = p.starmap(post_proces, params)

    ## collect results
    batch_tile_stat = []
    batch_tile = []
    for idx, (output_image, tile_stats) in enumerate(result):
        batch_tile.append(output_image)
        batch_tile_stat.append(tile_stats)

    return batch_tile, batch_tile_stat


def slide_concate(test_save_path, eval_save_dir, concate_path_txt):
    with open(concate_path_txt, 'r') as f:
        eval_slides = f.readlines()
    for eval_slide in eval_slides:
        eval_slide = eval_slide.strip('\n')[::-1].split("_", 5)[::-1]
        print(eval_slide)
        IMAGES_PATH = os.path.join(test_save_path, eval_slide[0][::-1])  #
        IMAGE_SAVE_PATH = os.path.join(eval_save_dir, eval_slide[0][::-1].split('/')[-1] + '.png')
        IMAGE_SIZE = int(eval_slide[1][::-1]), int(eval_slide[2][::-1])  # 48000 90000
        patch_size = int(eval_slide[3][::-1])  # 1000
        overlap = int(eval_slide[4][::-1])  # 500
        os.makedirs(os.path.dirname(IMAGE_SAVE_PATH), exist_ok=True)
        print(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap)
        image_concate(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap)
        print("saved path ", IMAGE_SAVE_PATH)


# Restore the small patch under the IMAGES_PATH path to the original IMAGE_SIZE image
def image_concate(IMAGES_PATH, IMAGE_SAVE_PATH, IMAGE_SIZE, patch_size, overlap):
    image_names = sorted(os.listdir(IMAGES_PATH))

    # The number of rows and columns of the image
    step_size = patch_size - overlap
    step_x_max = int(np.ceil((IMAGE_SIZE[0] - step_size) / step_size))
    step_y_max = int(np.ceil((IMAGE_SIZE[1] - step_size) / step_size))
    assert step_x_max * step_y_max == len(image_names), "Wrong number of files."

    # Define the image stitching function
    to_image = Image.new('L', IMAGE_SIZE)

    # Loop through and paste each picture to the corresponding position in order
    for x in range(step_x_max):
        for y in range(step_y_max):
            path = IMAGES_PATH + '/' + "%03d_%03d.png" % (x, y)
            from_image = Image.open(path).resize((patch_size, patch_size), Image.NEAREST)

            position = [x * step_size, y * step_size]
            if position[0] + patch_size >= IMAGE_SIZE[0]:
                position[0] = IMAGE_SIZE[0] - patch_size - 1
            if position[1] + patch_size >= IMAGE_SIZE[1]:
                position[1] = IMAGE_SIZE[1] - patch_size - 1
            to_image.paste(from_image, position)
    savePalette(to_image, IMAGE_SAVE_PATH)


def savePalette(image_array, save_path):
    mask = image_array.convert("L")
    palette = []
    for j in range(256):
        palette.extend((j, j, j))
        palette[:3 * 10] = np.array([
            [0, 0, 0],  # label 0
            [0, 255, 0],  # label 1
            [0, 0, 255],  # label 2
            [255, 255, 0],  # label 3
            [255, 0, 0],  # label 4
            [0, 255, 255],  # label 5
        ], dtype='uint8').flatten()
    mask = mask.convert('P')
    mask.putpalette(palette)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    mask.save(save_path)
