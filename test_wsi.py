import pandas as pd

from utils import test_single_patch, find_Tissue_regions, create_folder, create_patches, \
    data_generator, fill_holes_wsi_seg, tile_segmentation, pen_marker, tiles_selection
from inti_artifact import inti_model_artifact
import torch.backends.cudnn as cudnn
import numpy as np
import random
import torch
from network.DHUnet import DHUnet
from thop import profile
import os
import logging
import shutil
import sys
import time
from PIL import Image
from Ink_Removal.options.test_options import TestOptions
from Ink_Removal.models import create_model


if __name__ == "__main__":

    #####intialize artifact detection model#####
    args_artifact, config_artifact, transform_artifact = inti_model_artifact()
    if not args_artifact.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args_artifact.seed)
    np.random.seed(args_artifact.seed)
    torch.manual_seed(args_artifact.seed)
    torch.cuda.manual_seed(args_artifact.seed)
    torch.cuda.manual_seed_all(args_artifact.seed)  # if use multi-GPU
    dataset_name = args_artifact.dataset
    args_artifact.is_pretrain = True
    net_artifact = DHUnet(config_artifact, num_classes=args_artifact.num_classes)
    snapshot_artifact = args_artifact.pretrained_ckpt
    device_artifact = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_artifact = net_artifact.to(device_artifact)
    msg = net_artifact.load_state_dict(torch.load(snapshot_artifact, map_location=device_artifact))
    print("self trained DHUnet ", msg)
    total = sum([param.nelement() for param in net_artifact.parameters()])
    input = torch.randn(1, 3, 224, 224).cuda()
    flops, params = profile(net_artifact, inputs=(input, input))[:2]
    snapshot_name = snapshot_artifact.split('/')[-1]
    log_folder = args_artifact.output_dir + '/test_log_/' + dataset_name
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + snapshot_name + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args_artifact))
    logging.info(snapshot_name)

    if args_artifact.is_savenii:
        args_artifact.test_save_dir = os.path.join(args_artifact.output_dir, "predictions")
        test_save_path = args_artifact.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    #####intialize pen marker removal model#####
    args_penmarker = TestOptions().parse()  # get test options
    args_penmarker.num_threads = 0  # test code only supports num_threads = 0
    args_penmarker.batch_size = args_artifact.batch_size  # test code only supports batch_size = 1
    args_penmarker.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    args_penmarker.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model_penmarker = create_model(args_penmarker)  # create a model given opt.model and other options
    model_penmarker.setup(args_penmarker)  # regular setup: load and print networks; create schedulers
    if args_penmarker.eval:
        model_penmarker.eval()

    device_penmarker = torch.device(f"cuda:{args_penmarker.gpu_ids[0]}")
    max_marker_thr = args_artifact.max_marker_thr
    min_marker_thr = args_artifact.min_marker_thr
    clean_penmarker = args_artifact.clean_penmarker

    #####get WSI parameters #####
    thumbnail_size = args_artifact.thumbnail_size
    tilesize = args_artifact.img_size
    if tilesize != 270:
        assert "tile size larger than 270 is not recommended "
    overlap = args_artifact.overlap
    level = args_artifact.wsilevel
    data = args_artifact.wsi_folder
    cpu_workers = args_artifact.cpu_workers
    batch_size = args_artifact.batch_size
    tile_seg_sv = args_artifact.save_seg
    #####get list of WSI to process#####
    wsi_files = [f for f in os.listdir(data) if f.endswith(".svs") or f.endswith(".mrxs")]
    start_time = time.time()  # Start time

    #####iterate over WSIs#####
    for wsi_id, wsi_file in enumerate(wsi_files):

        print(wsi_file)
        stats = []
        stats.append(["tile", "%background", "%tissue", "%fold", "%blur", "classification"])
        ##### Generate processing and output folders#####
        Selected = os.path.join(data, wsi_file.split(".")[0] + "_results", "Selected_tiles")
        Tile_folder = os.path.join(data, wsi_file.split(".")[0] + "_results", "All_tiles")
        Tile_folder_seg = os.path.join(data, wsi_file.split(".")[0] + "_results", "All_tiles_seg")
        Tile_folder_clean = os.path.join(data, wsi_file.split(".")[0] + "_results", "All_tiles_clean")
        create_folder(Selected)
        create_folder(Tile_folder)
        create_folder(Tile_folder_seg)
        create_folder(Tile_folder_clean)

        ##### Step 1: detect tissue regions
        start_time = time.time()
        wsi_path = os.path.join(data, wsi_file)
        thumbnail, thumbnail_mask, thumbnail_roi, xmin_indx, ymin_indx, xmax_indx, ymax_indx, sf_w, sf_h = find_Tissue_regions(
            wsi_path, thumbnail_size, tilesize, level)

        ##### Step 2: generate rgb tiles from tissue regions using parrallel procssing
        create_patches(wsi_path, wsi_file, Tile_folder, cpu_workers, tilesize, xmin_indx, ymin_indx, xmax_indx,
                       ymax_indx, overlap, level)
        print("Tiles generation is done!")
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # Convert to minutes
        print(f"Execution time for WSI tiling: {execution_time:.6f} minutes")

        ##### Step 3: generate tile segmentation
        start_time = time.time()
        exel_file_path = os.path.join(data, wsi_file.split(".")[0] + "_results")
        data_loader, total_patches = data_generator(Tile_folder, test_transform=transform_artifact,
                                                   batch_size=batch_size, worker=cpu_workers)
        tile_segmentation(args_artifact, exel_file_path, net_artifact, data_loader, cpu_workers, Tile_folder_seg, "segmentation_stats.xlsx")
        print("Tiles segmentation is done!")
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # Convert to minutes
        print(f"Execution time for tile segmentation: {execution_time:.6f} minutes")

        ##### Step 4: Remove pen markers on tiles
        start_time = time.time()
        pen_marker(args_artifact, args_penmarker, exel_file_path,  model_penmarker, Tile_folder, Tile_folder_clean, max_marker_thr, min_marker_thr,
               clean_penmarker, "penmarker_stats.xlsx")
        print("Pen marker removal is done!")
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # Convert to minutes
        print(f"Execution time for pen-marker detection & removal: {execution_time:.6f} minutes")

        ##### Step 5: tiles selection
        start_time = time.time()
        wsi_seg_fill, wsi_clean_tiles_fill,wsi_seg_qt_fill,  sel_tiles, tile_cords = tiles_selection(thumbnail, sf_w, sf_h,overlap, "segmentation_stats.xlsx", "penmarker_stats.xlsx", Tile_folder_seg, Tile_folder_clean,Selected, exel_file_path,  bc=1, fo=1, bl=1, tilesize=270)
        print("Tile selection is done!")
        end_time = time.time()
        execution_time = (end_time - start_time) / 60  # Convert to minutes
        print(f"Execution time for tile selection: {execution_time:.6f} minutes")

        ##### Step 6: saving results
        #save wsi segmentation mask
        thumbnail_mask_3d = np.repeat(thumbnail_mask[:, :, np.newaxis], 3, axis=2)
        masked_rgb_image = wsi_seg_fill * thumbnail_mask_3d
        masked_rgb_image = masked_rgb_image.astype(np.uint8)
        masked_rgb_image_arr = Image.fromarray(masked_rgb_image)
        masked_rgb_image_arr.save(
            os.path.join(data, wsi_file.split(".")[0] + "_results" + "/" + wsi_file.split(".")[0] + "_thumbnail_seg.png"))

        #save WSI cleaned with selected tiles
        masked_rgb_image_arr = Image.fromarray(wsi_clean_tiles_fill)
        masked_rgb_image_arr.save(
            os.path.join(data, wsi_file.split(".")[0] + "_results" + "/" + wsi_file.split(".")[0] + "_thumbnail_clean_sel.png"))
        # save save qualified tissue mask
        masked_rgb_image_arr = Image.fromarray(wsi_seg_qt_fill)
        masked_rgb_image_arr.save(
            os.path.join(data, wsi_file.split(".")[0] + "_results" + "/" + wsi_file.split(".")[
                0] + "_qualified_mask.png"))

        #save WSI-RGB thumbnail
        thumbnail_arr = Image.fromarray(thumbnail)
        thumbnail_arr.save(
            os.path.join(data, wsi_file.split(".")[0] + "_results" + "/" + wsi_file.split(".")[0] + "_thumbnail.png"))

        #save wsi-RGB thumbnail with ROI
        thumbnail_roi_arr = Image.fromarray(thumbnail_roi)
        thumbnail_roi_arr.save(
            os.path.join(data,
                         wsi_file.split(".")[0] + "_results" + "/" + wsi_file.split(".")[0] + "_thumbnail_roi.png"))

        # save tile coordinates
        # df = pd.DataFrame(tile_cords, columns=['X_Coordinate', 'Y_Coordinate'])
        # df.to_excel(os.path.join(data, wsi_file.split(".")[0] + "_results" + "/" + wsi_file.split(".")[0] + "_coordinates.xlsx"), index=False)  # Set index=False to not save row indices in the file

        ##### remove created folders
        shutil.rmtree(Tile_folder)
        shutil.rmtree(Tile_folder_seg)
        shutil.rmtree(Tile_folder_clean)
        print("Results are saved")

    end_time = time.time()  # End time
    execution_time = end_time - start_time  # Calculate the execution time
    execution_time_minutes = execution_time / 60  # Convert seconds to minutes
    print(f"Total execution time is {execution_time_minutes} minutes")
