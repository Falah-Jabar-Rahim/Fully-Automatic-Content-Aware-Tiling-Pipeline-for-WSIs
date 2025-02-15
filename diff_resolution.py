import argparse
import os
import shutil
import random
import time
import cv2
import numpy as np
import torch
from utils import  find_Tissue_regions, create_folder, create_patches, \
    data_generator, fill_holes_wsi_seg, tile_seg
from config import get_config
import pandas as pd
from PIL import Image
from torchvision import transforms
Image.MAX_IMAGE_PIXELS = None
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'  # update this path

parser = argparse.ArgumentParser()
### network parameters
parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
parser.add_argument('--seed', type=int, default=301, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/DHUnet_224.yaml", metavar="FILE", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


### wsi parameters
parser.add_argument('--img_size', type=int, default=270, help='required tile size (Options:270, 540, or 1080)')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size per gpu')
parser.add_argument('--wsilevel', default=0, type=int, help='level from open slide to read')
parser.add_argument('--thumbnail_size', default=5000, type=int, help='required wsi thumbnail size')
parser.add_argument('--wsi_folder', default="input_WSI/TCGA_qaultative/test", type=str, help='folder contains wsi images')
parser.add_argument('--cpu_workers', default=40, type=int, help='number of cpu workers')
parser.add_argument('--save_seg', default=1, type=int, help='to save tile segmentation result')
parser.add_argument('--back_thr', default=50, type=int, help='% of background')
parser.add_argument('--blur_fold_thr', default=20, type=int, help='% of blur and fold')

args = parser.parse_args()
config = get_config(args)
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])


def inference(args, tilesize, test_loader, cpu_workers, wsi_seg_path, sf_w, sf_h):
    all_tiles = []
    all_stats = []
    all_names = []
    for data, names in test_loader:
        batch_output_seg, batch_tile_sta = tile_seg(names, tilesize, args, cpu_workers, wsi_seg_path, sf_w, sf_h)
        all_tiles.append(list(batch_output_seg))
        all_stats.append(batch_tile_sta)
        all_names.append(names)

    return all_tiles, all_stats, all_names


if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    openslidelevel = args.wsilevel
    thumbnail_size = args.thumbnail_size
    tilesize = args.img_size
    data = args.wsi_folder
    cpu_workers = args.cpu_workers
    wsi_files = [f for f in os.listdir(data) if f.endswith(".svs") or f.endswith(".mrxs")]
    batch_size = args.batch_size
    tile_seg_sv = args.save_seg

    for wsi_id, wsi_file in enumerate(wsi_files):

        stats = []
        stats.append(["tile", "%background", "%tissue", "%fold", "%blur", "classification"])
        ## generate output folders
        Qualified = os.path.join(data, wsi_file.split(".")[0]+"_results", "Qualified")
        Unqualified = os.path.join(data, wsi_file.split(".")[0]+"_results", "Unqualified")
        Tile_folder = os.path.join(data, wsi_file.split(".")[0]+"_results", "All_tiles")
        create_folder(Qualified)
        create_folder(Unqualified)
        create_folder(Tile_folder)

        ## generate tiles
        wsi_path = os.path.join(data, wsi_file)
        wsi_seg_path = wsi_path.split(".")[0]+"_seg.png"
        thumbnail, thumbnail_mask, thumbnail_roi, xmin_indx, ymin_indx, xmax_indx, ymax_indx, sf_w, sf_h = find_Tissue_regions(
            wsi_path, thumbnail_size, tilesize)
        create_patches(wsi_path, wsi_file, Tile_folder, cpu_workers, tilesize, xmin_indx, ymin_indx, xmax_indx,
                       ymax_indx)
        print("Tiles generation is done!")

        ## compute segmentation
        data_loader, total_patches = data_generator(Tile_folder, test_transform=test_transform,
                                                    batch_size=batch_size, worker=cpu_workers)
        output_seg, tile_stats, tile_names = inference(args, tilesize, data_loader, cpu_workers, wsi_seg_path, sf_w, sf_h)
        print("Tiles segmentation is done!")

        ## generate wsi segmentation mask
        thumbnail_h, thumbnail_w, _ = thumbnail.shape
        wsi_seg = np.zeros((thumbnail_h, thumbnail_w, 3), dtype=np.uint8)
        for btch_id, _ in enumerate(output_seg):
            batch_tile_name = tile_names[btch_id]
            batch_tile_img = output_seg[btch_id]
            tile_st = tile_stats[btch_id]

            for idx in range(0, len(batch_tile_name)):
                tile_img = batch_tile_img[idx]
                tile_name = batch_tile_name[idx]
                st = tile_st[idx]

                x_min_wsi = int(tile_name.split(".")[0].split("_")[-2])
                ymin_wsi = int(tile_name.split(".")[0].split("_")[-1])

                tile_img_resized = cv2.resize(tile_img, (int(tilesize / sf_w), int(tilesize / sf_h)),
                                              interpolation=cv2.INTER_NEAREST)
                if st[4] == "qualified":
                    source_path = os.path.join(Tile_folder, tile_name)
                    destination_path = os.path.join(Qualified, tile_name)
                    shutil.move(source_path, destination_path)
                    if tile_seg_sv:
                        tile_img_arr = Image.fromarray(tile_img)
                        tile_img_arr.save(os.path.join(Qualified, tile_name).split(".")[0] + "_seg.png")
                else:
                    source_path = os.path.join(Tile_folder, tile_name)
                    destination_path = os.path.join(Unqualified, tile_name)
                    shutil.move(source_path, destination_path)
                    if tile_seg_sv:
                        tile_img_arr = Image.fromarray(tile_img)
                        tile_img_arr.save(os.path.join(Unqualified, tile_name).split(".")[0] + "_seg.png")

                x_min_seg = int(x_min_wsi / sf_w)
                ymin_seg = int(ymin_wsi / sf_h)
                x_max_seg = int(x_min_seg + tile_img_resized.shape[0])
                y_max_seg = int(ymin_seg + tile_img_resized.shape[1])
                wsi_seg[ymin_seg:y_max_seg, x_min_seg:x_max_seg, :] = tile_img_resized
                stats.append([tile_name, st[0], st[1], st[2], st[3], st[4]])
        print("wsi segmentation mask is done!")

        ## save results
        wsi_seg_fill = fill_holes_wsi_seg(wsi_seg)
        thumbnail_mask_3d = np.repeat(thumbnail_mask[:, :, np.newaxis], 3, axis=2)
        masked_rgb_image = wsi_seg_fill * thumbnail_mask_3d
        masked_rgb_image = masked_rgb_image.astype(np.uint8)

        masked_rgb_image_arr = Image.fromarray(masked_rgb_image)
        masked_rgb_image_arr.save(
            os.path.join(data, wsi_file.split(".")[0] +"_results"+ "/" + wsi_file.split(".")[0] + "_seg_re-generated.png"))

        thumbnail_arr = Image.fromarray(thumbnail)
        thumbnail_arr.save(os.path.join(data, wsi_file.split(".")[0] +"_results"+ "/" + wsi_file.split(".")[0] + "_thumbnail.png"))

        thumbnail_roi_arr = Image.fromarray(thumbnail_roi)
        thumbnail_roi_arr.save(
            os.path.join(data, wsi_file.split(".")[0]+"_results" + "/" + wsi_file.split(".")[0] + "_thumbnail_roi.png"))

        df = pd.DataFrame(stats[1:], columns=stats[0])
        excel_file_path = os.path.join(data, wsi_file.split(".")[0] +"_results"+ "/" + wsi_file.split(".")[0] + "_tile_stats.xlsx")
        df.to_excel(excel_file_path, index=False)
        shutil.rmtree(Tile_folder)
        print("results are saved")



print("Completed !")
