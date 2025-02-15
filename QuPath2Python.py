import math
import os

import openslide
import pandas as pd
import staintools
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt
import geojson
from shapely.geometry import shape
from shapely.strtree import STRtree
from shapely.geometry import Polygon
from stain_normalization import stain_normalization
from openslide import open_slide
import ast


import numpy as np
import cv2

from stain_normalization.stain_normalization import stain_norm, reinhard_cn, initi_color_norm
from util import test_mat_files, poly_bbox, plt_figs, save_results, visualize_instances_dict, qupath_python
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'


if __name__ == '__main__':

    # file_name = 'BL-14-D43217_0.mat'
    # test_mat_files(file_name)

    openslidelevel = 0  # level from open slide to read
    tilesize = 500 # size in pixels
    min_nucleus = 10 #the minimum number of nucleus needed to be present within a tile
    paddingsize = 0
    overlap = 0.01  # overlap between adjacent tiles
    data = "data/"  # folder name contains the input wsi

    data = os.path.expanduser(data)

    wsi_ext = ".svs"  # extension for wsi
    wsi_ext_anno = ".json"  # extension for annotation file
    plt_fla = 0  # plot some figures or not
    qp_crop = 0  # if qupath used crop option

    # stain normalization
    color_norm = 1  # do stain color normalization or not
    method = "Macenko"  # stain normalization method: Macenko,vahadane, or Reinhard
    reference_img, normalizer = initi_color_norm(method)

    # for saving results
    tile_ext = ".png"
    anno_ext = ".mat"
    # define class names and lables
    dic = {"Other": 1, "Immune cells": 2, "Tumor": 3}

    # get a list of all wsi image
    wsi_files = [f for f in os.listdir(data) if f.endswith(wsi_ext)]

    stats = np.zeros((len(wsi_files) + 1, 5), dtype=object)
    stats[0, :] = ["WSI", "#tiles", "#Other", "#Immune cells", "#Tumor"]


    for wsi_id, wsi_file in enumerate(wsi_files):

        wsi_name = os.path.join(data, wsi_file)
        wsi_name_anno = wsi_name.split(".")[0] + wsi_ext_anno
        folder_name = wsi_name.split(".")[0]
        # Check if the folder already exists or create it if it doesn't
        tem = wsi_name.split(".")[0]
        folder_name = "data/training/"+tem.split("/")[-1]
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            print(f"Folder '{folder_name}' already exists.")

        ## load jason file
        with open(wsi_name_anno) as f:
            allobjects = geojson.load(f)
        # read wsi
        # wsi = openslide.OpenSlide(wsi_name)
        wsi = open_slide(wsi_name)
        width, height = wsi.dimensions
        scalefactor = int(wsi.level_downsamples[openslidelevel])

        ## load nucleus information
        allshapes = [shape(obj["nucleusGeometry"] if "nucleusGeometry" in obj.keys() else obj["geometry"]) for obj in
                     allobjects]
        allshapes_property = [obj["properties"]["classification"] for obj in allobjects]
        allcenters = [s.centroid for s in allshapes]
        allcords = [s.exterior.coords.xy for s in allshapes]

        ## convert qupath coordinates to python
        if qp_crop:
            # read qupath crop coordinates
            wsi_name_anno = wsi_name.split(".")[0] + ".txt"
            # Open the file in read mode
            with open(wsi_name_anno, 'r') as file:
                # Read the content of the file
                qp_cords = file.read()  # [x1, y1, x2, y2]
                qp_cords = ast.literal_eval(qp_cords)  # convert to list
            allcenters, allcords = qupath_python(allcenters, allcords, qp_cords)

        ## building tree search
        searchtree = STRtree(allcenters)

        # find ROI in wsi
        x_min, y_min, x_max, y_max = poly_bbox(allcords)

        tile_id = 0
        tile_cnt = 0
        Other_count = 0
        Immune_cells_count = 0
        Tumor_count = 0
        # do tilling
        for y in range(y_min, y_max, round(tilesize * scalefactor * (1 - overlap))):
            for x in range(x_min, x_max, round(tilesize * scalefactor * (1 - overlap))):


                # check boundary regions
                if x + tilesize <= x_max and y + tilesize <= y_max:



                    # read the tile
                    tile = np.asarray(wsi.read_region((x - paddingsize, y - paddingsize), openslidelevel,
                                                      (tilesize + 2 * paddingsize, tilesize + 2 * paddingsize)))[:, :,
                           0:3]
                    # check if current tile has detected cell
                    tilepoly = Polygon([[x, y], [x + tilesize * scalefactor, y],
                                        [x + tilesize * scalefactor, y + tilesize * scalefactor],
                                        [x, y + tilesize * scalefactor]])
                    hits = searchtree.query(tilepoly)
                    if len(hits) >= min_nucleus:

                        hits_x_coordinates = np.zeros(len(hits), dtype=float)
                        hits_y_coordinates = np.zeros(len(hits), dtype=float)
                        transformed_x = np.zeros(len(hits), dtype=float)
                        transformed_y = np.zeros(len(hits), dtype=float)
                        inst_type = np.zeros((len(hits), 1), dtype=float)
                        inst_centroid = np.zeros((len(hits), 2), dtype=float)
                        insta_color = []
                        insta_cords = []
                        height, width, ch = tile.shape
                        inst_map = np.zeros((height, width), dtype=float)
                        type_map = np.zeros((height, width), dtype=float)

                        # plot nuclus boundaries
                        if plt_fla:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.imshow(tile)
                            ax.axis("off")

                        for i, item in enumerate(hits):
                            x_cord = np.array(allcords[item][0])
                            y_cord = np.array(allcords[item][1])
                            cell_type = np.array(allshapes_property[item]["name"])

                            if cell_type == "Other":
                                Other_count += 1
                            elif cell_type == "Immune cells":
                                Immune_cells_count += 1
                            elif cell_type == "Tumor":
                                Tumor_count += 1
                            else:
                                print("undefine cel type")

                            cell_color = np.array(allshapes_property[item]["color"])
                            # Apply coordinate ransformation
                            transformed_x = (x_cord - x + paddingsize) // scalefactor
                            transformed_y = (y_cord - y + paddingsize) // scalefactor
                            x_y = list(zip(transformed_x, transformed_y))
                            x_y = np.array(x_y, np.int32)
                            insta_cords.append(x_y)
                            insta_color.append(cell_color)

                            ## Assigning classes to different nucleus
                            cv2.fillPoly(type_map, [x_y], dic[str(cell_type)])
                            ## Assigning 255  nucleus and 0 for background
                            cv2.fillPoly(inst_map, [x_y], i + 1)
                            ## get inst_type and inst_centroid
                            inst_type[i] = dic[str(cell_type)]
                            # Apply coordinate ransformation
                            x_c = (allcenters[item].x - x + paddingsize) // scalefactor
                            y_c = (allcenters[item].y - y + paddingsize) // scalefactor
                            inst_centroid[i] = [x_c, y_c]

                            if plt_fla:
                                ax.plot(transformed_x, transformed_y, color=cell_color / 255, marker='o',
                                        markersize=0.5)  # 'ro' means red circles

                        if plt_fla:
                            plt.show()
                            ## plot the tile
                            plt_figs(tile)
                            ## plot the type_map
                            plt_figs(inst_map)
                            ## plot the type_map
                            plt_figs(type_map)

                        print(wsi_name)

                        if color_norm:
                            if method == "Macenko" or method == "vahadane":
                                tile = stain_norm(tile, normalizer)
                            else:
                                tile = reinhard_cn(tile, reference_img)

                        overlay_tile = visualize_instances_dict(tile, insta_cords, inst_type, insta_color,
                                                                inst_centroid)
                        save_results(tile, overlay_tile, inst_map, type_map, inst_type, inst_centroid, folder_name,
                                     tile_ext,
                                     anno_ext,
                                     tile_id)

                        tile_id = tile_id + 1
                        tile_cnt = tile_cnt + 1
        stats[wsi_id + 1, 0] = wsi_file
        stats[wsi_id + 1, 1] = tile_cnt
        stats[wsi_id + 1, 2] = Other_count
        stats[wsi_id + 1, 3] = Immune_cells_count
        stats[wsi_id + 1, 4] = Tumor_count


    df = pd.DataFrame(stats)
    df.to_excel("data/stats.xlsx", index=False)

    print("Done!")

