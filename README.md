# Abstract
TBD

# Setting Up the Pipeline:
1. System requirements:
- Ubuntu 20.04 or 22.04
- CUDA version: 12.2
- Python version: 3.9 (using conda environments)
- Anaconda version 23.7.4
2. Steps to Set Up the Pipeline:
- Download the pipeline to your Desktop
- Navigate to the downloaded pipeline folder
- Right-click within the pipeline folder and select Open Terminal
- Create a conda environment:
`conda create -n WSISmartTiling python=3.9`

- Activate the environment:
  `conda activate WSISmartTiling`

- Install required packages:
  `pip install -r requirements.txt`

# Datasets:

- Contact the corresponding author to access the datasets described in the paper
- The datasets (only tiles and segmentation masks) are available for research purposes only

# Pretrained Weights:

TBD.

# Inference:

<p align="justify"> The pipeline starts by identifying the WSI tissue region and dividing it into smaller image tiles (e.g., 270x270). Pen-marking detection is then applied to categorize the tiles into two classes: those with high pen-marking (which are discarded) and those with medium and low pen-marking. Tiles with medium and low pen-marking undergo a pen-marking removal process, resulting in clean image tiles. Next, the clean image tiles are fed into the proposed artifact detection model to identify artifacts, followed by an optimization technique to select the best tiles—those with minimal artifacts and background and maximum qualified tissue. Finally, the WSI is reconstructed by combining the selected tiles to generate the final output. Additionally, the model generates a segmentation for the entire WSI and also provides statistics on the tile segmentations. </p>
- Place your Whole Slide Image (WSI) into the test_wsi folder
- The pre-trained weights are provided in `pretrained_ckpt` folder
- In the terminal execute:
  `python test_wsi.py`

- After running the inference, you will obtain the following outputs in `test_wsi` folder:
    - A thumbnail image of WSI
    - A thumbnail image of WSI with regions of interest (ROI) identified
    - A segmentation mask highlighting segmented regions of the WSI [Qualifed tissue: green, fold: red, blur: orange, and background: black]
    - A segmentation mask highlighting only qualified tissue regions of the WSI [background:0, qualified tissue:255]
    - Excel files contain statistics on identified artifacts
    - A folder named Selected_tiles containing qualified tiles
- If your WSI image has a format other than .svs or .mrxs, please modify line 92 in `test_wsi.py`
- It is recommended to use a tile size of 270 × 270 pixels
- To generate tiles of different sizes (e.g., 512x512):
    - Run the pipeline to generate the qualified tissue mask
    - Use the qualified tissue mask and the WSI to generate tiles of the desired size (a Python script will be provided soon to do this)
- If your WSI image contains pen-markings other than red, blue, green, or black, please update the `pens.py` file (located in the `wsi_tile_cleanup/filters folder`) to handle any additional pen-markings
- To generate a high-resolution thumbnail image and segmentation masks, you can adjust the `thumbnail_size` parameter in `inti_artifact.py`. However, note that this will increase the execution time
- Check out the useful parameters on line 58 of `inti_artifact.py` and adjust them if needed

# Training:

- To retrain the artifact detection model, refer to the details provided in: [GitHub](https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA)
- To retrain the ink removal detection model, refer to the details provided in: [GitHub](https://github.com/Vishwesh4/Ink-WSI)

# Results & Benchmarking

<p align="justify"> Benchmark models, GrandQC (https://github.com/cpath-ukk/grandqc) pixel-wise segmentation model developed for artifact detection, and four tile-wise classification models with different network architectures (https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline), namely MoE-CNN, MoE-ViT, multiclass-CNN, and multiclass-ViT. The proposed pixel-wise segmentation model is compared to GrandQC based on pixel segmentation accuracy and to MoE-CNN, MoE-ViT, Multiclass-CNN, and Multiclass-ViT based on tile classification. The classification considers three classes—artifact-free, fold, and blur. The model takes input tiles and generates segmentation masks, which are then used for tile classification. The classification process follows these criteria: (1) If the background occupies more than 50% of the tile, it is classified as a background tile. (2) If the background occupies less than 50%, but blurring and/or folding artifacts exceed 10% of the tile, it is classified as either fold or blur. (3) If the background is less than 50% and blurring and/or folding artifacts are below 10%, the tile is classified as artifact-free. The internal and external datasets are described in the manuscript. For segmentation, the ground truth segmentation masks are compared to the segmentation masks generated by the model. For classification, the predicted classes are compared to the ground truth labels. Quantitative metrics, including total accuracy (Acc), precision, recall, and F1 score, were used to evaluate classification performance, and the Dice metric was used to evaluate segmentation performance. The source code and model weights for benchmark models were obtained from the original GitHub repositories. </p>


