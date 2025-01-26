# Fully Automatic Content-Aware Tiling Pipeline for Whole Slide Images 
![WSI-QA](./WSI-QA.bmp)
# Abstract: 
TBD.
# Setting Up the Pipeline:
1. System requirements:
- Ubuntu 20.04 or 22.04
- CUDA version: 12.2
- Python version: 3.9 (using conda environments)
- Anaconda version 23.7.4

2. Steps to Set Up the Pipeline:
- Download the pipeline to your Desktop
- Navigate to the downloaded pipeline folder
- Right-click within the pipeline folder and select `Open Terminal`
- Create a conda environment:
```bash
  conda create -n WSISmartTiling python=3.9
```
- Activate the environment:
```bash
  conda activate WSISmartTiling
```
- Install required packages:
```bash
  pip install -r requirements.txt
```


# Datasets:

- Contact the corresponding author to access the datasets described in the paper.
- The datasets will be available soon for research purposes only.

# Pretrained Weights:

- Download the pretrained weights from this link: https://drive.google.com/drive/folders/1mfCa6YiyFgbrjwOxLUKOKeA3JlIjDQ77?usp=drive_link
- Place the downloaded weights into the `pretrained_ckpt` folder.

# Inference:

- Place your Whole Slide Image (WSI) into the `test_wsi` folder
- The pre-trained weights are provided in `pretrained_ckpt` folder
- In the terminal execute:
```bash
  python test_wsi.py
```
- After running the inference, you will obtain the following outputs in `test_wsi` folder:
  - A thumbnail image of WSI
  - A thumbnail image of WSI with regions of interest (ROI) identified
  - A segmentation mask highlighting segmented regions of the WSI
  - A segmentation mask highlighting only qualified tissue regions of the WSI [background:0, qualified tissue:255]
  - Excel files contains statistics on identified artifacts
  - A folder named `Selected_tiles` containing qualified tiles
- You can adjust the testing parameters on line 58 of `inti_artifact.py`
- If your WSI image has a format other than .svs or .mrxs, please modify line 92 in `test_wsi.py`
- It is recommended to use a tile size of 270 × 270 pixels
- To generate a high-resolution thumbnail image and segmentation mask, you can adjust the `thumbnail_size` parameter in `test_wsi.py`. However, note that this will increase the execution time.
- To generate tiles of different sizes (e.g., 512x512):
    - Run the pipeline to generate the qualified tissue mask.
    - Use the qualified tissue mask and the WSI to generate tiles of the desired size (a python script will be provided soon to do this)


# Training:

- To retrain the artifact detection model, refer to the details provided in: https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA
- To retrain the ink removal detection model, refer to the details provided in: https://github.com/Vishwesh4/Ink-WSI

# Results 

![WSI-QA](./Performance-metrics.png)

![WSI-QA](./WSI-Segmentation.png)

# Notes:

- If your WSIs do not contain pen-marking artifacts, you can also use this pipeline: https://github.com/Falah-Jabar-Rahim/A-Fully-Automatic-DL-Pipeline-for-WSI-QA
- WSI-SmartTiling is designed to clean and prepare WSIs for deep learning model development, prioritizing performance over efficiency


# Acknowledgment:

Some parts of this pipeline were adapted from work on [GitHub](https://github.com/pengsl-lab/DHUnet). If you use this pipeline, please make sure to cite their work.


# Contact: 
If you have any questions or comments, please feel free to contact: falah.rahim@unn.no
