# A Fully Automatic Deep Learning Pipeline for WSI Quality Assessment
![WSI-QA](./WSI-QA.bmp)
# Abstract: 
In recent years, the use of deep learning (DL) methods, including convolutional neural networks (CNNs) and vision transformers (ViTs), has significantly advanced computational pathology, enhancing both diagnostic accuracy and efficiency. Whole Slide Imaging (WSI) plays a crucial role by providing detailed tissue samples for the analysis and training of DL models. However, WSIs often contain regions with artifacts such as tissue folds, blurring, as well as non-tissue regions (background), which can negatively impact DL model performance in diagnostic tasks. These artifacts are diagnostically irrelevant and can lead to inaccurate results. This paper proposes a fully automatic DL pipeline for WSI Quality Assessment (WSI-QA) that uses a fused model combining CNNs and ViTs to detect and exclude WSI regions with artifacts, ensuring that only qualified WSI regions are used to build DL-based computational pathology applications. The proposed pipeline employs a pixel-based segmentation model to classify WSI regions as either qualified or non-qualified based on the presence of artifacts. The proposed model was trained on a large and diverse dataset and validated with internal and external data from various human organs, scanners, and staining procedures. Quantitative and qualitative evaluations demonstrate the superiority of the proposed model, which outperforms state-of-the-art methods in WSI artifact detection. The proposed model consistently achieved over 95% accuracy, precision, recall, and F1 score across all artifact types. Furthermore, the WSI-QA pipeline shows strong generalization across different tissue types and scanning conditions.
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
  conda create -n WSI-QA python=3.9
```
- Activate the environment:
```bash
  conda activate WSI-QA
```
- Install required packages:
```bash
  pip install -r requirements.txt
```


# Datasets:

- Contact the corresponding author to access the datasets described in the paper.
- The datasets are available for research purposes only.

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
  - Excel file contains statistics on identified artifacts
  - A folder named `qualified` containing qualified tiles
  - A folder named `unqualified` containing unqualified tiles
- You can adjust the testing parameters on line 67 of `test_wsi.py`
- If your WSI image has a format other than .svs or .mrxs, please modify line 143 in `test_wsi.py`
- It is recommended to use a tile size of 270 × 270 pixels
- To generate a high-resolution thumbnail image and segmentation mask, you can adjust the `thumbnail_size` parameter in `test_wsi.py`. However, note that this will increase the execution time.


# Training:

- Visit https://drive.google.com/drive/folders/1mbnLH1JIztTMw7Cgv8pSzNxba-aGv1jT?usp=share_link and download the develpment artifact datasets (and external validation dataset)
- Extract and place the dataset into a folder named `train_dataset`
- Within `train_dataset`, refer to the example files provided to understand the structure. First run `python train.py` as a demo to check if the pipeline is ready 
- Create two files, `train_images.txt` and `train_masks.txt`, with lists of the corresponding image and mask paths, that used for training.

     Example content for `train_images.txt`:
     ```
     path/to/image1.png
     path/to/image2.png
     ...
     ```
     Example content for `train_masks.txt`:
     ```
     path/to/mask1.png
     path/to/mask2.png
     ...
     ```
- Create an account on [Weights and Biases](https://docs.wandb.ai)
- After signing up, go to your account settings and obtain your API key. It will look something like: `wandb.login(key='xxxxxxxxxxxxx')`
- Open the file `trainer.py`
- Find the line where the Weights and Biases login is required
- Update it with your API key like this:
     ```python
     wandb.login(key='your_actual_key_here')
     ```
- Adjust the training parameters (e.g., epoch, learning rate) in `train.py` if needed
- Open a terminal in the directory where `train.py` is located
- Run the following command to start the training:
     ```bash
     python train.py
     ```
- When training starts, a link to the Weights and Biases interface will appear in the terminal
- Click on the link to track and visualize the progress of your training
- After the training is complete, the weights will be saved in the `logs` folder within your project directory

# Results 

![WSI-QA](./Performance-metrics.png)

![WSI-QA](./WSI-Segmentation.png)


# Acknowledgment:

Some parts of this pipeline were adapted from work on [GitHub](https://github.com/pengsl-lab/DHUnet). If you use this pipeline, please make sure to cite their work.


# Contact: 
If you have any questions or comments, please feel free to contact: falah.rahim@unn.no
