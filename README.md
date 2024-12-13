# Replicated_TemAdapter
This repository contains the replication of the Tem-Adapter model, which adapts image-text pretraining for video question answering. The original implementation is detailed in the [Tem-Adapter repository](https://github.com/XLiu443/Tem-adapter).

## Repository Structure

- **configs/**: Contains configuration files specifying model parameters and training settings.
- **data/**: Intended for storing datasets and preprocessed features.
- **model/**: Includes the implementation of model architectures, such as the Temporal and Semantic Aligners.
- **preprocess/**: Scripts for preprocessing data, including feature extraction using pre-trained models like CLIP.
- **DataLoader.py**: Manages data loading and batching during training and evaluation.
- **SemanticAligner.py**: Implements the Semantic Aligner module for refining textual embeddings.
- **config.py**: Handles configuration settings for the project.
- **requirements.txt**: Lists the Python dependencies required to run the project.
- **train.py**: Script to initiate model training.
- **utils.py**: Contains utility functions used across the project.
- **validate.py**: Script for evaluating the trained model's performance.


## Environment Setup

1. Install PyTorch (should install [miniconda](https://docs.conda.io/en/latest/miniconda.html) first):

```bash
conda create --name myenv python=3.7
conda activate myenv
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

2. Install dependencies 

```bash
conda install -c conda-forge ffmpeg
conda install -c conda-forge scikit-video
pip install ftfy regex tqdm
pip install timm
pip install jsonlines
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

## Download the dataset and Pre-process with CLIP visual encoder

The SUTD-TrafficQA dataset is publicly released. You can download the original videos and text annotations from https://sutdcv.github.io/SUTD-TrafficQA/#/explore

You can use OpenAI's CLIP as the pre-trained image encoder (ViT-B32). The following instructions can be followed.

1. Create a folder `./data/` in current directory, such as:
```
Tem-adapter/
|–– configs/
|–– data/
|–– model/
|–– ...
```

2. Unzip downloaded video file 'raw_videos.zip' to 'data' as `./data/raw_videos/`.

3. Put the downloaded annotation file 'R3_all.jsonl' to 'data' as `./data/annotation_file/R3_all.jsonl`.


The directory should have the following structure:

```
Tem-adapter/
|–– configs/
|–– data/
|   |–– raw_videos
|       |–– b_1a4411B7sb_clip_005.mp4
|       |–– b_1a4411B7sb_clip_006.mp4
|       |__ ...  
|   |-- annotation_file
|       |–– R3_all.jsonl
|–– model/
|–– ...
```

4. Run the following command to extract features with the CLIP visual encoder.

```bash
python preprocess/preprocess_features.py --gpu_id 0 --dataset sutd-traffic --model clip_image 
```
Then there will be a new folder `./data/sutd-traffic/` under the current path.


5. Download the texts (QA pairs) from [here](https://drive.google.com/drive/folders/1NgfWg6MBD_LYGBXlJqUtlZ52ZEivFSKE?usp=sharing) and put them under the path `./data/sutd-traffic/` 

The dataset directory should have the following structure:

```
Tem-adapter/
|–– configs/
|–– data/
|   |–– raw_videos
|       |–– b_1a4411B7sb_clip_005.mp4
|       |–– b_1a4411B7sb_clip_006.mp4
|       |__ ...
|   |-- annotation_file
|       |–– R3_all.jsonl
|   |-- sutd-traffic
|       |–– sutd-traffic_transition_appearance_feat.h5
|       |–– output_file_train.jsonl
|       |–– output_file_test.jsonl
|–– model/
|–– ...
```

## Evaluate the trained model

1. Create a new folder "pretrained" under the path 'Tem-adapter/'

2. Download the trained checkpoints from this [link](https://drive.google.com/drive/folders/1SplEKEjrp-Uw-PxziyBHvUuU-yQ0YevX?usp=sharing) and put them under the path 'Tem-adapter/pretrained/'


The directory should have the following structure:

```
Tem-adapter/
|–– configs/
|–– data/
|   |–– raw_videos
|       |–– b_1a4411B7sb_clip_005.mp4
|       |–– b_1a4411B7sb_clip_006.mp4
|       |__ ...
|   |-- annotation_file
|       |–– R3_all.jsonl
|   |-- sutd-traffic
|       |–– sutd-traffic_transition_appearance_feat.h5
|       |–– output_file_train.jsonl
|       |–– output_file_test.jsonl
|–– model/
|–– pretrained
    |-- semanticaligner_49.pt
    |-- tempaligner_49.pt
|–– ...
```

3. Uncomment related lines in the 'validate.py' (Check the file for further reference).

4. To evaluate the trained model, run the following command:

```bash
python validate.py --cfg configs/sutd-traffic_transition.yml
```




## Training

Choose the config file in 'configs/sutd-traffic_transition.yml', run the following command:

```bash
python train.py --cfg configs/sutd-traffic_transition.yml
```


## Evaluation

To evaluate the trained model, run the following:

```bash
python validate.py --cfg configs/sutd-traffic_transition.yml
```

