# Stroke Lesion Segmentation UNet
Pytorch implementation of a 2D U-Net for stroke lesion segmentation

## Installation
* Clone this repo
```
git clone https://github.com/ZihaoChen0319/Stroke-Lesion-Segmentation-UNet
cd Stroke-Lesion-Segmentation-UNet
```
* Install required python packages with pip and requirements.txt
```
pip install -r requirements.txt
```

## Data
* Data used here comes from [sub-acute ischemic stroke lesion segmentation challenge (SISS), Ischemic Stroke Lesion Segmentation Challenge (ISLES) 2015](http://www.isles-challenge.org/ISLES2015/). The dataset is not provided here. which is recommended for you to register by yourself.
* In the first time, enable the `preprocess` line in `./main2D.py` to transform the raw dataset to 2D slices and save it in `./data/`, which contributes a lot to training speed.
* The `./data/` and `./raw_data/` foldder provides examples of data structure.
