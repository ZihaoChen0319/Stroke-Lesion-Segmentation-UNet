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
* The whole dataset is not provided here.
* Data used in this project comes from [sub-acute ischemic stroke lesion segmentation challenge (SISS), Ischemic Stroke Lesion Segmentation Challenge (ISLES) 2015](http://www.isles-challenge.org/ISLES2015/).
* Data is processed by transforming them to 2D slices, normalizing and saving them in `./data/`, which contributes a lot to training speed.
* `./data/` and `./raw_data/` show examples of data structure.

## Train and test
* Run `./main2D.py` to start training process.
* After training, run `./evaluate.py` to evaluate the performance.
