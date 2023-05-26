add all raw models are in the trained_models folder
download from https://github.com/ultralytics/assets/releases : take n,m,x

all trained model on drive : put in our_retrained_models

our retrained models on the BTSD dataset are in the folder our_retrained_models

test that everything works on a new environment (load the right files, and not the ultralytics librairy)

data in /work/vita/...

use val function of yolo for evaluate performanace of models 

if other data loc, change path in streamloader.py : data_dir = "/work/vita/nmuenger_trinca/annotations/"


# YODSO : YOU ONLY DETECT SIGN ONCE

### instalation:

```
pip instal ...
ultralytics
```


### data:

path in .yalm + hardcoded in stream_loader

data are in scitas

say wich sign corespond to which class name

The defined traffic signs are given in the image at the end of the readme.

### models:

Before everything, one must first download the models from google drive and put them in the correct files.  
There are two kinds of models. The models pretrained by the ultralytics team on the coco dataset and our models which we finetuned on the BelgiumTS dataset. Both kinds comes with 3 different size : nano, medium and x-tra large. 

The pretrained models must first be trained with our training script. Our finetuned models can directly be used for inference. 

The pretrained models form ultralytics must be placed in the *trained_models* folder and our finetuned models in the *our_retrained_models* folder.



### training:

The script to train the network is called our_train.py. One can run it with this command, specifying the model to use and the number of epochs.
```
python our_train.py --model yolov8m.pt --epochs 3
```
The model trained and some training metrics are then stored in the *run/detect* folder.

### inference:

To infer the bounding boxes and classes of an image, we use the *our_inference.py* script. One can specify the model to use, the path to the image (wich must be in the same folder as the path specified in the .yaml file) and the file to outpur the predictions. An example command is provided below.

```
python our_inference.py --model_path yolov8m_tsd_30epochs.pt --data_path images/01/image.006950.jp2 --output_file our_predictions/prediction.txt
```

### testing:

change in the yaml file

<p align="center">
<img src="https://github.com/TicaGit/yolov8_tsd/blob/tibo_yolo_retrain/defined_sign.png" width="400">
</p>






