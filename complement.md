### Introducton:

This project is part of the Deep learning for autonomous vehicule class. Each group had to solve a specific task related to autonomous driving. Our group chose the task "traffic sign detection and classification", where we have to find the bounding boxes and class labels of signs, given an image. As describe in the milestone 1, we chose to apply and finetune the model Yolov8 to this task. Yolov8 is the state-of-the-art deep learning model for object detection, develloped by ultranalytics and realease in 2023. In this document, we will explain how to obtain our results and comment them.

### Instalation:

All the script are designed to be run on the scitas clusters. One can connect and request ressources with the following commands:
```
ssh -X <gaspar_username>@izar.epfl.ch
Sinteract  -g gpu:1 -p gpu -c 8 -m 30G -t 01:29:00 -a civil-459-2023
```

Once on a node in the scitas clusters, one must setup the virtual environment.
```
virtualenv --system-site-packages <venvs_name>
source <venvs_name>/bin/activate
pip install ultralytics --user
pip install thop --user
```

Then a module is required to install
```
module load gcc/8.4.0-cuda python/3.7.7		
```

Once all of that is done, the scripts are ready to be runned.
Please note that the scripts exepts the available ressource to be 8 cores when running. If there is not enough ressource available, a warning will be displayed, but it should nevertheless work.

