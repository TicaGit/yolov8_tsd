from ultralytics import YOLO

model = YOLO('yolov8n.pt') #pretrained

# data_folder = "/work/nmuenger_trinca/annotations/" #real
data_folder = "/work/vita/nmuenger_trinca/annotations_reduced/" #for tests
#in yolo data dataloader l.180 : added this dir (hardcoded) :(

#NEED to test with data side to model (also slooooow ?)

pred = model(data_folder + "train.txt") #file that say where the images are
print(pred)

#check if everythink is on gpu '??