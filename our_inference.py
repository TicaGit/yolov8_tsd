import argparse

from test_yolo_data_cluster import YOLOCustom

def write(preds, output_file):
    with open(output_file, "w+") as f:
        f.write("img_id,class,box")
        for pred in preds:
            for box in pred.boxes:
                f.write(str(pred.name)+","+str(box.cls)+","+str(box.xywhn)) #format wrong

if __name__ == "__main__":

    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='our_retrained_models/yolov8n_tsd.pt',
                            help='path of the model')
    parser.add_argument('--data_path', default="",
                            help='path of the data')
    parser.add_argument('--output_file', default="prediction.txt",
                        help='path to store the output predictions')
    args = parser.parse_args()

    model = YOLOCustom(model = args.model_path)
    preds = model(args.data_path + "train.txt")
    write(preds, args.output_file)
    print("prediction done, stroed at {args.data_path}")
