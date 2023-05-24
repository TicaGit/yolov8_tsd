import argparse
import matplotlib.pyplot as plt
import os
from test_yolo_data_cluster import YOLOCustom

if __name__ == "__main__":

    all_test_set = True
    if all_test_set:
        # CHANGE val:val.txt to test.txt in the yalm file !!!!!
        test_data_file = "tsr_dataset.yaml"  #rest of the path is hardcoded in stream_loader.py


        maps50 = []
        maps50_95 = []
        time = []

        #NANO
        model_path = "our_retrained_models/yolov8n_tsd_30epochs.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_data_file)
        print(f"Final results for yolov8 nano: mAP50 {metrics.results_dict['metrics/mAP50(B)']}, "+
            f"mAP50 {metrics.results_dict['metrics/mAP50-95(B)']} and inference time {metrics.speed['inference']}")
        maps50.append(metrics.results_dict['metrics/mAP50(B)'])
        maps50_95.append(metrics.results_dict['metrics/mAP50-95(B)'])
        time.append(metrics.speed['inference'])
        # Final results for yolov8 nano: mAP50 0.8417955264534036, mAP50 0.7083651463474747 and inference time 2.656630054692028

        #MEDIUM
        model_path = "our_retrained_models/yolov8m_tsd_30epochs.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_data_file)
        print(f"Final results for yolov8 medium: mAP50 {metrics.results_dict['metrics/mAP50(B)']}, "+
            f"mAP50 {metrics.results_dict['metrics/mAP50-95(B)']} and inference time {metrics.speed['inference']}")
        maps50.append(metrics.results_dict['metrics/mAP50(B)'])
        maps50_95.append(metrics.results_dict['metrics/mAP50-95(B)'])
        time.append(metrics.speed['inference'])
        # Final results for yolov8 medium: mAP50 0.8988986169684553, mAP50 0.8020364596757812 and inference time 5.640034697708059
        
        #LARGE
        model_path = "our_retrained_models/yolov8x_tsd_30epochs.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_data_file)
        print(f"Final results for yolov8 x-tra large: mAP50 {metrics.results_dict['metrics/mAP50(B)']}, "+
            f"mAP50 {metrics.results_dict['metrics/mAP50-95(B)']} and inference time {metrics.speed['inference']}")
        maps50.append(metrics.results_dict['metrics/mAP50(B)'])
        maps50_95.append(metrics.results_dict['metrics/mAP50-95(B)'])
        time.append(metrics.speed['inference'])
        # Final results for yolov8 x-tra large: mAP50 0.9343034899392261, mAP50 0.8510913090950882 and inference time 13.632428332863856

        plt.plot(time, maps50, label = "maps50")
        plt.plot(time, maps50_95, label = "maps50_95")
        plt.legend()
        plt.xlabel("Inference time [s]")
        plt.ylabel("mAP50 / mAP50-95")
        plt.title("Comparison of the models' complexity")

        labels = ["nano", "medium", "x-tra large"]
        for i, (x1,y1) in enumerate(zip(time,maps50_95)):

            label = labels[i]

            plt.annotate(label, # this is the text
                    (x1,y1), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(10,20), # distance from text to points (x,y)
                    ha='right')
        
        plt.savefig("our_predictions/random.png")
    
    else:
        test_image = "fafaa.txt"  #rest of the path is hardcoded in stream_loader.py

        print("here")
        time = []

        #NANO
        model_path = "our_retrained_models/yolov8n_tsd_30epochs.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_image)
        print(f"Final inference time for yolov8 nano {metrics.speed['inference']}")
        time.append(metrics.speed['inference'])

        #NANO
        model_path = "our_retrained_models/yolov8m_tsd_30epochs.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_image)
        print(f"Final inference time for yolov8 medium {metrics.speed['inference']}")
        time.append(metrics.speed['inference'])

        #NANO
        model_path = "our_retrained_models/yolov8x_tsd_30epochs.pt"
        model = YOLOCustom(model = model_path)
        metrics = model.val(data = test_image)
        print(f"Final inference time for yolov8 x-tra large {metrics.speed['inference']}")
        time.append(metrics.speed['inference'])

        plt.bar(["nano", "medium", "x-tra large"], time)
        plt.savefig("our_predictions/bar.png")