import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from src.datasets import ResizedChestXRayDataset
from src.utilities import getImageNetClasses
from src.methods.new_grad_cam import GradCAM as My_GCAM

from shapely.geometry import Polygon

from chexnet import DenseNet121
from collections import OrderedDict

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
    # NOTE: bounding boxes were resized using 'resize_bounding_boxes.py'!|
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
    # Retrieved from https://drive.google.com/file/d/1w8uLZlKhdVsss2yN934hTXMBWpV52itG/view
    image_folder = "../../../datasets/chest-xray/images_small/"
    # Dataset metadata was retrieved from https://www.kaggle.com/nih-chest-xrays/data
    dataset_csv  = "../../../datasets/chest-xray/Resized_BBox_List.csv"

    dataset = ResizedChestXRayDataset(dataset_csv, image_folder)
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, layer_name = load_model("chexnet")

    gradcam_object = My_GCAM(model, layer_name)


    results = pd.DataFrame(
        columns=["label", "class_prob", "is_hit", "iou_score", "iou_threshold"])

    for batch_idx, (images, labels, true_bboxes) in enumerate(tqdm(dataloader)):
        true_class_names = [dataset.index_to_label[l.item()] for l in labels]
        heatmaps = gradcam_object.generate_heatmaps(images, labels)
        label_probs = [gradcam_object.logits[i, l] for i,l in enumerate(labels)]

        # Iterate over all heatmaps in the batch
        for idx, heatmap in enumerate(heatmaps):
            pred_bbox = generate_bounding_box(heatmap, 0.5)
            true_bbox = true_bboxes[idx].tolist()
            
            if len(pred_bbox) == 0:
                iou_score = 0
            else:
                iou_score = compute_iou(pred_bbox, true_bbox)
                
            threshold = compute_threshold(true_bbox)
            is_hit = iou_score >= threshold
            
            image_idx = (batch_size*batch_idx)+idx

            results.loc[image_idx] = {
                "label" : true_class_names[idx], 
                "class_prob": label_probs[idx].item(),
                "is_hit": is_hit,
                "iou_score": iou_score,
                "iou_threshold": threshold}
    results.to_csv("results.csv")
            # print(, true_class_names[idx], label_probs[idx].item(), "is hit:", is_hit, iou_score)

def load_model(name):
    if name == "chexnet":
        ckpt_path = "chexnet.pt"
        state_dict = torch.load(ckpt_path, map_location=DEVICE)['state_dict']
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            for layer in ["norm.1", "conv.1", "norm.2", "conv.2"]:
                name = name.replace(layer, layer[:-2] + layer[-1])
                
            new_state_dict[name] = v        
            
        model = DenseNet121(14)
        model.load_state_dict(new_state_dict)
        layer_name = "densenet121.features.denseblock4.denselayer16.conv2"
        print("Loaded CheXNet.")
    elif name == "covid-pretrained":
        # Retrieved from https://github.com/gustavo-beck/DD2424-Deep_Learning-COVID-Project/tree/master/saved_models
        model_file = "final_model_16.pt"
        model = torch.load(model_file, map_location=DEVICE)
        layer_name = "features.denseblock4.denselayer16.conv2"
        print(f"Loaded {name}.")
    else:
        print(f"ERROR: unknown model name {name}. Possible values are 'chexnet' and 'covid-pretrained'.")
        exit()
    
    return model, layer_name

def compute_threshold(true_bbox):
    _, _, w, h = true_bbox
    
    return min(0.5, w*h / ((w+10)*(h+10)))

def generate_bounding_box(heatmap, threshold_pct):
    heatmap = heatmap_threshold(heatmap, threshold_pct)
    
    contours = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) <= 2 else contours[1]
    
    best = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if len(best) == 0:
            best = [x, y, w, h]
        else:
            if w * h > best[2] * best[3]:  # found bigger square
                best = [x, y, w, h]
                
    return best # x,y,w,h values

def heatmap_threshold(heatmap, threshold_pct):
    max_value = heatmap.max()
    binarizedHeatmap = np.zeros_like(heatmap, dtype=np.uint8)
    idx = heatmap > threshold_pct * max_value
    binarizedHeatmap[idx] = 255

    return binarizedHeatmap

def draw_bounding_box(bbox, ax, color='r'):
    rectangle = patches.Rectangle(
        (bbox[0], bbox[1]), bbox[2], bbox[3], 
        linewidth=1, edgecolor=color, facecolor='none')

    ax.add_patch(rectangle)

def compute_iou(predicted_bbox, true_bbox):
    # the IOU score will be calculated using the Polygon class,
    # which requires the coordinates of the 4 corners of the bounding boxes
    x, y, w, h = predicted_bbox
    pred_coords = []
    pred_coords.append([x,     y])
    pred_coords.append([x + w, y])
    pred_coords.append([x + w, y + h])
    pred_coords.append([x,     y + h])
    
    x, y, w, h = true_bbox
    true_coords = []
    true_coords.append([x,     y])
    true_coords.append([x + w, y])
    true_coords.append([x + w, y + h])
    true_coords.append([x,     y + h])
    
    poly_1 = Polygon(pred_coords)
    poly_2 = Polygon(true_coords)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    
    return iou

if __name__ == "__main__":
    main()
    