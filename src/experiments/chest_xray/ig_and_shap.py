import numpy as np
import torch
import pandas as pd
import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from captum.attr import IntegratedGradients, GradientShap

import src.experiments.chest_xray.xray_localization as xl
from src.datasets import ResizedChestXRayDataset

def experiment(model_name="chexnet", method="ig", plot=False):
    # Retrieved from https://drive.google.com/file/d/1w8uLZlKhdVsss2yN934hTXMBWpV52itG/view
    image_folder = "../../../datasets/chest-xray/images_small/"
    # Dataset metadata was retrieved from https://www.kaggle.com/nih-chest-xrays/data
    dataset_csv  = "../../../datasets/chest-xray/Resized_BBox_List.csv"

    # Load dataset
    dataset = ResizedChestXRayDataset(dataset_csv, image_folder)
    batch_size = 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model, layer_name = xl.load_model(model_name)
    
    # Load method
    if method == "ig":
        method = IntegratedGradients(model)
    else:
        method = GradientShap(model)

    # Iterate through batches
    for batch_idx, (images, labels, true_bboxes) in enumerate(dataloader):
        image = images[0]
        label = labels[0]
        true_bbox = true_bboxes[0]
        
        baselines = torch.zeros_like(images)
        attributions, delta = method.attribute(images,
                                               baselines=baselines,
                                               target=labels,
                                               return_convergence_delta=True)
        attributions = attributions[0].detach().cpu().numpy()
        
        # Show image
        if plot:
            ax = plt.gca()
            ax.imshow(image.detach().numpy().transpose(1, 2, 0))
            xl.draw_bounding_box(true_bbox.tolist(), ax)
            plt.show()
        
        # Transpose axes
        attributions = attributions.transpose(1, 2, 0)
        
        # Convert to grayscale
        attributions = np.dot(attributions[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Thresholding (remove negatives)
        attributions = (attributions > 0) * attributions
        
        # Normalize
        min_val = np.min(attributions)
        attributions = attributions - min_val
        max_val = np.max(attributions)
        attributions /= max_val
        
        if plot:
            plt.imshow(attributions)
            plt.show()
        
        if plot:
            ax = plt.gca()
            ax.imshow(attributions)
            xl.draw_bounding_box(true_bbox.tolist(), ax)
            plt.show()
        
        # Add values
        sum_points = np.sum(attributions)
        
        # Sum points that are within the bounding box
        x_min = int(true_bbox[0])
        y_min = int(true_bbox[1])
        x_len = int(true_bbox[2])
        y_len = int(true_bbox[3])
        x_max = x_min + x_len
        y_max = y_min + y_len
        
        attributions_wi_bbox = np.zeros_like(attributions)
        attributions_wi_bbox[y_min:y_max, x_min:x_max] = attributions[y_min:y_max, x_min:x_max]
        sum_points_wi_bbox = np.sum(attributions_wi_bbox)
        
        image_size = 224 * 224
        score = (sum_points_wi_bbox / x_len * y_len) / (sum_points / image_size)
        print(score)
        
        if plot:
            plt.imshow(attributions_wi_bbox)
            plt.show()
    
experiment(model_name="chexnet", method="ig", plot=True)
experiment(model_name="chexnet", method="shap", plot=True)
experiment(model_name="covid-pretrained", method="ig", plot=True)
experiment(model_name="covid-pretrained", method="shap", plot=True)