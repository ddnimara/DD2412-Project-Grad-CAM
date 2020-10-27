import numpy as np
import torch
import pandas as pd
import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from captum.attr import IntegratedGradients, GradientShap
import src.experiments.chest_xray.xray_localization as xl
from src.methods.gradCAM import *
from src.datasets import ResizedChestXRayDataset

def experiment(dataset_csv, image_folder, model_name="chexnet", method="ig", plot=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = ResizedChestXRayDataset(dataset_csv, image_folder)
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    model, layer_name = xl.load_model(model_name)
    model = model.to(device)
    model.eval()
    
    # Load method
    if method == "ig":
        method = IntegratedGradients(model)
    else:
        method = GradientShap(model)
        
    scores = []

    # Iterate through batches
    for batch_idx, (images, labels, true_bboxes) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Calculate attribution
        baselines = torch.zeros_like(images).to(device)
        attributions, delta = method.attribute(images,
                                               baselines=baselines,
                                               target=labels,
                                               return_convergence_delta=True)
        attributions = attributions.detach().cpu().numpy()
        
        for image, label, true_bbox, attribution in zip(images, labels, true_bboxes, attributions):
            score = process_attributions(attribution, image=image, label=label, true_bbox=true_bbox, plot=plot)
            scores.append(score)
    
    return np.array(scores)

def experiment_gradcam(dataset_csv, image_folder, model_name="chexnet", plot=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = ResizedChestXRayDataset(dataset_csv, image_folder)
    batch_size = 16
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model, layer_name = xl.load_model(model_name)
    model = model.to(device)
    model.eval()
        
    # Load method
    gcm = gradCAM(model, [layer_name])
    
    scores = []

    # Iterate through batches
    for batch_idx, (images, labels, true_bboxes) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Calculate heatmap
        gcm.forward(images)
        mapCam = gcm.generateMapClassBatch(labels)
        heatmaps = gcm.generateCam(mapCam, layer_name, image_path=None, mergeWithImage=False, isBatch=True)
        
        for image, label, true_bbox, heatmap in zip(images, labels, true_bboxes, heatmaps):
            score = process_attributions(heatmap, image=image, label=label, true_bbox=true_bbox, 
                                         plot=plot, transpose=False)
            scores.append(score)
    
    return np.array(scores)

def process_attributions(attributions, image, label, true_bbox, plot=False, transpose=True):
    # Show image
    if plot:
        ax = plt.gca()
        ax.imshow(image.detach().numpy().transpose(1, 2, 0))
        xl.draw_bounding_box(true_bbox.tolist(), ax)
        plt.show()
        
    if transpose:
        # Transpose axes
        attributions = attributions.transpose(1, 2, 0)
        
        # Convert to grayscale
        attributions = np.dot(attributions[..., :3], [0.2989, 0.5870, 0.1140])
    
    # Remove negatives
    attributions = (attributions > 0) * attributions
    
    # Normalize
    min_val = np.min(attributions)
    attributions = attributions - min_val
    max_val = np.max(attributions)
    attributions /= max_val
    
    if plot:
        ax = plt.gca()
        ax.imshow(attributions)
        xl.draw_bounding_box(true_bbox.tolist(), ax)
        plt.show()
    
    # Thresholding
    attributions = (attributions > 0.5) * attributions
    
    if plot:
        ax = plt.gca()
        ax.imshow(attributions)
        xl.draw_bounding_box(true_bbox.tolist(), ax)
        plt.show()
    
    # Add values
    sum_points = np.count_nonzero(attributions)
    
    # Sum points that are within the bounding box
    x_min = int(true_bbox[0])
    y_min = int(true_bbox[1])
    x_len = int(true_bbox[2])
    y_len = int(true_bbox[3])
    x_max = x_min + x_len
    y_max = y_min + y_len
    
    attributions_wi_bbox = np.zeros_like(attributions)
    attributions_wi_bbox[y_min:y_max, x_min:x_max] = attributions[y_min:y_max, x_min:x_max]
    sum_points_wi_bbox = np.count_nonzero(attributions_wi_bbox)
    
    if sum_points == 0:
        score = 0
    else:
        score = (sum_points_wi_bbox / sum_points)
    print(score)
    
    if plot:
        plt.imshow(attributions_wi_bbox)
        plt.show()
        
    return score
    

torch.manual_seed(123)
np.random.seed(123)

# Retrieved from https://drive.google.com/file/d/1w8uLZlKhdVsss2yN934hTXMBWpV52itG/view
image_folder = "../../../datasets/chest-xray/images_small/"
# Dataset metadata was retrieved from https://www.kaggle.com/nih-chest-xrays/data
dataset_csv  = "../../../datasets/chest-xray/Resized_BBox_List.csv"

scores = experiment(dataset_csv, image_folder, model_name="chexnet", method="ig", plot=False)
np.savetxt("result_chexnet_ig.csv", scores, delimiter=",")
print("Chexnet, IG:")
print("Mean:", np.average(scores))
print("Standard deviation:", np.std(scores))
print()

scores = experiment(dataset_csv, image_folder, model_name="chexnet", method="shap", plot=False)
np.savetxt("result_chexnet_shap.csv", scores, delimiter=",")
print("Chexnet, SHAP:")
print("Mean:", np.average(scores))
print("Standard deviation:", np.std(scores))
print()

scores = experiment(dataset_csv, image_folder, model_name="covid-pretrained", method="ig", plot=False)
np.savetxt("result_covid_pretrained_ig.csv", scores, delimiter=",")
print("Covid pre-trained, IG:")
print("Mean:", np.average(scores))
print("Standard deviation:", np.std(scores))
print()

scores = experiment(dataset_csv, image_folder, model_name="covid-pretrained", method="shap", plot=False)
np.savetxt("result_covid_pretrained_shap.csv", scores, delimiter=",")
print("Covid pre-trained, SHAP:")
print("Mean:", np.average(scores))
print("Standard deviation:", np.std(scores))
print()

#######################################################################################xx

scores = experiment_gradcam(dataset_csv, image_folder, model_name="chexnet", plot=False)
np.savetxt("result_chexnet_gradcam.csv", scores, delimiter=",")
print("Chexnet, Grad-CAM:")
print("Mean:", np.average(scores))
print("Standard deviation:", np.std(scores))
print()

scores = experiment_gradcam(dataset_csv, image_folder, model_name="covid-pretrained", plot=False)
np.savetxt("result_covid_pretrained_gradcam.csv", scores, delimiter=",")
print("Chexnet, Grad-CAM:")
print("Mean:", np.average(scores))
print("Standard deviation:", np.std(scores))
print()