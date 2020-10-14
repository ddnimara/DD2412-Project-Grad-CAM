from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
from src.dataSetLoader import *
import matplotlib.pyplot as plt
from os import path
import cv2
import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json

# Measuring the localization accuracy of Grad-CAM in the context of the Pointing Game on 
# the ILSVRC-12 val datasets, using VGG-16 and GoogLeNet

def get_hit_or_miss(model, df, layer, k = 1):
    # Get device (so you can use gpu if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    df_count_size = df.shape[0]

    # Create data loader
    validationSet = dataSetILS(df, transforms)
    batch_size = 16
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)

    # Run Grad-CAM
    gcm = gradCAM(model, layer)
    
    it = 0  # we will use to keep track of the batch location
    
    # Store whether the maximally activated point is within the corresponding bounding box.
    # 0 if can't find corresponding ground truth (i.e. the prediction is wrong)
    hits_and_misses = np.zeros((df_count_size, k))
    
    # Store whether the prediction is good (i.e. it matches at least one of the true labels)
    good_predictions = np.zeros((df_count_size, k))
    
    # Store max. activations
    max_acts = np.zeros((df_count_size, k))

    # Get ImageNet classes
    imagenetClasses = getImageNetClasses()

    # Loop through batches
    for batch_images in tqdm(validationLoader):
        batch_images = batch_images.to(device)

        # df_view gives us access the dataframe on the batch window
        df_view = df.iloc[it:min(it + batch_size, df_count_size)]
        gcm.forward(batch_images)

        # get top k classes (indeces)
        topk = gcm.probs.sort(dim=1, descending=True)[1].cpu().numpy()[:,:k]
    
        # loop through top k classes
        for classNumber in range(k):  # loop through likeliest predictions
            batch_label = topk[:, classNumber]
            # generate the heatmaps
            map = gcm.generateMapClassBatch(torch.from_numpy(batch_label).to(device))
            heatmap = gcm.generateCam(map, layer[0], image_path=None, mergeWithImage=False, isBatch=True, rescale=False)

            for i in range(heatmap.shape[0]):  # iterate over batch
                max_ind = np.unravel_index(heatmap[i].argmax(), heatmap[i].shape)
                max_acts[it + i, classNumber] = heatmap[i].max()
                truthLabels = json.loads(df_view.iloc[i].loc["id"])
                truthBoxes = json.loads(df_view.iloc[i].loc["bounding box"])
                
                for index, true_class in enumerate(truthLabels):  # loop through the true labels
                    if int(true_class) != int(batch_label[i]):  # find matching object
                        continue
                    
                    bb = truthBoxes[index]

                    # Check whether the maximally activated point is within the relevant bounding box
                    x_min, y_min, x_max, y_max = bb
                    x, y = max_ind[1], max_ind[0]
                    is_inside_box = (x >= x_min and x <= x_max and y >= y_min and y <= y_max)
                    hits_and_misses[it + i, classNumber] = max(hits_and_misses[it + i, classNumber], is_inside_box)
                    good_predictions[it + i, classNumber] = 1
                    
        it += batch_size
        
    # Return a list of hits and misses + max activations
    return hits_and_misses, good_predictions, max_acts
    
def get_localization_accuracy(hits_and_misses):
    hits_and_misses_firstcol = hits_and_misses[:, 0]
    hit = np.count_nonzero(hits_and_misses_firstcol)
    miss = np.count_nonzero(hits_and_misses_firstcol == 0)
    return hit / (hit + miss)

def get_localization_recall(hits_and_misses, good_predictions, max_acts, threshold=0.5):
    hits_and_misses_new = np.logical_or( \
        np.logical_and(hits_and_misses, max_acts >= threshold), \
        #hits_and_misses,
        np.logical_and(np.logical_not(good_predictions), max_acts < threshold) \
        )
    hit = np.count_nonzero(hits_and_misses_new)
    miss = np.count_nonzero(hits_and_misses_new == 0)
    return hit / (hit + miss)

def pointing_game(model, layer, df, threshold=0.5):
    hits_and_misses, good_predictions, max_acts = get_hit_or_miss(model, layer=layer, df=df, k=5)
    accuracy = get_localization_accuracy(hits_and_misses)
    recall = get_localization_recall(hits_and_misses, good_predictions, max_acts, threshold=threshold)
    return accuracy, recall

df = pd.read_csv("../../../datasets/res.csv")
print("Running Pointing Game on VGG16...")
accuracy, recall = pointing_game(getVGGModel(16), layer=['features.29'], df=df)  # features
print("Accuracy: ", accuracy)
print("Recall: ", recall)

# print("Running Pointing Game on GoogLeNet...")
# accuracy, recall = pointing_game(getGoogleModel(), layer=['inception5b'], df=df)  # inception5b.branch4.1.conv
# print("Accuracy: ", accuracy)
# print("Recall: ", recall)
