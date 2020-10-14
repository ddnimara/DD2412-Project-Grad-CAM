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
from tqdm import tqdm
import json
from scipy.stats import spearmanr

# Evaluating faithfulness via image occlusion by calculating the rank correlation of patches that change the CNN
# score and patches that have high heatmap values for Grad-CAM and Guided Grad-CAM on the ILSVRC-12 val
# dataset. Rank correlation should be averaged over 2510 images.

def calculate_rank_correlation(model, df, layer, guided_gradcam=False, plot=False):
    # TODO guided Grad-CAM
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
    scores = []

    # Loop trough batches
    for batch_images in tqdm(validationLoader):
        batch_images = batch_images.to(device)

        # df_view gives us access the dataframe on the batch window
        df_view = df.iloc[it:min(it + batch_size, df_count_size)]
        gcm.forward(batch_images)

        # Get ImageNet classes
        imagenetClasses = getImageNetClasses()

        batch_prob, batch_label = gcm.probs.topk(1, dim = 1)
        batch_prob = batch_prob[:, 0]
        batch_label = batch_label[:, 0]
    
        # generate the heatmaps
        map = gcm.generateMapClassBatch(batch_label)
        heatmap = gcm.generateCam(map, layer[0], image_path=None, guided=True, isBatch=True)

        for i in range(heatmap.shape[0]):  # iterate over batch
            # generate the occlusion maps
            occlusion_map = occlusion(model, batch_images[i], batch_label[i], batch_prob[i], plot=plot)
            
            # Rank correlation between heatmap and occlusion map
            coeff, p = spearmanr(heatmap[i][0], occlusion_map)
            scores.append(coeff)

        it += batch_size
        
    # Return a list of hits and misses + max activations
    return np.array(scores)
    
def occlusion(model, image, label, prob, plot=False):
    prob = prob.detach().numpy()
    label = label.detach().numpy()
    original_image = image.clone()
    
    h = image.shape[1]
    w = image.shape[2]
    block_h = 7
    block_w = 7
    
    occlusion_map = np.empty((h, w))
    
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            # Cover a segment of the original image
            image = original_image.clone()
            image[:, i:(i + block_h), j:(j + block_w)] = 0
            
            # Forward pass
            image = image.unsqueeze(0)
            pred = F.softmax(model(image), dim=1)
            batch_prob = pred[:, label]
            batch_prob = batch_prob.detach().numpy()[0]
            
            # Fill occlusion map
            occlusion_map[i:(i + block_h), j:(j + block_w)] = batch_prob
            
    if plot:
        plt.imshow(occlusion_map, cmap=plt.cm.seismic, vmin=0, vmax=1)
        plt.colorbar()
        plt.show()
        
    return occlusion_map

df = pd.read_csv("../../../datasets/res.csv")
print("Running Grad-CAM on VGG16 and calculating rank correlation with occlusion maps...")
scores = calculate_rank_correlation(getVGGModel(16), layer=['features'], df=df, guided_gradcam=False, plot=False)
print("Average score: ", np.average(scores))

print("Running Guided Grad-CAM on VGG16 and calculating rank correlation with occlusion maps...")
scores = calculate_rank_correlation(getVGGModel(16), layer=['features'], df=df, guided_gradcam=True, plot=False)
print("Average score: ", np.average(scores))