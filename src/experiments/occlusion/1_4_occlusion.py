from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
import src.methods.new_grad_cam as ngc
from src.utilities import *
from src.dataSetLoader import *
import matplotlib.pyplot as plt
from os import path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from ast import literal_eval

# Evaluating faithfulness via image occlusion by calculating the rank correlation of patches that change the CNN
# score and patches that have high heatmap values for Grad-CAM and Guided Grad-CAM on the ILSVRC-12 val
# dataset. Rank correlation should be averaged over 2510 images.

def calculate_rank_correlation(model, df, layer, guided_gradcam=False, plot=False, use_pred=False):
    # Get device (so you can use gpu if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    df_count_size = df.shape[0]

    # Create data loader
    validationSet = dataSetILS(df, transforms)
    batch_size = 1 # Do not change!
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)

    # Run Grad-CAM
    gcm = gradCAM(model, layer)
    if guided_gradcam:
        gbp = guidedBackProp(model)
    
    it = 0  # we will use to keep track of the batch location
    scores = []

    # Loop through batches
    for batch_images in tqdm(validationLoader):
        batch_images = batch_images.to(device)

        # df_view gives us access the dataframe on the batch window
        df_view = df.iloc[it:min(it + batch_size, df_count_size)]
        gcm.forward(batch_images)
        if guided_gradcam:
           gbp.forward(batch_images)

        if use_pred: # Use top-1 prediction as label
            batch_prob, batch_label = gcm.getTopK(k=1)
            batch_prob = batch_prob[:, 0]
            batch_label = batch_label[:, 0]
        else: # Use one of the true labels
            true_label = literal_eval(df_view.iloc[0].loc["id"])[0] # Using iloc[0] because there's only 1 element in the batch
            batch_prob = gcm.probs[:, true_label]
            batch_label = torch.Tensor([true_label]).to(torch.int64).to(device)
        
        # generate the heatmaps
        mapCam = gcm.generateMapClassBatch(batch_label)
        heatmaps = gcm.generateCam(mapCam, layer[0], image_path=None, mergeWithImage=False, isBatch=True)
        if guided_gradcam:
            mapGuidedBackProp = gbp.generateMapClassBatch(batch_label)
            gradientNumpy = gradientToImageBatch(mapGuidedBackProp)
            heatmaps = heatmaps * gradientNumpy
            
        # Generate the occlusion maps
        with torch.no_grad():
            occlusion_maps = occlusion(model, batch_images, batch_label, batch_prob, invert=True)

        for i in range(batch_size):  # iterate over batch
            hm = heatmaps[i]
            occlusion_map = occlusion_maps[i]
                
            # Convert heatmap to grayscale image if it's 3D
            if guided_gradcam:
                hm = np.dot(hm[..., :3], [0.2989, 0.5870, 0.1140])
            
            # Normalize the occlusion map
            occlusion_map = occlusion_map - occlusion_map.min()
            occlusion_map = occlusion_map / occlusion_map.max()

            # Calculate rank correlation between heatmap and occlusion map
            hm_rank = get_rank(hm)
            occlusion_map_rank = get_rank(occlusion_map)
            coeff, p = spearmanr(hm_rank, occlusion_map_rank)
            scores.append(coeff)
            print(coeff)
            
            # Plot images
            if plot:
                ngc.gradCAM.plot_heatmap(batch_images[i], hm)
                plt.show()
                
                plt.imshow(occlusion_map, cmap=plt.cm.bwr_r, vmin=0, vmax=1)
                plt.colorbar()
                plt.show()
            
        it += batch_size
        
    # Return a list of hits and misses + max activations
    return np.array(scores)

def get_rank(map):
    return np.argsort(map.flatten())
    
def occlusion(model, images, label, prob, invert=True):
    original_images = images.clone()
    
    nr, ch, h, w = images.shape
    block_h = 45 # Path height
    block_w = 45 # patch width
    mean = 0.5 # Color used for padding and filling the patches (should be gray by default)
    batch_s = 160 # Number of modified images to collect for forward pass
    
    occlusion_maps = np.empty((nr, h, w))
    
    # Pad original images
    pad_h, pad_w = block_h // 2, block_w // 2
    original_images = F.pad(original_images, (pad_w, pad_w, pad_h, pad_h), value=mean)
    
    for n in range(nr):
        it = 0
        images_to_process = []
        batch_prob = []
        for i in tqdm(range(0, h, 1)): # Stride is 1
            for j in range(0, w, 1):
                # Cover a segment of the original image
                image = original_images[n].clone().unsqueeze(0)
                image[:, :, i:(i + block_h), j:(j + block_w)] = mean
                images_to_process.append(image)
                
                if (it + 1) % batch_s == 0 or i == h - 1 and j == w - 1:
                    # Forward pass
                    images_to_process = torch.cat(images_to_process, dim=0)
                    
                    pred = F.softmax(model(images_to_process), dim=1)
                    pred = (pred[:, int(label[n])]).detach().cpu().numpy().tolist()
                    batch_prob.extend(pred)
                    del images_to_process
                    images_to_process = []
                
                it += 1
             
        it = 0
        for i in range(0, h, 1):
            for j in range(0, w, 1):
                # Fill occlusion map
                occlusion_maps[n, i, j] = batch_prob[it]
                it += 1
        
    if invert:
        occlusion_maps = 1 - occlusion_maps
        
    return occlusion_maps

df = pd.read_csv("../../../datasets/res2_2510.csv")
print("Running Grad-CAM on VGG16 and calculating rank correlation with occlusion maps...")
scores = calculate_rank_correlation(getVGGModel(16), layer=['features.29'], df=df, guided_gradcam=False, plot=False, use_pred=False)
print("Average score: ", np.average(scores))

print("Running Guided Grad-CAM on VGG16 and calculating rank correlation with occlusion maps...")
scores = calculate_rank_correlation(getVGGModel(16), layer=['features.29'], df=df, guided_gradcam=True, plot=False, use_pred=False)
print("Average score: ", np.average(scores))
