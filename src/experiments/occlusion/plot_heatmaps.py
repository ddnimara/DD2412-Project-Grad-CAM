from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
import src.methods.new_grad_cam as ngc
from src.utilities import *
from src.dataSetLoader import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from ast import literal_eval
from skimage.transform import resize

def plot_heatmaps(model, df, layer, use_pred=False):
    # Get device (so you can use gpu if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    df_count_size = df.shape[0]

    # Create data loader
    validationSet = dataSetILS(df, transforms)
    validationSetUnnormalized = dataSetILS(df, to_tensor_transform)
    batch_size = 1 # Do not change!
    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=batch_size, shuffle=False)
    validationLoaderUnnormalized = torch.utils.data.DataLoader(validationSetUnnormalized, batch_size=batch_size, shuffle=False)

    # Run Grad-CAM
    gcm = gradCAM(model, layer)
    # Run Guided Backprop
    gbp = guidedBackProp(model)
    
    it = 0  # we will use to keep track of the batch location
    grad_cam_scores = []
    guided_grad_cam_scores = []
    
    # Loop through batches
    for batch_images, batch_images_unn in tqdm(zip(validationLoader, validationLoaderUnnormalized)):
        batch_images = batch_images.to(device)

        # df_view gives us access the dataframe on the batch window
        df_view = df.iloc[it:min(it + batch_size, df_count_size)]
        
        # Forward pass
        gcm.forward(batch_images)
        gbp.forward(batch_images)

        if use_pred: # Use top-1 prediction as label
            batch_prob, batch_label = gcm.getTopK(k=1)
            batch_prob = batch_prob[:, 0]
            batch_label = batch_label[:, 0]
        else: # Use one of the true labels
            true_label = literal_eval(df_view.iloc[0].loc["id"])[0] # Using iloc[0] because there's only 1 element in the batch
            batch_prob = gcm.probs[:, true_label]
            batch_label = torch.Tensor([true_label]).to(torch.int64).to(device)
        
        # Generate the heatmaps
        # Grad-CAM
        mapCam = gcm.generateMapClassBatch(batch_label)
        heatmaps = gcm.generateCam(mapCam, layer[0], image_path=None, mergeWithImage=False, isBatch=True)
        # Guided Grad-CAM
        mapGuidedBackProp = gbp.generateMapClassBatch(batch_label)
        gradientNumpy = gradientToImageBatch(mapGuidedBackProp)
        heatmaps_guided = heatmaps * gradientNumpy

        for i in range(batch_size):  # iterate over batch
            # Fetch maps corresponding to current iteration
            grad_cam_hm = heatmaps[i]
            guided_grad_cam_hm = heatmaps_guided[i]
            guided_backprop_hm = gradientNumpy[i]
            
            # Grad-CAM heatmap
            ngc.GradCAM.plot_heatmap(batch_images_unn[i], grad_cam_hm)
            plt.show()
            
            # Guided Grad-CAM heatmap
            plt.imshow(guided_grad_cam_hm, cmap=plt.get_cmap('gray'))
            plt.show()
                
            # Guided Backprop heatmap
            plt.imshow(guided_backprop_hm)
            plt.show()
            
        it += batch_size
        
    # Return a list of hits and misses + max activations
    return np.array(grad_cam_scores), np.array(guided_grad_cam_scores)

df = pd.read_csv("../../../datasets/res2_2510.csv")
plot_heatmaps(getVGGModel(16), layer=['features.29'], df=df)