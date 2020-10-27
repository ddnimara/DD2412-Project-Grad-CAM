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

# Evaluating faithfulness via image occlusion by calculating the rank correlation of patches that change the CNN
# score and patches that have high heatmap values for Grad-CAM and Guided Grad-CAM on the ILSVRC-12 val
# dataset. Rank correlation should be averaged over 2510 images.

def calculate_rank_correlation(model, df, layer, plot=False, use_pred=False):
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
    # Run Guided Backprop
    gbp = guidedBackProp(model)
    
    it = 0  # we will use to keep track of the batch location
    grad_cam_scores = []
    guided_grad_cam_scores = []
    
    # Loop through batches
    for batch_images in tqdm(validationLoader):
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
            
        # Generate the occlusion maps
        with torch.no_grad():
            occlusion_maps = occlusion(model, batch_images, batch_label, batch_prob, invert=True)

        for i in range(batch_size):  # iterate over batch
            # Fetch maps corresponding to current iteration
            grad_cam_hm = heatmaps[i]
            guided_grad_cam_hm = heatmaps_guided[i]
            occlusion_map = occlusion_maps[i]
                
            # Convert heatmap to grayscale image (Guided Grad-CAM)
            guided_grad_cam_hm = np.dot(guided_grad_cam_hm[..., :3], [0.2989, 0.5870, 0.1140])
            
            # Normalize the occlusion map
            occlusion_map = occlusion_map - occlusion_map.min()
            occlusion_map = occlusion_map / occlusion_map.max()
            
            # Plot images
            if plot:
                # Grad-CAM heatmap
                ngc.gradCAM.plot_heatmap(batch_images[i], grad_cam_hm)
                plt.show()
                
                # Guided Grad-CAM heatmap
                plt.imshow(guided_grad_cam_hm, cmap=plt.get_cmap('gray'))
                plt.colorbar()
                plt.show()
                
                # Occlusion map
                plt.imshow(occlusion_map, cmap=plt.cm.bwr_r, vmin=0, vmax=1)
                plt.colorbar()
                plt.show()
            
            # Downsample the maps
            new_size = (14, 14)
            grad_cam_hm = resize(grad_cam_hm, new_size)
            guided_grad_cam_hm = resize(guided_grad_cam_hm, new_size)
            occlusion_map = resize(occlusion_map, new_size)

            # Calculate rank correlation between each heatmap and the occlusion map
            # Get ranks
            grad_cam_hm_rank = get_rank(grad_cam_hm)
            guided_grad_cam_hm_rank = get_rank(guided_grad_cam_hm)
            occlusion_map_rank = get_rank(occlusion_map)
            
            # Grad-CAM heatmap
            grad_cam_coeff, grad_cam_p = spearmanr(grad_cam_hm_rank, occlusion_map_rank)
            grad_cam_scores.append(grad_cam_coeff)
            
            # Guided Grad-CAM heatmap
            guided_grad_cam_coeff, guided_grad_cam_p = spearmanr(guided_grad_cam_hm_rank, occlusion_map_rank)
            guided_grad_cam_scores.append(guided_grad_cam_coeff)
            print(grad_cam_coeff, guided_grad_cam_coeff)
            
        it += batch_size
        
    # Return a list of hits and misses + max activations
    return np.array(grad_cam_scores), np.array(guided_grad_cam_scores)

def get_rank(map):
    return np.argsort(map.flatten())
    
def occlusion(model, images, label, prob, invert=True):
    original_images = images.clone()
    
    nr, ch, h, w = images.shape
    block_h = 45 # Path height
    block_w = 45 # patch width
    mean = 0.5 # Color used for padding and filling the patches (should be gray by default)
    batch_s = 40 # Number of modified images to collect for forward pass
    
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
                # If the desired number of images have been collected or this is the last iteration
                if (it + 1) % batch_s == 0 or i == h - 1 and j == w - 1:
                    # Forward pass
                    images_to_process = torch.cat(images_to_process, dim=0)
                    
                    # Collect prediction
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

df = pd.read_csv("../../../datasets/res2_120.csv")
print("Running Grad-CAM on VGG16 and calculating rank correlation with occlusion maps...")
grad_cam_scores, guided_grad_cam_scores = calculate_rank_correlation(getVGGModel(16), layer=['features.29'], df=df, plot=False, use_pred=False)
# np.savetxt("result_gradcam.csv", grad_cam_scores, delimiter=",")
# np.savetxt("result_guided_gradcam.csv", guided_grad_cam_scores, delimiter=",")
# print("[Grad-CAM] Average score: ", np.average(grad_cam_scores))
# print("[Guided Grad-CAM] Average score: ", np.average(guided_grad_cam_scores))
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# path = os.path.abspath("../../../datasets/ILSVRC2012 val/resized")
# print(path)
# paths = df['path']
#
# for p in paths:
#     print(os.path.join(path,p.split("\\")[-1]))


