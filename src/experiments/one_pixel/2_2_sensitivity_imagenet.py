from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
from src.dataSetLoader import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ast import literal_eval
import src.methods.new_grad_cam as ngc

def perturb_image(xs, img):
    if xs.ndim < 2:
            xs = np.array([xs])
    batch = len(xs)
    imgs = img.repeat(batch, 1, 1, 1)
    xs = xs.astype(int)

    count = 0
    for x in xs:
            pixels = np.split(x, len(x)/5)
            for pixel in pixels:
                    x_pos, y_pos, r, g, b = pixel
                    imgs[count, 0, x_pos, y_pos] = (r / 255.0 - mean[0])/std[0]
                    imgs[count, 1, x_pos, y_pos] = (g / 255.0 - mean[1])/std[1]
                    imgs[count, 2, x_pos, y_pos] = (b / 255.0 - mean[2])/std[2]
            count += 1

    return imgs

def sensitivity(model, layer, results_file, plot=False):
    # Get ImageNet classes
    imagenetClasses = getImageNetClasses()

    # Initialize model
    model.eval()

    # Load data
    df = pd.read_csv("../../../datasets/res2_2510.csv")
    results = pd.read_csv(results_file)
    validationSetOriginal = dataSetILS(df)
    validationSet = dataSetILS(df, transforms)
    
    heatmap_activations_before = []
    heatmap_activations_after = []
    
    for index, row in results.iterrows():
        ind = row['index']
        arr = np.array(literal_eval(row['value']))
        print("Element #", ind)
       
        imageOriginal = validationSetOriginal[ind]
        image = validationSet[ind]

        # Grad-CAM
        gcm = gradCAM(model, layer)
        gcm.forward(image.unsqueeze(0))
        probs, labels = gcm.getTopK(k=3)
        probs = probs[0]
        labels = labels[0]

        # Print original probabilities and labels
        print("Original predictions:")
        for prob, label in zip(probs, labels):
            print(float(prob), imagenetClasses[int(label)])
        print()
        # Plot heatmap
        imageOriginalTensor = to_tensor_transform(imageOriginal)
        map = gcm.generateMapClass(labels[0])
        heatmap = gcm.generateCam(map, layer[0], image_path=None, isBatch=False, rescale=False, mergeWithImage=False)
        
        heatmap_activation = heatmap[arr[0], arr[1]]
        print("Relevant pixel in heatmap (before):", heatmap_activation)
        heatmap_activations_before.append(heatmap_activation)
        print()
        
        if plot:
            ngc.GradCAM.plot_heatmap(imageOriginalTensor, heatmap, ratio=(0.5, 0.5))
            plt.title(imagenetClasses[int(labels[0])])
            plt.show()

        #########################################################################

        # Perturb original image
        im_array = np.array(imageOriginal)
        im_array[arr[0], arr[1], 0] = arr[2]
        im_array[arr[0], arr[1], 1] = arr[3]
        im_array[arr[0], arr[1], 2] = arr[4]
    
        # Plot new image
        if plot:
            plt.imshow(im_array)
            plt.scatter(arr[1], arr[0], s=80, facecolors='none', edgecolors='r')
            plt.show()

        # Perturb image
        image = perturb_image(arr, image)

        # Predict
        gcm.forward(image)
        probs, labels = gcm.getTopK(k=3)
        probs = probs[0]
        labels = labels[0]

        # Print top probabilities and labels
        print("New predictions:")
        for prob, label in zip(probs, labels):
            print(float(prob), imagenetClasses[int(label)])
        print()

        # Plot heatmap
        im_array = to_tensor_transform(im_array)
        map = gcm.generateMapClass(labels[0])
        heatmap = gcm.generateCam(map, layer[0], image_path=None, isBatch=False, rescale=False, mergeWithImage=False)

        heatmap_activation = heatmap[arr[0], arr[1]]
        print("Relevant pixel in heatmap (after):", heatmap_activation)
        heatmap_activations_after.append(heatmap_activation)
        print()
        
        if plot:
            ngc.GradCAM.plot_heatmap(im_array, heatmap, ratio=(0.5, 0.5))
            plt.title(imagenetClasses[int(labels[0])])
            plt.show()
            
        print("---------------------------------------------------")
    
    return np.array(heatmap_activations_before), np.array(heatmap_activations_after)
    
# VGG-16
heatmap_activations_before, heatmap_activations_after = sensitivity(model=getVGGModel(16), layer=['features.29'], results_file="results_VGG16.csv", plot=False)
print("Heatmap activations (before):")
print("Average:", np.average(heatmap_activations_before))
print("Standard deviation:", np.std(heatmap_activations_before))
print("Minimum:", np.min(heatmap_activations_before))
print("Maximum:", np.max(heatmap_activations_before))
print()
print("Heatmap activations (after):")
print("Average:", np.average(heatmap_activations_after))
print("Standard deviation:", np.std(heatmap_activations_after))
print("Minimum:", np.min(heatmap_activations_after))
print("Maximum:", np.max(heatmap_activations_after))

# GoogLeNet
# heatmap_activations_before, heatmap_activations_after = sensitivity(model=getGoogleModel(), layer=['inception5b'], results_file="results_GoogLeNet.csv", plot=False)
# print("Heatmap activations (before):")
# print("Average:", np.average(heatmap_activations_before))
# print("Standard deviation:", np.std(heatmap_activations_before))
# print("Minimum:", np.min(heatmap_activations_before))
# print("Maximum:", np.max(heatmap_activations_before))
# print()
# print("Heatmap activations (after):")
# print("Average:", np.average(heatmap_activations_after))
# print("Standard deviation:", np.std(heatmap_activations_after))
# print("Minimum:", np.min(heatmap_activations_after))
# print("Maximum:", np.max(heatmap_activations_after))