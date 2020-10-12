from src.models import *
from src.methods.gradCAM import *
from src.methods.guided_backprop import *
from src.utilities import *
import matplotlib.pyplot as plt
from os import path
import cv2
import numpy as np
import torch

def get_bounding_box(cl):
    """
    Format: (xmin, ymin, xmax, ymax)
    """
    # TODO get bounding box from the corresponding annotation file. Format depends on dataset
    # return (48, 240, 195, 371)
    return (48*(224/353), 240*(224/500), 195*(224/353), 371*(224/500))
    
# Measuring the localization accuracy of Grad-CAM in the context of the Pointing Game on the VOC 2007 test and
# the MS COCO val datasets, using VGG-16 and GoogLeNet
def get_hit_or_miss(image_path, model_path=None, model_type='vgg', k=1, layer_list = ['features.30']):
    pretrained = model_path is None
    if model_type == 'vgg':
        model = getVGGModel(pretrained=pretrained)
    elif model_type == 'google':
        model = getGoogleModel(pretrained=pretrained)
        
    if not pretrained:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    gm = gradCAM(model, layer_list)
    imageOriginal = getImagePIL(image_path)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gm.forward(image)
    imagenetClasses = getImageNetClasses()
    topk = gm.getTopK(k)
    
    hits_and_misses = []
    max_acts = []
    
    for cl in topk:
        map = gm.generateMapClass(cl)
        for layers in layer_list:
            className = imagenetClasses[cl]
            heatmap = gm.generateCam(map, layers, imageOriginal, className)
            gray = rgb2gray(heatmap)
            max_ind = np.argwhere(gray.max() == gray)
            print('Class: ', className, ', Maximal point: ', gray.max(), ', index: ', max_ind)

            # Check whether the maximally activated point is within the relevant bounding box
            bb = get_bounding_box(cl)
            print(bb)
            
            x_min, y_min, x_max, y_max = bb
            x, y = max_ind[0][1], max_ind[0][0]
            if x >= x_min and x <= x_max and y >= y_min and y <= y_max:
                hits_and_misses.append(True)
            else:
                hits_and_misses.append(False)
            max_acts.append(gray.max())
    
    # Return a list of hits and misses + max activations
    return hits_and_misses, max_acts
    
def get_localization_accuracy(image_path_list, model_path=None):
    # TODO Handle images differently?
    hit = 0
    miss = 0
    for image_path in image_path_list:
        hits_and_misses, _ = get_hit_or_miss(image_path, model_path, k=1)
        for is_hit in hits_and_misses:
            if (is_hit):
                hit += 1
            else:
                miss += 1
    return hit / (hit + miss)

def get_localization_recall(image_path_list, model_path=None, threshold=0.5):
    # TODO Handle images differently?
    hit = 0
    miss = 0
    for image_path in image_path_list:
        
        is_below_threshold = True
        hits_and_misses, max_acts = get_hit_or_miss(image_path, model_path, k=5)
        for is_hit, max_act in zip(hits_and_misses, max_acts):
            if is_hit and max_act >= threshold or not is_hit and max_act < threshold:
                hit += 1
            else:
                miss += 1
    return hit / (hit + miss)

image_path = path.abspath("../../../images/000001.jpg")
model_path = path.abspath("../../../models/vgg16_voc2007.pth")
hits_and_misses, max_acts = get_hit_or_miss(image_path, model_path=None, k=1)
print(hits_and_misses, max_acts)

image_path_list = [image_path]
accuracy = get_localization_accuracy(image_path_list)
print(accuracy)

recall = get_localization_recall(image_path_list)
print(recall)