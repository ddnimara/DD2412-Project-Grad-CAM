import pickle
import urllib.request as req
import sys, os
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
from os.path import abspath
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import minmax_scale as minmax
def getImageNetClasses():
    url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/' \
          'imagenet1000_clsid_to_human.pkl'
    classes = pickle.load(req.urlopen(url))
    return classes

def getImagePIL(image_path, verbose = False):
    # Get image location
    image_folder_path = abspath(image_path)
    
    if verbose:
        print('Image path', image_path)

    image = Image.open(image_path).convert('RGB')
    return image

def processImage(image):
    image = image.copy()
    image = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # mean, std of imagenet
            ]
        )(image)  #  apply some transformations
    return image

def gradientToImage(gradient, verbose = False):
    """ Gradients seem to be of the form [batch_size, n_channels, w, h]
        This method transforms it to (w,h,n_channels). It also transforms
        the pixel values in [0,255]"""

    gradientNumpy = gradient[0].numpy().transpose(1, 2, 0)  #
    gradientNumpy = (gradientNumpy - gradientNumpy.min())
    gradientNumpy = gradientNumpy/gradientNumpy.max()
    #gradientNumpy*= 255.0
    if verbose:
        print('gradient size', gradientNumpy.shape)
        print('gradient values', gradientNumpy)

    return gradientNumpy

def tensorToHeatMap(tensor, verbose = False):
    gradientNumpy = tensor[0].detach().numpy().transpose(1, 2, 0)  #
    gradientNumpy = (gradientNumpy - gradientNumpy.min())
    gradientNumpy = gradientNumpy/gradientNumpy.max()
    if verbose:
        print('gradient size', gradientNumpy.shape)
        print('gradient values', gradientNumpy)
    return gradientNumpy


def evaluate(path, model):
    model.eval()
    imageOriginal = getImagePIL(path)
    dictionary = getImageNetClasses()
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    out = model(image)
    print(out.shape)
    probs = F.softmax(out, dim=1)
    predictions, indeces = probs.sort(dim = 1, descending=True)
    results = [(predictions[0][i].item()*100,dictionary[indeces[0][i].item()]) for i in range(5)]
    print('most likely classes', results)
