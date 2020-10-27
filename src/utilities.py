import pickle
import urllib.request as req
import sys, os
from PIL import Image
import torchvision.transforms as transforms
from os.path import abspath

def getImageNetClasses():
    """ Gets a dictionary {class_id: class_name}"""
    url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/' \
          'imagenet1000_clsid_to_human.pkl'
    classes = pickle.load(req.urlopen(url))
    return classes

def translatewnidToClassNames():
    """ Gets a dictionary {wnid: class_id}"""
    path = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\datasets\ILSVRC2012 val\val info\map_clsloc.txt"
    path_to_pickle = r"C:\Users\dumit\Documents\GitHub\DD2412-Project-Grad-CAM\pickles\imagenetDictionary.pickle"
    if os.path.exists(path_to_pickle):
        with open(path_to_pickle, 'rb') as handle:
            final_dictionary = pickle.load(handle)
        return final_dictionary
    else:
        file = open(path, 'r')
        lines = file.readlines()
        dictionary = {}
        for line in lines:
            line_list = line.split(" ")
            dictionary[line_list[0]] = line_list[1]
        import collections
        od = collections.OrderedDict(sorted(dictionary.items()))
        final_dictionary = {}
        i = 0
        for k, v in od.items():
            final_dictionary[k] = i
            i+=1

        with open(path_to_pickle, 'wb') as handle:
            pickle.dump(final_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return final_dictionary

def getImagePIL(image_path, verbose = False):
    """ Works on a SINGLE image, specified by image_path. This code currently works only for the cat_dog experiment.
        Don't use it if you want to load batches of images (instead use the dataloared class).
    """
    # Get image location
    image_folder_path = abspath(image_path)
    
    if verbose:
        print('Image path', image_path)

    image = Image.open(image_path).convert('RGB')
    return image, image_path

def processImage(image):
    """ Useful for processsing a single image. Once more, if you wish to process a batch, use the dataloader class instead."""
    image = image.copy()

    image = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # mean, std of imagenet
            ]
        )(image)  #  apply some transformations
    return image

def gradientToImage(gradient):
    """ Gradients seem to be of the form [batch_size, n_channels, w, h]
        This method transforms it to (w,h,n_channels). It also transforms
        the pixel values in [0,1] (float)"""

    gradientNumpy = gradient[0].cpu().numpy().transpose(1, 2, 0)  #
    for i in range(gradientNumpy.shape[0]):

        gradientNumpy[i] = (gradientNumpy[i] - gradientNumpy[i].min())
        if gradientNumpy[i].max() > 0:
            gradientNumpy[i] = gradientNumpy[i] / gradientNumpy[i].max()

    return gradientNumpy


def gradientToImageBatch(gradient):
    """ Similar to gradientToImage, but works with batches """
    gradientNumpy = gradient.cpu().numpy().transpose(0, 2, 3, 1)  #
    for i in range(gradientNumpy.shape[0]):

        gradientNumpy[i] = (gradientNumpy[i] - gradientNumpy[i].min())
        if gradientNumpy[i].max() > 0:
            gradientNumpy[i] = gradientNumpy[i] / gradientNumpy[i].max()

    return gradientNumpy



def tensorToHeatMap(tensor, rescale = True):
    """ Gradients seem to be of the form [batch_size, n_channels, w, h]
        This method transforms it to (w,h,n_channels). It also transforms
        the pixel values in [0,1] (float)"""
    gradientNumpy = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)  #
    if rescale:
        gradientNumpy = (gradientNumpy - gradientNumpy.min())

        if gradientNumpy.max() > 0:
            gradientNumpy = gradientNumpy/gradientNumpy.max()

    return gradientNumpy


def tensorToHeatMapBatch(tensor, rescale = True):
    """ Same as above, but with batches """
    gradientNumpy = tensor.cpu().detach().numpy().transpose(0, 2, 3, 1)  #
    if rescale:
        for i in range(gradientNumpy.shape[0]):
            gradientNumpy[i] = (gradientNumpy[i] - gradientNumpy[i].min())
            if gradientNumpy[i].max() > 0:
                gradientNumpy[i] = gradientNumpy[i]/gradientNumpy[i].max()


    return gradientNumpy


