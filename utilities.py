import pickle
import urllib.request as req
import sys, os
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F

def getImageNetClasses():
    url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/' \
          'imagenet1000_clsid_to_human.pkl'
    classes = pickle.load(req.urlopen(url))
    return classes

def getImagePIL(path, verbose = False):
    pathname = os.path.dirname(sys.path[0])
    imagepath = os.path.join(pathname,"Images",path)

    if verbose:
        print('Image path', imagepath)

    image = Image.open(imagepath).convert('RGB')
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

    gradientNumpy = 255.0 * (gradientNumpy - gradientNumpy.min())/gradientNumpy.max()
    if verbose:
        print('gradient size', gradientNumpy.shape)
        print('gradient values', gradientNumpy)

    return gradientNumpy
def evaluate(url, model):
    model.eval()
    imageOriginal = getImagePIL(url)
    dictionary = getImageNetClasses()
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    out = model(image)
    print(out.shape)
    probs = F.softmax(out, dim=1)
    predictions, indeces = probs.sort(dim = 1, descending=True)
    results = [(predictions[0][i].item()*100,dictionary[indeces[0][i].item()]) for i in range(5)]
    print('most likely classes', results)