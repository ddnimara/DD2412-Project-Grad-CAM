import numpy as np
import torch
from torch.nn import functional as F
from src.models import *
from src.utilities import *
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from os import path
from model import *
from utilities import *
import matplotlib.pyplot as plt
import copy



class Hook:
    def __init__(self, module, backward=False):
        self.module = module
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input.detach()
        self.output = output

    def close(self):
        self.hook.remove()


class gradCAM:

    def __init__(self,model, importantLayer):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.activationHooks = {}  # dictionary layername -> activation
        self.gradientHooks = {}  # layername -> gradient
        self.layers = importantLayer

        self.populateHooks()

    def removeHooks(self):
        for hooks in self.activationHooks.keys():
            self.activationHooks[hooks].close()
        for hooks in self.gradientHooks.keys():
            self.gradientHooks[hooks].close()

    def reset(self):
        """ Reset model (should be applied for every new image to not accumulate gradients) """
        self.removeHooks()
        self.model.zero_grad()
        self.logits = None
        self.probs = None
        self.image = None

    def forward(self,image):
        self.image = image
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)  # dim is [batch_size, classes]


    def populateHooks(self):
        """ Populate Hooks based on image (clip negative gradients on relu modules) """
        for module in self.model.named_modules():  # module is of format [name, module]
            if module[0] in self.layers:
                self.activationHooks[module[0]] = Hook(module[1])
                self.gradientHooks[module[0]] = Hook(module[1],True)

    def getTopK(self,k):
        """ Returns top k classes (based on the output probabilities) """
        return [self.probs.sort(dim=1, descending=True)[1][0][i].item() for i in range(k)]

    def get_one_hot(self, class_label):
        """ Transforms class 'm' to corresponding one hot vector """
        n = self.logits.size(1)  # get number of classes
        one_hot = F.one_hot(class_label, n).double()
        return one_hot

    def backward(self,class_label):
        one_hot = self.get_one_hot(class_label)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)  # dy/dx

    def generateMapK(self,image, k=1):
        """ Work in Progress: (should print heatmap). Returns gradient maps on the 'k' most likely classes """
        #self.reset()
        self.forward(image)
        map = []
        mostLikelyClasses = self.getTopK(k)
        for classes in mostLikelyClasses:
            class_label = torch.tensor(np.array([classes])).type(torch.int64)
            self.backward(class_label)
            map.append([self.activationHooks, self.gradientHooks])  # in guided backprop we want dy/dx so we need the grad of the image
        return map

    def generateMapClass(self, image, classLabel=242):  # 242 -> boxer in imagenet
        """ Work in Progress: (should print heatmap). Returns gradient maps on the specified class """
        self.forward(image)
        map = []
        class_label = torch.tensor(np.array([classLabel])).type(torch.int64)
        self.backward(class_label)
        map.append([self.activationHooks, self.gradientHooks])
        return map

    def generateCam(self, hooks, layer, image, className, alpha = 0.1):
        """ Generates CAMs. """
        # Get activation A_k and the gradients dy/dA_k
        activation = hooks[0][layer].output
        gradient = hooks[1][layer].output[0]
        # compute a_k coefficient by performing a avg pool operation on the gradients
        a_k = F.adaptive_avg_pool2d(gradient,1)

        # take the sum and pass it through the relu
        cam = (activation * a_k).sum(dim = 1, keepdim=True)  # we want [batch, channels, h, w] so we need to keep the channel dim
        cam = F.relu(cam)

        #interpolate to original image dimensions
        cam = F.interpolate(cam, self.image.shape[2:], mode = 'bilinear', align_corners=False)

        # convert to numpy so we can plot it
        numpyCam = tensorToHeatMap(cam)

        # make everything between (0,1) to plot as image
        originalImage = np.array(image).astype(np.float64)/255

        # combine heatmap + original map
        final = alpha*originalImage + (1-alpha)*numpyCam
        plt.imshow(final)
        plt.title(className)
        plt.show()

def gradCamTest(k = 1):
    model = getResNetModel(152)
    model.eval()
    layerList=['layer4.2.conv3']
    gm = gradCAM(model,layerList)
    url = "cat_dog.png"
    imageOriginal = getImagePIL(url)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    map = gm.generateMapK(image,k)
    imagenetClasses = getImageNetClasses()
    topk = gm.getTopK(k)
    for i in range(len(map)):
        for layers in layerList:
            className = imagenetClasses[topk[i]]
            gm.generateCam(map[i],layers, imageOriginal,className)
