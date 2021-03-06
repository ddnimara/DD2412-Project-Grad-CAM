import numpy as np
import torch
from src.utilities import *
import cv2
from torch.nn import functional as F

class Hook:
    def __init__(self, module, backward=False):
        self.module = module
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
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
        """ Returns top k probabilities and classes, respectively """
        return self.probs.topk(k, dim = 1)

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
        _, mostLikelyClasses = self.getTopK(k)
        mostLikelyClasses = mostLikelyClasses[0]
        
        for classes in mostLikelyClasses:
            class_label = torch.tensor(np.array([int(classes)])).type(torch.int64)
            self.backward(class_label)
            map.append([self.activationHooks, self.gradientHooks])  # in guided backprop we want dy/dx so we need the grad of the image
        return map

    def generateMapClass(self, classLabel=242):  # 242 -> boxer in imagenet
        """ Work in Progress: (should print heatmap). Returns gradient maps on the specified class """
        class_label = torch.tensor(np.array([classLabel])).type(torch.int64)#.to(self.device)
        self.backward(class_label)
        map = [self.activationHooks, self.gradientHooks]
        return map

    def generateMapClassBatch(self, classLabels):  # 242 -> boxer in imagenet
        """ Work in Progress: (should print heatmap). Returns gradient maps on the specified class """
        self.backward(classLabels)
        map = [self.activationHooks, self.gradientHooks]
        return map

    def generateCam(self, hooks, layer, image_path, mergeWithImage = True, counterFactual = False, isBatch = False, rescale = True, clip = False):
        """ Generates CAMs. """
        # Get activation A_k and the gradients dy/dA_k
        activation = hooks[0][layer].output
        gradient = hooks[1][layer].output[0]
        if counterFactual:
            gradient = - gradient
        # compute a_k coefficient by performing a avg pool operation on the gradients
        a_k = F.adaptive_avg_pool2d(gradient, 1)

        # take the sum and pass it through the relu
        cam = (activation * a_k).sum(dim = 1, keepdim=True)  # we want [batch, channels, h, w] so we need to keep the channel dim
        cam = F.relu(cam)
        #interpolate to original image dimensions
        cam = F.interpolate(cam, self.image.shape[2:], mode = 'bilinear', align_corners=False)

        # convert to numpy so
        # numpyCam = tensorToHeatMap(cam)
        if isBatch:
            numpyCam = tensorToHeatMapBatch(cam, rescale= rescale)
        else:
            numpyCam = tensorToHeatMap(cam, rescale= rescale)
        if mergeWithImage:  # Normal grad-cam wants to impose the heatmap over the image
            # reformat it to represent an image. Also adjust it's colours (to be the same as in the paper)

            heatmap = cv2.applyColorMap(np.uint8(255 * numpyCam), cv2.COLORMAP_JET)

            # get original image via path
            originalImage = cv2.imread(image_path,1)

            # combine them
            finalImage = cv2.addWeighted(heatmap, 0.7, originalImage, 0.3, 0)
            print("im",finalImage.shape)
            if clip:
                indeces = numpyCam[:,:,0] < 0.20
                finalImage[indeces,:] = originalImage[indeces,:]
            # make it rgb (cv2 by default is bgr for some reason)
            finalImage = cv2.cvtColor(finalImage, cv2.COLOR_BGR2RGB)
        else:  # guided gradcam simply wants the heatmap
            finalImage = numpyCam
        return finalImage


