
import numpy as np
import torch
from torch.nn import functional as F
from model import *
from utilities import *
from PIL import Image
import torchvision.transforms.functional as TF
class guidedBackProp:

    def __init__(self,model):
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hooks = []
        #self.populateHooks()

    def removeHooks(self):
        for hooks in self.hooks:
            hooks.remove()

    def reset(self):
        """ Reset model (should be applied for every new image to not accumulate gradients) """
        self.removeHooks()
        self.model.zero_grad()
        self.logits = None
        self.probs = None
        self.image = None

    def forward(self,image):
        self.image = image.requires_grad_()
        self.logits = self.model(image)
        print('logit size', self.logits.size())
        self.probs = F.softmax(self.logits, dim=1)  # dim is [batch_size, classes]
        print('probs size', self.probs.size())


    def populateHooks(self):
        """ Populate Hooks based on image (clip negative gradients on relu modules) """
        for module in self.model.named_modules():  # module is of format [name, module]
            self.hooks.append(module[1].register_backward_hook(lambda module, gradInput, gradOutput: (F.relu(gradInput[0]),)
            if isinstance(module,torch.nn.ReLU) else None))

    def print(self):
        """ Get basic info """
        print("Model", self.model)
        print("Hooks", self.hooks)
        print("Probabiltiies descending", self.probs.sort(dim=1, descending=True))  # returns list [,] with [0] -> probs
                                                                                    # and [1] -> corresponding indeces
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
        self.reset()
        self.forward(image)
        map = []
        mostLikelyClasses = self.getTopK(k)
        for classes in mostLikelyClasses:
            class_label = torch.tensor(np.array([classes])).type(torch.int64)
            self.backward(class_label)
            map.append(self.image.grad)  # in guided backprop we want dy/dx so we need the grad of the image
        print('gradient', map[0].size())
        return map


    def generateMapClass(self,image, classLabel = 242):  # 242 -> boxer in imagenet
        """ Work in Progress: (should print heatmap). Returns gradient maps on the specified class """
        self.reset()
        self.forward(image)
        map = []
        class_label = torch.tensor(np.array([classLabel])).type(torch.int64)
        self.backward(class_label)
        map.append(self.image.grad)  # in guided backprop we want dy/dx so we need the grad of the image
        print('gradient', map[0].size())
        return map
# test

def test(k = 1):
    model = getResNetModel(152)
    gbb = guidedBackProp(model)
    url = "cat_dog.png"
    imageOriginal = getImagePIL(url)
    image = processImage(imageOriginal)
    image = image.unsqueeze(0)  # add 'batch' dimension
    gbb.forward(image)
    best = gbb.getTopK(10)
    dictionary = getImageNetClasses()
    # print('best predictions', [dictionary[i] for i in best])
    map = gbb.generateMapClass(image)
    for heatmap in map:
        gradientNumpy = gradientToImage(heatmap)
        hm = Image.fromarray(np.uint8(gradientNumpy))
        # print('hm size',hm.size)
        hm.show()
        res = Image.blend(imageOriginal, hm, 0.6)
        res.show()
test()


