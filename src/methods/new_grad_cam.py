import numpy as np
import torch
from src.utilities import *
import cv2
from torch.nn import functional as F
from matplotlib import pyplot as plt
class Hook:
    def __init__(self, module, is_backward=False):
        self.module = module

        if is_backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)
            
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

class gradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.model.eval()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer_name = layer_name
        
        self.init_hooks()

    def init_hooks(self):
        """ Populate Hooks based on image (clip negative gradients on relu modules) """
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self.activation_hook = Hook(module)
                self.gradient_hook = Hook(module, is_backward = True)
                return
    
        raise Exception(f"Layer '{self.layer_name}' was not found in the model!")

    def reset(self):
        """ Reset model (should be applied for every new image to not accumulate gradients) """
        self.remove_hooks()
        self.model.zero_grad()
        self.logits = None
        self.image = None
        self.init_hooks()
    
    def remove_hooks(self):
        self.activation_hook.close()
        self.gradient_hook.close()

    def generate_heatmap(self, image, class_label, is_counterfactual=False):
        self.reset()
        if isinstance(class_label, int):
            class_label = torch.Tensor([class_label]).type(torch.int64)
            
        self.logits = self.model.forward(image)

        # TODO (RN): what if class_label is not one-hot? 
        #       check if it can happen (i.e. if any images have different classes present)
        self.model.zero_grad()
        self.logits.backward(gradient = self.get_one_hot(class_label),
                             retain_graph = True)
        
        activations = self.activation_hook.output
        gradients = self.gradient_hook.output[0]
        
        if is_counterfactual:
            gradients = - gradients
        
        weights = F.adaptive_avg_pool2d(gradients, 1)
        
        heatmap = F.relu((activations * weights).sum(dim = 1, keepdim = True))
        # Upscale the heatmap to match image size
        heatmap = F.interpolate(heatmap, image.shape[2:], mode='bilinear', align_corners = False)
        heatmap_array = heatmap.squeeze(dim=0).cpu().detach().numpy().transpose(1, 2, 0)
        
        return heatmap_array
    
    def get_one_hot(self, class_label):
        """ Transforms class 'm' to corresponding one hot vector """
        n = self.logits.size(1)  # get number of classes
        one_hot = F.one_hot(class_label, n).double()
        return one_hot
    
    @staticmethod
    def plot_heatmap(image, heatmap, axis):
        # reformat it to represent an image. 
        # also adjust it's colours (to be the same as in the paper)
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        image = torch.clamp(image, 0,1)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        combined_image = cv2.addWeighted(np.uint8(255 * image.detach().squeeze(dim=0).numpy().transpose(1,2,0)), 0.3, heatmap, 0.7, 0)
        
        axis.imshow(combined_image)