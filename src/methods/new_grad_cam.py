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

class GradCAM:
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
        self.init_hooks()
    
    def remove_hooks(self):
        self.activation_hook.close()
        self.gradient_hook.close()

    def generate_heatmaps(self, image_batch, class_label_batch, is_counterfactual=False):
        self.reset()
        self.logits = self.model.forward(image_batch)
        self.model.zero_grad()
        self.logits.backward(gradient = self.get_one_hot(class_label_batch),
                             retain_graph = True)
        
        activations = self.activation_hook.output
        gradients = self.gradient_hook.output[0]
 
        if is_counterfactual:
            gradients = - gradients
        
        weights = F.adaptive_avg_pool2d(gradients, 1)
        heatmaps = F.relu((activations * weights).sum(dim = 1, keepdim = True))
        # Upscale the heatmaps to match image size
        heatmaps = heatmaps - heatmaps.min()
        heatmaps = heatmaps / heatmaps.max()
        heatmaps = F.interpolate(heatmaps, image_batch.shape[2:], mode='bicubic', align_corners = False)
        heatmaps = heatmaps.detach().cpu().numpy().transpose(0,2,3,1)
        return heatmaps
    
    def get_one_hot(self, class_label_batch):
        """ Transforms class 'm' to corresponding one hot vector """
        n = self.logits.size(1)  # get number of classes
        one_hot = F.one_hot(class_label_batch, n).float()
        return one_hot
    
    @staticmethod
    def plot_heatmap(image, heatmap, axis=None, ratio=(0.3, 0.7), save_file=None):
        # reformat it to represent an image. 
        # also adjust it's colours (to be the same as in the paper)
        heatmap = heatmap - heatmap.min()
        max_value = heatmap.max()
        if max_value > 0:
            heatmap = heatmap / max_value
        image = torch.clamp(image, 0,1)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        combined_image = cv2.addWeighted(np.uint8(255 * image.detach().squeeze(dim=0).cpu().numpy().transpose(1,2,0)), ratio[0], heatmap, ratio[1], 0)
        
        if save_file is not None:
            plt.imsave(save_file, combined_image)
        elif axis is not None:
            axis.imshow(combined_image)
        else:
            plt.imshow(combined_image)

class GradCAM_plusplus(GradCAM):
    def __init__(self, model, layer_name):
        super().__init__(model, layer_name)
        
    def generate_heatmaps(self, image_batch, class_label_batch):
        self.reset()
        self.logits = self.model.forward(image_batch)
        self.model.zero_grad()
        self.logits.backward(gradient = self.get_one_hot(class_label_batch),
                             retain_graph = True)
        
        activations = self.activation_hook.output
        gradients = self.gradient_hook.output[0]
        
        grad2 = gradients.pow(2)
        grad3 = gradients.pow(3)
        activation_sum = activations.sum(dim=(2,3), keepdim=True)
        
        alpha = grad2 / ( 2 * grad2 + activation_sum * grad3 + 1e-7)
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        heatmaps = F.relu(torch.sum(weights * torch.exp(activations), dim=1, keepdim=True))
        
        # Normalize each heatmap in the batch
        for i in range(heatmaps.shape[0]):
            heatmaps[i] -= heatmaps[i].min()
            heatmaps[i] /= heatmaps[i].max()
        
        heatmaps = F.interpolate(heatmaps, image_batch.shape[2:], mode='bicubic', align_corners = False)
        heatmaps = heatmaps.detach().cpu().numpy().transpose(0,2,3,1)  
         
        return heatmaps        