from src.utilities import getImageNetClasses
from src.methods.new_grad_cam import GradCAM

import torch
from torch.functional import F

import matplotlib.pyplot as plt

def get_label(label_batch):
    return int(torch.argmax(label_batch[0]))

def get_confidence(output, label):
    return F.softmax(output, dim=1)[0,label]

class FastGradientSignMethod:
    def __init__(self, model):                
        self.model = model
        self.model.eval()
        # Detach all the model parameters since we won't train the network
        for p in self.model.parameters():
            p.requires_grad = False

        self.imagenet_classes = getImageNetClasses()   

    def generate_attack(self, unnormalized_image, normalized_image, 
                        target, epsilon = 0.07):     
        if (target[0] != 0).sum() > 1:
            print("ERROR: Adversarial attacks are not yet implemented for multi-labeled examples!")
            return None
                
        normalized_image.requires_grad_()        
        output = self.model.forward(normalized_image)
        
        true_label = get_label(target)
        predicted_label = get_label(output)
        
        if predicted_label != true_label:
            print("Image is incorrectly classified, skipping adversarial attack...")
            return None, None, None
        
        original_confidence = get_confidence(output, true_label)

        self.model.zero_grad()
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        image_gradients = normalized_image.grad.data
        
        noise = torch.sign(image_gradients)
        perturbed_image = normalized_image + epsilon * noise
        
        output = self.model.forward(perturbed_image)
        new_prediction = get_label(output)
        new_confidence = get_confidence(output, new_prediction)
        
        if new_prediction == true_label:
            print("Attack failed.")
            return None, None, None
        
        print("Succesful attack!")
        axes = self.visualize(unnormalized_image, noise, epsilon,
                             true_label, original_confidence,
                             new_prediction, new_confidence)
            
        return perturbed_image, new_prediction, axes
    
    def visualize(self, unnormalized_image, noise, epsilon,
                  original_prediction, original_confidence,
                  new_prediction, new_confidence):

        fig, axes = plt.subplots(2,3)
        for axis in axes.flatten():
            axis.axis("off")
            
        # 1. Original image
        title = "{} ({:.2f})".format(self.imagenet_classes[original_prediction],
                                     original_confidence)
        axes[0, 0].set_title(title)
        axes[0, 0].imshow(unnormalized_image.squeeze().detach().numpy().transpose(1,2,0))
        
        # 2. Noise
        axes[0, 1].set_title("Adversarial noise")
        # Transform the noise from [-1,1] to [0,1]
        axes[0, 1].imshow((noise * 0.5 + 0.5).squeeze().numpy().transpose(1,2,0))
        
        # 3. Perturbed image 
        title = "{} ({:.2f})".format(self.imagenet_classes[new_prediction],
                                     new_confidence)
        axes[0, 2].set_title(title)
        perturbed_image = unnormalized_image + epsilon * noise
        axes[0, 2].imshow(torch.clamp(perturbed_image, 0, 1).squeeze().detach().numpy().transpose(1,2,0))
        
        return axes