import torch
from torch.functional import F
import matplotlib.pyplot as plt
from src.utilities import getImageNetClasses

class FastGradientSignMethod:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        # Detach all the model parameters since we won't train the network
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.imagenet_classes = getImageNetClasses()
        
    def generate_attack(self, image, target, epsilon = 0.03, max_iters = 20, 
                        visualize = True, verbose = True):
        """
        Attempt to create a perturbed version of the given image using the 
        Fast Gradient Sign Method in order to confuse the network. 
        
        Args:
            image:      [1 x C x H x W] image
            target:     [1 x N_class] ground truth label
            epsilon:    float multiplier of the noise, controlling its severity
            max_iters:  maximum number of FGSM iterations
            visualize:  if True, plot the original image, the noise and the 
                        perturbed image at the end
        
        Returns:  
            the perturbed image (if the attack was succesful),
            None                (otherwise)    
        """
        original_image = image
        
        if (target[0] != 0).sum() > 1:
            print("ERROR: Adversarial attacks are not yet implemented for multi-labeled examples!")
            return None

        # We will need the gradients of the image
        image.requires_grad_()
        output = self.model.forward(image)
        
        true_label = int(torch.argmax(target[0]))
        predicted_label = int(torch.argmax(output[0]))
        
        if predicted_label != true_label:
            print("Image is incorrectly classified, skipping adversarial attack...")
            return None
        
        iters = 0
        while predicted_label == true_label and iters <= max_iters:
            iters += 1
            # 1. Calculate gradients from the previous forward pass
            self.model.zero_grad()
            loss = F.binary_cross_entropy_with_logits(output, target)
            loss.backward()
            image_gradients = image.grad.data
                        
            # 2. Perturb the image with an adversarial noise
            
            # We have to detach the image, otherwise the following + operation in image+noise
            # would be tracked by the autograd (which disables gradient computations)
            image = image.detach()
            noise = epsilon * torch.sign(image_gradients)
            image = image + noise
            
            # 3. Feed the perturbed image to the network
            # Here we reattach the image tensor to the computational graph
            output = self.model.forward(image.requires_grad_())
            old_prediction = predicted_label
            predicted_label = int(torch.argmax(output[0]))
            
            # We can print the probability of the true class to keep track of the progress
            if verbose:
                true_label_prob = F.softmax(output[0], dim=0)[true_label]
                print("{0:.0%}".format(true_label_prob), end=" ", flush=True)
       
        # 4. Visualize results
        if visualize:
            self.visualize(original_image, noise, image, old_prediction, predicted_label)
        
        return image
    
    def visualize(self, image, noise, perturbed_image, old_prediction, new_prediction):
        fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3)
        
        ax_1.imshow(image.squeeze().detach().numpy().transpose(1,2,0))
        ax_1.set_title(self.imagenet_classes[old_prediction])
        # Transform the noise from [-1,1] to [0,1]
        ax_2.imshow((noise * 0.5 + 0.5).squeeze().numpy().transpose(1,2,0))
        
        ax_3.imshow(perturbed_image.squeeze().detach().numpy().transpose(1,2,0))
        ax_3.set_title(self.imagenet_classes[new_prediction])
        
        plt.show()