import src.models as models
from src.datasets import ResizedImagenetDataset
from src.methods.adversarial_attacks import FastGradientSignMethod
from src.methods.new_grad_cam import gradCAM

from torch.utils.data.dataloader import DataLoader
from numpy.random import randint
import torch
from matplotlib import pyplot as plt
# Resized ImageNet _validation_ dataset
imagenet_csv_path = "../../../datasets/imagenet/resized.csv"

dataset = ResizedImagenetDataset(imagenet_csv_path)
unnormalized_dataset = ResizedImagenetDataset(imagenet_csv_path)

model = models.getVGGModel(16)
attacker = FastGradientSignMethod(model)

print("Generating adversarial attacks for VGG-16...")
while True:    
    random_idx = 25974#randint(0,50000)
    print(f"Image {random_idx}", end="\t")
    image, label = dataset[random_idx]
    # Unnormalized image is only for visualization (it looks much better)
    unnormalized_image, _ = unnormalized_dataset[random_idx]
    
    perturbed_image, new_prediction, axes = attacker.generate_attack(unnormalized_image, image, label)
    
    if perturbed_image is None:
        continue
    
    grad_cam = gradCAM(model, 'features.29')
    
    true_class = int(torch.argmax(label[0]))
    
    # 1. Grad-CAM on original image for true class
    heatmap = grad_cam.generate_heatmap(image, true_class)
    gradCAM.plot_heatmap(image, heatmap, axes[1,0])
    # 2. Grad-CAM on perturbed image for true class
    heatmap = grad_cam.generate_heatmap(perturbed_image, true_class)
    gradCAM.plot_heatmap(perturbed_image, heatmap, axes[1,1])
    # 2. Grad-CAM on perturbed image for predicted class
    heatmap = grad_cam.generate_heatmap(perturbed_image, new_prediction)
    gradCAM.plot_heatmap(perturbed_image, heatmap, axes[1,2])
    
    plt.show()
    