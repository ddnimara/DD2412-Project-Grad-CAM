import src.models as models
from src.datasets import ResizedImagenetDataset
from src.methods.adversarial_attacks import FastGradientSignMethod
from src.methods.new_grad_cam import GradCAM

from os import makedirs, removedirs

from torch.utils.data.dataloader import DataLoader
from numpy.random import randint
import torch
from matplotlib import pyplot as plt
# Resized ImageNet _validation_ dataset
imagenet_csv_path = "../../../datasets/imagenet/new/resized.csv"

dataset = ResizedImagenetDataset(imagenet_csv_path)
unnormalized_dataset = ResizedImagenetDataset(imagenet_csv_path)

model = models.getVGGModel(16)
attacker = FastGradientSignMethod(model)

print("Generating adversarial attacks for VGG-16...")
while True:    
    random_idx = randint(0,50000)
    print(f"Image {random_idx}", end="\t")
    image, label = dataset[random_idx]
    image.unsqueeze_(0)
    label.unsqueeze_(0)
    # Unnormalized image is only for visualization (it looks much better)
    unnormalized_image, _ = unnormalized_dataset[random_idx]
    
    makedirs(f"{random_idx}")    
    perturbed_image, new_prediction, axes = attacker.generate_attack(
        unnormalized_image, image, label, save_dir=f"{random_idx}", epsilon=0.05)
    
    if perturbed_image is None:
        removedirs(f"{random_idx}")
        continue
    
    grad_cam = GradCAM(model, 'features.29')
    
    true_class = torch.argmax(label[0]).unsqueeze_(0)
    
    # 1. Grad-CAM on original image for true class
    heatmap = grad_cam.generate_heatmaps(image, true_class)
    GradCAM.plot_heatmap(image[0], heatmap[0], save_file=f"{random_idx}/original_true.png")
    # 2. Grad-CAM on perturbed image for true class
    heatmap = grad_cam.generate_heatmaps(perturbed_image, true_class)
    GradCAM.plot_heatmap(perturbed_image, heatmap[0], save_file=f"{random_idx}/pert_true.png")
    # 2. Grad-CAM on perturbed image for predicted class
    heatmap = grad_cam.generate_heatmaps(perturbed_image, torch.LongTensor([new_prediction]))
    GradCAM.plot_heatmap(perturbed_image, heatmap[0], save_file=f"{random_idx}/pert_pred.png")
    
    plt.show()
    