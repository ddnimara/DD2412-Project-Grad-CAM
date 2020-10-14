import src.models as models
from src.methods.adversarial_attacks import FastGradientSignMethod
from torch.utils.data.dataloader import DataLoader

from src.datasets import ResizedImagenetDataset

# Resized ImageNet _validation_ dataset
imagenet_csv_path = "../../../datasets/imagenet/resized.csv"
dataset = ResizedImagenetDataset(imagenet_csv_path)

dataloader = DataLoader(dataset)
model = models.getVGGModel(16)
attacker = FastGradientSignMethod(model)

image, label = dataset[6465]
attacker.generate_attack(image, label)
