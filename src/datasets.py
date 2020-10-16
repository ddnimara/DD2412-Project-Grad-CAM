from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import torchvision
import pandas as pd
from PIL import Image
from ast import literal_eval

class ResizedImagenetDataset(Dataset):
    def __init__(self, csv_path, normalize=True):
        self.df = pd.read_csv(csv_path)
        
        if normalize:
            self.transforms = self._default_transforms()
        else:
            self.transforms = torchvision.transforms.ToTensor()
            
    def __getitem__(self, index):       
        image_path = self.df['path'][index]        
        image = self.transforms(Image.open(image_path).convert('RGB'))
        # Add batch dimension
        image = image.unsqueeze(0)
        
        # the labels in the csv for an image are lists represented as strings
        # e.g. "[10]" or "[670, 670]" (there can be multiple objects in the image)
        # literal_eval correctly interprets them as lists of ints instead of strings
        image_labels = self.df['id'][index]
        image_labels = literal_eval(image_labels)
        
        # A 1 x 1000 dimensional vector 
        target_vector = torch.zeros(1,1000)
        for idx in image_labels:
            target_vector[0,idx] += 1

        target_vector /= len(image_labels)
        
        return image, target_vector
    
    def __len__(self):
        return self.df.shape[0]

    def _default_transforms(self):
        # Imagenet should be normalized to predefined mean/std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
            
        default_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std)])
        
        return default_transform