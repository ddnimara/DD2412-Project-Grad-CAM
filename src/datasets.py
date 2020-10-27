from ast import literal_eval
from os import path

import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import pandas as pd
from PIL import Image

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
        
        # the labels in the csv for an image are lists represented as strings
        # e.g. "[10]" or "[670, 670]" (there can be multiple objects in the image)
        # literal_eval correctly interprets them as lists of ints instead of strings
        image_labels = self.df['id'][index]
        image_labels = literal_eval(image_labels)
        
        # A 1 x 1000 dimensional vector 
        target_vector = torch.zeros(1000)
        for idx in image_labels:
            target_vector[idx] += 1

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

class ResizedChestXRayDataset(Dataset):
    def __init__(self, csv_path, image_folder, normalize=True):
        self.df = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transforms = self.init_transforms(normalize)
                
        # The label order is taken from this csv:
        # https://raw.githubusercontent.com/gustavo-beck/DD2424-Deep_Learning-COVID-Project/master/xray_dataset/organized_dataset.csv

        self.label_to_index = {
            "Cardiomegaly":        0,
            "Emphysema":           1,
            "Effusion":            2,
            "Hernia":              3,
            "Infiltrate":          4,
            "Mass":                5,
            "Nodule":              6,
            "Atelectasis":         7,
            "Pneumothorax":        8,
            "Pleural_Thickening":  9,
            "Pneumonia":          10,
            "Fibrosis":           11,
            "Edema":              12,
            "Consolidation":      13
        }
        
        self.index_to_label = {v:k for k,v in self.label_to_index.items()}
             
    def __getitem__(self, index):
        """Return the image, the label and the bounding box at the given index."""
        df_row = self.df.iloc[index]
        
        image_path = path.join(self.image_folder + df_row['Image Index'])
        image = self.transforms(Image.open(image_path).convert('RGB'))
        
        label = self.label_to_index[df_row['Finding Label']]
        
        bounding_box = torch.Tensor([df_row['Bbox [x'], df_row['y'], df_row['w'], df_row['h]']])
        
        return image, label, bounding_box
    
    def __len__(self):
        return self.df.shape[0]
    
    def init_transforms(self, normalize):
        if normalize:
            return torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                 std = [0.229, 0.224, 0.225])])
        else:
            return torchvision.transforms.ToTensor()