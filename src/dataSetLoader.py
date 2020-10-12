import torch
from PIL import Image
from torchvision import transforms

# Transforms from imagenet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


class dataSetILS(torch.utils.data.Dataset):
    def __init__(self, dataframe, transforms=None):
        self.df = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        """ Returns the image based on the path stored in the dataframe"""
        row = self.df.iloc[index]
        full_path = row.loc['path']
        img_array = Image.open(full_path).convert('RGB')
        if self.transforms is not None:
            img_array = self.transforms(img_array)
        return img_array

    def __len__(self):
        return self.df.shape[0]
