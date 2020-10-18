import torch
from src.datasets import ResizedChestXRayDataset
from src.utilities import getImageNetClasses
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Retrieved from https://www.kaggle.com/nih-chest-xrays/data
dataset_csv  = "../../../datasets/chest-xray/BBox_List_2017.csv"
# Retrieved from https://drive.google.com/file/d/1w8uLZlKhdVsss2yN934hTXMBWpV52itG/view
image_folder = "../../../datasets/chest-xray/images_small/"
# Retrieved from https://github.com/gustavo-beck/DD2424-Deep_Learning-COVID-Project/tree/master/saved_models
model_file = "final_model_16.pt"


model = torch.load(model_file, map_location=device)
dataset = ResizedChestXRayDataset(dataset_csv, image_folder)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for image, label, bbox in dataloader:
    out = model.forward(image)
    predicted_class_name = dataset.index_to_label[int(torch.argmax(out[0]))]
    true_class_name = dataset.index_to_label[int(label)]
    
    plt.imshow(torch.clamp(image.squeeze(0), 0, 1).numpy().transpose(1,2,0))

    plt.title(f"Predicted: {predicted_class_name} | True: {true_class_name}")
    plt.show()