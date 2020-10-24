import pandas as pd
from os.path import join
root_folder = "../../../datasets/chest-xray/"

bbox_csv = join(root_folder, "BBox_List_2017.csv")
bbox_df = pd.read_csv(bbox_csv)

resized_width = 224
resized_height = 224

horizontal_scaling = 224 / 1024
vertical_scaling = 224 / 1024 

bbox_df["Bbox [x"] *= horizontal_scaling
bbox_df["y"] *= vertical_scaling
bbox_df["w"] *= horizontal_scaling
bbox_df["h]"] *= vertical_scaling

bbox_df.to_csv(join(root_folder, "Resized_BBox_List.csv"))