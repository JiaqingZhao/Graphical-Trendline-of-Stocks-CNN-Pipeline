import pre_processing as ppr
import os

# file paths
raw_data='Output'
data_path='processed_data'
model_save_name='saved/checkpoints'

# target image size
height=100
width=100

# parse class list from path
if not os.path.exists(data_path):
    ppr.image_processing(raw_data,data_path,height,width)
raw_classes = os.listdir(data_path)
all_classes = []
for i in raw_classes:
    if "DS_Store" in i:
        pass
    else:
        all_classes.append(i)

# other parameters
number_of_classes = len(all_classes)
color_channels=3
epochs=300
batch_size=10
batch_counter=0

