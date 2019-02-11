import preprocessing as ppr
import os

#Parameters
raw_data='output'
data_path='processed_data'
height=100
width=100
if not os.path.exists(data_path):
    ppr.image_processing(raw_data,data_path,height,width)
raw_classes = os.listdir(data_path)
all_classes = []
for i in raw_classes:
    if "DS_Store" in i:
        pass
    else:
        all_classes.append(i)
number_of_classes = len(all_classes)
color_channels=3
epochs=300
batch_size=10
batch_counter=0
model_save_name='saved/checkpoints'
