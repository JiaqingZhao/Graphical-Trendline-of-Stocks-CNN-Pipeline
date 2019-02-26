import cv2
import tensorflow as tf
import os
import numpy as np
from model_components import model_tools

model=model_tools()
model_folder='checkpoints'
image='sup.jpg'
img=cv2.imread(image)
session=tf.Session()
img=cv2.resize(img,(100,100))
img=img.reshape(1,100,100,3)
labels = np.zeros((1, 2))

# load trained model
saver = tf.train.import_meta_graph(os.path.join(model_folder,'.meta'))
saver.restore(session,os.path.join(model_folder,'.\\'))

# create graph object for getting the same network architecture
graph = tf.get_default_graph()
network = graph.get_tensor_by_name("add_4:0")

# create placeholders to pass the image and get output labels
im_ph= graph.get_tensor_by_name("Placeholder:0")
label_ph = graph.get_tensor_by_name("Placeholder_1:0")

# softmax activation for model
network=tf.nn.softmax(network)

feed_dict_testing = {im_ph: img, label_ph: labels}
result=session.run(network, feed_dict=feed_dict_testing)
print(result)



