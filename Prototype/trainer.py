import os
from config import *
import prototype_model
import tensorflow as tf
from utils import utils
from model_components import model_tools
from tensorflow.python.client import device_lib

# check current device
print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# create tensorflow session
session=tf.Session()

# create placeholders for images and labels
images_ph=tf.placeholder(tf.float32,shape=[None,height,width,color_channels])
labels_ph=tf.placeholder(tf.float32,shape=[None,number_of_classes])

# main trainer function
def trainer(network,number_of_images):

    """
    :param network: import initialized network architecture
    :param number_of_images: literally number of images
    :return: train the network using pre-processed image data and save models periodically
    """

    # find error of nodes
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=network,labels=labels_ph)

    # average errors of all nodes
    cost=tf.reduce_mean(cross_entropy)
    tf.summary.scalar("cost", cost)#for tensorboard visualisation

    # backpropagate to using AdamOptimizer.
    optimizer=tf.train.AdamOptimizer().minimize(cost)

    #print(optimizer)
    session.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(model_save_name, graph=tf.get_default_graph())
    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=4)
    counter=0

    # go through epochs
    for epoch in range(epochs):
        tools = utils()
        print(number_of_images )
        #print(int(number_of_images / batch_size))
        for batch in range(int(number_of_images / batch_size)):
            counter+=1
            images, labels = tools.batch_dispatch()
            if images == None:
                break
            loss,summary = session.run([cost,merged], feed_dict={images_ph: images, labels_ph: labels})
            print('loss', loss)
            session.run(optimizer, feed_dict={images_ph: images, labels_ph: labels})
            print('Epoch number ', epoch, 'batch', batch, 'complete')
            writer.add_summary(summary,counter)
        #print(os.path.join(model_save_name))
        saver.save(session, os.path.join("./" + model_save_name))

if __name__=="__main__":
    """
    run this part to train model. 
    """
    tools=utils()
    model=model_tools()
    network=prototype_model.generate_model(images_ph,number_of_classes)
    #print (network)
    number_of_images = sum([len(files) for r, d, files in os.walk("processed_data")])

    trainer(network,number_of_images)



