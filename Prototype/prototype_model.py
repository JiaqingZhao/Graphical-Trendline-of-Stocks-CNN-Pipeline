from model_components import model_tools
import tensorflow as tf
model=model_tools()

def generate_model(images_ph,number_of_classes):
    """
    :param images_ph: image placeholders of the same size as our images
    :param number_of_classes: number of classes
    :return: fully initialized model. cuurently just a simple prototype
    """
    #level 1 convolution
    network=model.conv_layer(images_ph,5,3,16,1)
    network=model.pooling_layer(network,5,2)
    network=model.relu_function(network)
    print(network)

    #level 2 convolution
    network=model.conv_layer(network,4,16,32,1)
    network=model.pooling_layer(network,4,2)
    network=model.relu_functionr(network)
    print(network)

    #level 3 convolution
    network=model.conv_layer(network,3,32,64,1)
    network=model.pooling_layer(network,3,2)
    network=model.relu_function(network)
    print(network)

    #flattening layer
    network,features=model.flattening_layer(network)
    print(network)

    #fully connected layer
    network=model.fully_connected_layer(network,features,1024)
    network=model.relu_function(network)
    print(network)

    #output layer
    network=model.fully_connected_layer(network,1024,number_of_classes)
    print(network)
    return network


if __name__== "__main__":
    images_ph = tf.placeholder(tf.float32, shape=[None, 100,100,3])
    generate_model(images_ph,2)