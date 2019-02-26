import tensorflow as tf

class model_tools:
    # Defined functions for all the basic tensorflow components that we needed for building a model.

    def add_weights(self,shape):

        """
        :param shape: shape of the layer
        :return: output weights as a variable with the shape of the layer
        """
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))

    def add_biases(self,shape):

        """
        :param shape: shape of the layer
        :return: output biases as a variable with the shape of the layer
        """
        return tf.Variable(tf.constant(0.05, shape=shape))

    def conv_layer(self,layer, kernel, input_shape, output_shape, stride_size):

        """
        :param layer: the image that feeds in this layer
        :param kernel: filter side length
        :param input_shape: the input shape
        :param output_shape: the output shape
        :param stride_size: the size of the stride
        :return: the initialized convolution layer with weights and constants added
        """
        weights = self.add_weights([kernel, kernel, input_shape, output_shape])
        biases = self.add_biases([output_shape])
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,1,1,1] mostly
        stride = [1, stride_size, stride_size, 1]
        layer = tf.nn.conv2d(layer, weights, strides=stride, padding='SAME') + biases
        return layer

    def pooling_layer(self,layer, kernel_size, stride_size):

        """
        :param layer: imagee inputs for the layer
        :param kernel_size: the size of the filter
        :param stride_size: size of the stride
        :return: the initialized pooling layer with weights and constants added
        """
        #kernel=[image_jump,rows,columns,depth]
        kernel = [1, kernel_size, kernel_size, 1]
        #stride=[image_jump,row_jump,column_jump,color_jump]=[1,2,2,1] mostly
        stride = [1, stride_size, stride_size, 1]
        return tf.nn.max_pool(layer, ksize=kernel, strides=stride, padding='SAME')

    def flattening_layer(self,layer):

        """
        :param layer: the pooled layer before the flattening layer
        :return: reshape the pooled layer into one dimension for fully connected layer
        """
        input_size = layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(layer, [-1, new_size]),new_size

    def fully_connected_layer(self,layer, input_shape, output_shape):

        """
        :param layer: flattened 1-D layer
        :param input_shape: size of input
        :param output_shape: size of output
        :return: the calculated fully connected layer which is just a matrix multiplication with matmul -> mX+b
        """
        weights = self.add_weights([input_shape, output_shape])
        biases = self.add_biases([output_shape])
        #most important operation
        layer = tf.matmul(layer,weights) + biases
        return layer

    def relu_function(self,layer):

        """
        :param layer: input values
        :return: output values after going though relu
        """
        return tf.nn.relu(layer)