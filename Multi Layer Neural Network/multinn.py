# %tensorflow_version 2.x
# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.multinn = []
        self.weights = []


    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        self.num_nodes = num_nodes
        if not self.multinn:
            weights = np.random.randn(self.input_dimension, self.num_nodes)
        else:
            weights = np.random.randn(self.multinn[-1]["weights"].shape[1], self.num_nodes)
        bias = np.random.randn(self.num_nodes)
        self.weights.append(None)
        layer = {"transfer_function":transfer_function, "weights":weights, "bias":bias}
        self.multinn.append(layer)
        
         

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.multinn[layer_number]["weights"]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.multinn[layer_number]["bias"]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.multinn[layer_number]["weights"] = weights 

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.multinn[layer_number]["bias"] = biases 


    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        yhat = tf.Variable(X)

        for layer in range(len(self.multinn)):
            
            weightedX = tf.matmul(yhat, self.get_weights_without_biases(layer))
            biasedX = tf.add(weightedX, self.get_biases(layer))
            if self.multinn[layer] ["transfer_function"] == "Relu" or self.multinn[layer] ["transfer_function"] == "relu":
                biasedX = tf.nn.relu(biasedX, name="Relu")
            elif self.multinn[layer] ["transfer_function"] == "Sigmoid" or self.multinn[layer] ["transfer_function"] == "sigmoid":
                biasedX = tf.nn.sigmoid(biasedX, name="Sigmoid")
            else:
                biasedX = biasedX
            yhat = biasedX

        return yhat 


    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """

        for epoch in range(0, num_epochs):
            for sample in range (0, len(X_train),batch_size):
                
                xX = X_train[sample:sample+batch_size]
                yY = y_train[sample:sample+batch_size]
                
                weights_b = []
                bias_w = []
                for i in range(len(self.multinn)):
                    weights_b.append(self.get_weights_without_biases(i))
                    bias_w.append(self.get_biases(i))
                with tf.GradientTape() as tape:
                    results = self.predict(xX)
                    loss = self.calculate_loss(yY, results)
                    weights_dw, bias_db = tape.gradient(loss, [weights_b, bias_w])
                for m in range(len(weights_b)):
                    weights_b[m].assign_sub(alpha*weights_dw[m])
                    bias_w[m].assign_sub(alpha*bias_db[m])



    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        x = 0
        yhat = y
        results = self.predict(X)
        results = np.argmax(results, axis = 1)
        for i in range(len(results)):
            if yhat[i] != results[i]:
                x+=1
        return(x/len(results))


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        yhat = y
        results = self.predict(X)
        results = np.argmax(results, axis = 1)
        return tf.math.confusion_matrix(yhat,results)

