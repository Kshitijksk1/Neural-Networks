import numpy as np
from sklearn.metrics import mean_squared_error 


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions)
        

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        if(self.weights.shape != W.shape):
            return -1
        self.weights = W

        

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        return self.weights
        #print(W)

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        # Multiply the weight matrix, W, by the input matrix X
        results = np.dot(self.weights, X)

        if self.transfer_function == "Hard_limit":
            actualResults = np.where(results < 0, 0, 1)
        else:
            actualResults = results
        return actualResults
            
        #print(actualResults)
        
    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        X_inverse = np.linalg.pinv(X)
        self.weights = np.dot(y,X_inverse)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        rem = int(np.ceil(len(X[0])/batch_size))
        for epoch in range(num_epochs):
            art = 0;
            for sample in range(rem):
                end = art + batch_size

                # Get a sample (column from X and Y) where the size of the sample is given by the batch size
                sampleX = X[:, art : end]
                sampleY = y[:, art : end]
                #print (sampleX)

                # Get the prediction
                results = self.predict(sampleX)
                art += batch_size

                if learning == "Delta" or learning == "delta":
                    # Calculate e
                    e = np.subtract(sampleY, results)

                    # Calculate e dot p, where p is the input matrix
                    ep = np.dot(e, np.transpose(sampleX))

                    # Multiply this new matrix by the scalar alpha
                    aep = np.multiply(alpha, ep)

                    # Calculate the new weights along with the bias
                    self.weights = np.add(self.weights, aep)
                    
                elif learning == "Filtered" or learning == "filtered":

                    # Calculate e dot p, where p is the input matrix
                    ep = np.dot(sampleY, np.transpose(sampleX))

                    # Multiply this new matrix by the scalar alpha
                    aep = np.multiply(alpha, ep)

                    # Multiply the old weights by some scalar gamma
                    gw = np.multiply(1 - gamma, self.weights)

                    self.weights = np.add(gw, aep)

                elif learning == "Unsupervised_hebb" or learning == "unsupervised_hebb":
                    # Add a row of one's to the top of the input matrix
                    #newX = np.vstack((np.array([1 for column in range(sampleX.shape[1])]), sampleX))

                    # Calculate e dot p, where p is the input matrix
                    ep = np.dot(results, np.transpose(sampleX))

                    # Multiply this new matrix by the scalar alpha
                    aep = np.multiply(alpha, ep)

                    # Calculate the new weights along with the bias
                    self.weights = np.add(self.weights, aep)

        
                

    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        mserror = 0
        results = self.predict(X)
        
        mserror = mean_squared_error(results,y)
        #print(error)
        return mserror
