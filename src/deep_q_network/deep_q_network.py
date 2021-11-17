class DeepQNetwork(object):
    def __init__(self, num_units, num_layers, batch_size, learning_rate, **kwargs):

        # All things I need from the agent class
        self.num_units = num_units
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def predict(self, x):
        # Should be able to handle a single or batch input
        # Should return the exact output of the network (ie do not argmax it)
        raise NotImplementedError

    def fit(self, x, y, **kwargs):
        # Fit like keras (use whatever args you want)
        raise NotImplementedError

    def clone(self):
        # Return a deep copy of this object
        raise NotImplementedError

    def set_weights(self, model):
        # Set this models weights using the argument's weights
        raise NotImplementedError

