import tensorflow as tf

class DeepQNetwork(object):
    def __init__(self, num_actions: int, learning_rate=0.1, batch_size=32):
        """
        An implementation of the exact network used in the Atari paper. So not many arguments needed.

        Args:
            num_actions (int): The number of possible actions. This will define the output shape of the model.
            learning_rate (float, optional): Defaults to 0.1.
            batch_size (int, optional): Defaults to the size that is used in the paper.
        """
        self.model = self.__build_model(num_actions)

        self.__compile_model(learning_rate, batch_size)

    def __compile_model(self, learning_rate, batch_size):
        """
        Compiles the built model with RMSProp
        """
        pass

    @staticmethod
    def __build_model(num_actions:int) -> tf.keras.Model:
        """
        This function builds the exact DQN model from the Atari paper.
            Input shape is (84, 84, 4).
                - This is the preprocessed images of the last 4 frames in the history

            1st Hidden layer convolves 16 8x8 filters with stride 4.
                - followed by rectifier nonlinear

            2nd hidden layer convolves 32 4x4 filters with stride 2.
                - again followed by rectifier nonlinearity

            Output layer is a fully connected linear layer.
                - shape -> (a, ) where a is the number of actions
                - the ouput corresponds to the predicted Q-values

        Args:
            num_actions (int): this determines the output shape

        Returns:
            tf.keras.model: The DQN model from the paper
        """
        # first layer takes in the 4 grayscale cropped image 
        input_lyr = tf.keras.layers.Input((84,84,4), name="Input_last_4_frames")
        
        # second layer convolves 16 8x8 then applies ReLU activation
        x = tf.keras.layers.Conv2D(16, (8,8), strides=4, name="Hidden_layer_1")(input_lyr)
        x = tf.keras.layers.Activation('relu')(x)

        # third layer is the same but with 32 4x4 filters
        x = tf.keras.layers.Conv2D(32, (4,4), strides=2, name="Hidden_layer_2")(x)
        x = tf.keras.layers.Activation('relu')(x)

        # output layer is a fullyconnected linear layer
        x = tf.keras.layers.Dense(num_actions, activation='linear')(x)

        return tf.keras.Model(inputs=input_lyr, outputs=x, name="ATARI_DQN")

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

dqn = DeepQNetwork(10)
print(dqn.model.summary())