"""
Code by: Jean Charle Yaacoub
Implementation and wrapper for the DQN model from the Atari deepmind paper.
"""

import tensorflow as tf
import numpy as np

class DeepQNetwork(object):
    def __init__(self, num_actions: int, learning_rate=0.1, batch_size=32):
        """
        An implementation of the exact network used in the Atari paper. So not many arguments needed.

        Args:
            num_actions (int): The number of possible actions. This will define the output shape of the model.
            learning_rate (float, optional): Defaults to 0.1.
            batch_size (int, optional): Defaults to the size that is used in the paper.
        """
        self.Model = self.__build_model(num_actions)

        # Compiling the model with RMSProp
        self.Model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                            loss=tf.keras.losses.CategoricalCrossentropy(), 
                            metrics=[tf.keras.metrics.MeanSquaredError(),
                                    tf.keras.metrics.CategoricalAccuracy()])

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

    def fit(self, *args, **kwargs):
        """
        Calls tf.keras.Model.fit() on the DQN model
        """
        self.Model.fit(*args, **kwargs)

    def predict(self, x: np.array, *args, **kwargs) -> np.array:
        """
        Runs tf.keras.Model.predict()
        
        Args
            x (np.array): Array of input samples (each sample is 4 frames of 84x84 crops)

        Returns:
            np.array: The list of action-value predictions.
        """

        return self.Model.predict(x, *args, **kwargs)

    def clone(self) -> tf.keras.Model:
        """
        Clones the DQN model with tf.keras.models.clone_model().

        Returns:
            tf.keras.Model: The deep clone of the model
        """
        return tf.keras.models.clone_model(self.Model)

    def set_weights(self, model_weights:np.array):
        """
        Sets the weights of the DQN model.

        Args:
            model_weights (np.array): the model weights
        """
        self.Model.set_weights(model_weights)

    def get_weights(self) -> np.array:
        """
        Gets the weights of the DQN model.

        Returns:
            (np.array): the model weights
        """
        return self.Model.get_weights(model_weights)

    def save_weights(self, filepath: str, *args, **kwargs):
        """
        Calls tf.keras.Model.save_weights() on the DQN model

        Args:
            filepath (str): path to save the weights file to.
        """
        self.Model.save_weights(filepath, *args, *kwargs)

    def load_weights(self, filepath: str, *args, **kwargs):
        """
        Calls tf.keras.Model.load_weights() on the DQN model

        Args:
            filepath (str): path to load the weights file from.
        """
        self.Model.load_weights(filepath, *args, *kwargs)

    def summary(self):
        """
        Runs built-in tf.keras.Model.summary() function on the DQN model
        """
        self.Model.summary()
