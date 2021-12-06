"""
Code by: Jean Charle Yaacoub
Implementation and wrapper for the DQN model from the Atari deepmind paper.
"""

import tensorflow as tf
import numpy as np

class DeepQNetwork(object):
    def __init__(self, num_actions: int, learning_rate=0.1, batch_size=32, weights:str=None, model_config="2013"):
        """
        An implementation of the exact network used in the Atari paper. So not many arguments needed.

        Args:
            num_actions (int): The number of possible actions. This will define the output shape of the model.
            learning_rate (float, optional): Defaults to 0.1.
            batch_size (int, optional): Defaults to the size that is used in the paper.
            weights (str, optional): The file name of the model weights to load up. Defaults to None.
            model_config (str, optional): Which paper's model to use (2013 or 2015)
        """

        assert model_config in ["2013", "2015"], "The model config must be either 2013 or 2015"
        self.config = model_config

        self.Model = self.__build_model(num_actions, model_config)
        if weights is not None:
            self.Model.load_weights(weights)

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate

    @staticmethod
    def __build_model(num_actions:int, config) -> tf.keras.Model:
        """
        This function builds the exact DQN model from the Atari paper.
            Input shape is (84, 84, 4).
                - This is the *normalized* preprocessed images of the last 4 frames in the history

            1st Hidden layer convolves input with 32 8x8 filters with stride 4.
            2nd hidden layer convolves 64 4x4 filters with stride 2.
            3rd hidden layer convolves 64 3x3 filters with stride 1.
            4th hidden layer convolves 1024 7x7 filters with stride 1.

            Output layer is a fully connected linear layer.
                - shape -> (a, ) where a is the number of actions.
                - the ouput corresponds to the predicted Q-values for each action taken.

        Args:
            num_actions (int): this determines the output shape.
            config (str, optional): Which paper's model to use (2013 or 2015)

        Returns:
            tf.keras.model: The DQN model from the paper.
        """
        if config == "2015":
            # First layer takes in the 4 grayscale cropped and normalized image
            input_lyr = tf.keras.layers.Input((84,84,4), name="Input_last_4_frames")

            # Convolutional layers
            x = tf.keras.layers.Conv2D(32, (8,8), activation='relu', strides=4, use_bias=False, name="Hidden_layer_1")(input_lyr)
            x = tf.keras.layers.Conv2D(64, (4,4), activation='relu', strides=2, use_bias=False, name="Hidden_layer_2")(x)
            x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', strides=1, use_bias=False, name="Hidden_layer_3")(x)
            x = tf.keras.layers.Conv2D(1024, (7,7), activation='relu', strides=1, use_bias=False, name="Hidden_layer_4")(x)

            # Flattening for dense output
            x = tf.keras.layers.Flatten(name="Final_flatten")(x)
            x = tf.keras.layers.Dense(num_actions, activation='linear')(x)

            return tf.keras.Model(inputs=input_lyr, outputs=x, name="ATARI_DQN")
        else:
            # first layer takes in the 4 grayscale cropped image
            input_lyr = tf.keras.layers.Input((84, 84, 4), name="Input_last_4_frames")

            # second layer convolves 16 8x8 then applies ReLU activation
            x = tf.keras.layers.Conv2D(16, (8,8), strides=4, name="Hidden_layer_1")(input_lyr)
            x = tf.keras.layers.Activation('relu')(x)

            # third layer is the same but with 32 4x4 filters
            x = tf.keras.layers.Conv2D(32, (4,4), strides=2, name="Hidden_layer_2")(x)
            x = tf.keras.layers.Activation('relu')(x)
            # James Note: Missing final dense hidden layer:
            x = tf.keras.layers.Flatten(name="Flatten")(x)
            x = tf.keras.layers.Dense(256, name="Hidden_layer_3")(x)
            x = tf.keras.layers.Activation('relu')(x)

            # output layer is a fullyconnected linear layer
            x = tf.keras.layers.Dense(num_actions, activation='linear')(x)
            
            return tf.keras.Model(inputs=input_lyr, outputs=x, name="ATARI_DQN")

    def compile(self):
        """
        Compiles the model with Adam optimizer with the learning rate that was passed it at __init__(), and huber loss.
        """
        self.Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                            loss=tf.keras.losses.Huber(),
                            metrics=[tf.keras.metrics.MeanSquaredError(),
                                    tf.keras.metrics.CategoricalAccuracy()])

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
        try:
            out = self.Model.predict(x, *args, **kwargs)
        except ValueError as e:
            raise Exception('MISS-MATCHED DIMENSIONS: try np.asarray() on the input before running predict.') from e

        return out

    def clone(self) -> tf.keras.Model:
        """
        Clones the DQN model with tf.keras.models.clone_model().

        Returns:
            tf.keras.Model: The deep clone of the model
        """
        new_model = DeepQNetwork(self.num_actions, self.learning_rate, self.batch_size)
        new_model.Model = tf.keras.models.clone_model(self.Model)
        return new_model
