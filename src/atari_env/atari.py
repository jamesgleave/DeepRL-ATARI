import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt

# James Note: Importing deque for stack
from collections import deque


class Atari(object):
    def __init__(self, game: str, resized_width: int, resized_height: int, frame_skip=4, clip_reward: bool = True):
        """[summary]

        Args:
            game (str): [description]
            resized_width (int): [description]
            resized_height (int): [description]
            frame_skip (int, optional): [description]. Defaults to 4.
            clip_reward (bool, optional): [description]. Defaults to True.
        """
        self.env = gym.make(game)
        self.action_space_size = self.env.action_space.n  # The number of actions an agent can perform (int)
        self.resized_width = resized_width
        self.resized_height = resized_height

        # James Note: Added these two attrs for frame stacking
        self._frame_stack = deque([], frame_skip)
        self.frame_skip = frame_skip
        self.output = None

        # Also clip reward
        self.clip_reward = clip_reward

    def reset(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Should reset the environment to the begining
        # Returns initial state
        # James Note: Implemented stacking from reset
        reset_obs = self.get_preprocessed_frame(self.env.reset())
        for _ in range(self.frame_skip):
            self._frame_stack.append(reset_obs)
        self.output = np.concatenate(self._frame_stack, axis=-1)
        return self.output

    def num_actions_available(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Return total number of actions
        return self.env.action_space.n

    def get_preprocessed_frame(self, observation):
        """[summary]

        Args:
            observation ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Convert image to grayscale
        # Rescale image
        # James Note: Rewrote this method to return a uint8 and expand the dims
        img = observation[34:-16, :, :]
        # Resize image
        img = cv2.resize(img, (84,84))
        # Grayscale
        img = img.mean(-1,keepdims=True)
        # Return as an unsigned integer to save space
        # Normalization occurs upon model input

        #import matplotlib.pyplot as plt
        #f, axarr = plt.subplots(2,2)
        #axarr[0,0].imshow((img[:, :, 0] / 255.0).astype(np.float32),  cmap='gray')
        #axarr[0,0].set_title("Frame-1")


        return img.astype("uint8")


    def get_action_meanings(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Prints meaings of all possible actions
        return self.env.get_action_meanings()

    def step(self, action: int, plot_frames: bool = False) -> tuple:
        """[summary]

        Args:
            action (int): [description]
            frame_skip (int, optional): [description]. Defaults to 1.

        Returns:
            tuple: [description]
        """

        # Note from James
        # Frame skip was not implemented >>> Now it is (rewrote method)

        total_reward = 0
        second_to_last = None
        last = None
        for i in range(self.frame_skip):
            # take a step in the env and return the next state, reward,
            # and if the game is done as a tuple
            observation, reward, done, info = self.env.step(action)

            # Sum up the total reward
            total_reward += reward

            # Store the second to last frame
            if i == self.frame_skip - 1:
                last = self.get_preprocessed_frame(observation)
            elif i == self.frame_skip - 2:
                second_to_last = self.get_preprocessed_frame(observation)

        # Get the max of the last frame
        max_frame = np.array([second_to_last, last]).max(axis=0)

        # Append the newest frame
        self._frame_stack.append(max_frame)

        # If the num of stacks is correct then we concat
        assert len(self._frame_stack) == self.frame_skip or self.output is not None, "Must Reset Env To Step"
        self.output = np.concatenate(self._frame_stack, axis=-1)

        if plot_frames:
            import matplotlib.pyplot as plt
            f, axarr = plt.subplots(2,2)

            axarr[0,0].imshow((self.output[:, :, 0] / 255.0).astype(np.float32), cmap='gray')
            axarr[0,0].set_title("Frame-1")

            axarr[0,1].imshow((self.output[:, :, 1] / 255.0).astype(np.float32), cmap='gray')
            axarr[0,1].set_title("Frame-2")

            axarr[1,0].imshow((self.output[:, :, 2] / 255.0).astype(np.float32), cmap='gray')
            axarr[1,0].set_title("Frame-3")

            axarr[1,1].imshow((self.output[:, :, 3] / 255.0).astype(np.float32), cmap='gray')
            axarr[1,1].set_title("Frame-4")
            plt.show()

        # Clip the reward
        if self.clip_reward:
            total_reward = np.clip(total_reward, -1, 1)

        return self.output[:, :, ::-1], total_reward, done, info

    def render(self):
        # Render the game state
        self.env.render()


if __name__ == "__main__":
    game = Atari("Breakout-v4", 84, 84)
    game.reset()
    print(game.step(0, plot_frames=True)[0].shape)
    print