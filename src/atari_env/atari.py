import numpy as np
import gym
import cv2
import matplotlib.pyplot as plt


# Added this to plot the four frame skips if true (only for debugging)
PLOT_FRAMES = False


class Atari(object):
    def __init__(self, game: str, max_steps: int, resized_width: int, resized_height: int):
        self.env = gym.make(game)
        self.env.reset()
        self.max_steps = max_steps # Max steps per episode (int)
        self.action_space_size = self.env.action_space.n # The number of actions an agent can perform (int)
        self.resized_width = resized_width
        self.resized_height = resized_height

    def reset(self, frame_skip=1, **kwargs):
        # Should reset the environment to the begining
        # Returns initial state
        return np.stack([self.get_preprocessed_frame(self.env.reset()) for _ in range(frame_skip)], axis=2).astype('float16')

    def num_actions_available(self):
        # Return total number of actions
        return self.env.action_space.n

    def get_preprocessed_frame(self, observation):
        
        # crop image (top and bottom, top from 34, bottom remove last 16)
        img = observation[34:-16, :, :]
        
        # resize image
        img = cv2.resize(img, (84, 84))
        img = img.mean(-1,keepdims=True)
        
        img = img.astype('float32') / 255.
        return img[:,:,0]

    def print_action_meanings(self):
        # Prints meaings of all possible actions
        print(self.env.get_action_meanings())

    def step(self, action: int, frame_skip=1) -> tuple:
        """[summary]

        Args:
            action (int): [description]
            frame_skip (int, optional): [description]. Defaults to 1.

        Returns:
            tuple: [description]
        """

        # Note from James
        # Frame skip was not implemented >>> Now it is

        total_reward = 0
        observations = []
        for _ in range(frame_skip):
            # take a step in the env and return the next state, reward,
            # and if the game is done as a tuple
            observation, reward, done, lives = self.env.step(action)
            observations.append(self.get_preprocessed_frame(observation))
            # Step function to set rewards to -1, +1 or 0
            if reward > 0:
                total_reward += 1
            elif reward < 0:
                total_reward += -1
            else:
                total_reward += 0

        stacked_images = np.stack(observations, axis=2).astype('float16')


        if PLOT_FRAMES:
            import matplotlib.pyplot as plt
            f, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(stacked_images[:, :, 0])
            axarr[0,0].set_title("Frame-1")

            axarr[0,1].imshow(stacked_images[:, :, 1])
            axarr[0,1].set_title("Frame-2")

            axarr[1,0].imshow(stacked_images[:, :, 2])
            axarr[1,0].set_title("Frame-3")

            axarr[1,1].imshow(stacked_images[:, :, 3])
            axarr[1,1].set_title("Frame-4")
            plt.show()

        return stacked_images, total_reward, done

    def render(self):
        # Render the game state
        self.env.render()




if __name__ == "__main__":
    game = Atari("Breakout-v0", 10000, 80, 80)
    game.print_action_meanings()
    print(game.step(0))