import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from collections import deque
from src.atari_env.atari import Atari
from src.deep_q_agent.agent_logger import DeepQLog
from src.deep_q_network.deep_q_network import DeepQNetwork


class DeepQAgent(object):
    def __init__(self,
                 game: Atari,
                 model: DeepQNetwork,
                 gamma: float,
                 epsilon: float,
                 replay_memory_size: int,
                 main_model_train_horizon: int,
                 target_update_horizon: int,
                 exploration_steps: int,

                 min_replay_memory_size: int = None,
                 target_model: DeepQNetwork = None,
                 epsilon_decay: float = None,
                 min_epsilon: float = 0.1,
                 save_name = "deep_q_agent.csv",
                 save_frequency = 100
                 ):
        """[summary]

        Args:
            game (Atari): [description]
            model (DeepQNetwork): [description]
            gamma (float): [description]
            epsilon (float): [description]
            replay_memory_size (int): [description]
            main_model_train_horizon (int): [description]
            target_update_horizon (int): [description]
            exploration_steps (int): Number of steps before starting to train.
            min_replay_memory_size (int, optional): [description]. Defaults to replay_memory_size / 50.
            target_model (DeepQNetwork, optional): [description]. Defaults to model.clone().
            epsilon_decay (float, optional): [description]. Defaults to 0.99975.
            min_epsilon (float, optional): [description]. Defaults to 0.1.
            save_name (str, optional): [description]. Defaults to "deep_q".
            save_frequency (int, optional): Number of episodes between saving agent. Defaults to 100.
        """

        # Save the main model
        self.main_model = model
        # Create a clone of the main model for the target model
        self.target_model = model.clone() if target_model is None else target_model
        # Create a deque with size replay_memory_size
        self.replay_memory = deque(maxlen=replay_memory_size)
        # Set the replay memory size (defaulted to replay_memory_size / 50)
        self.min_replay_memory_size = replay_memory_size / 50 if replay_memory_size is None else min_replay_memory_size

        # The number of episodes that pass before we update the target model
        self.target_update_horizon = target_update_horizon
        self.main_model_train_horizon = main_model_train_horizon

        # We store the total number of frames
        self.step_count = 0
        self.exploration_steps = exploration_steps

        # Store gamma
        assert 0 < gamma < 1, "Gamma must be 0 < gamma < 1"
        self.gamma = gamma

        # Setup our epsilon values
        assert 0 <= epsilon <= 1, "Epsilon must be 0 < epsilon < 1"
        assert 0 <= min_epsilon <= epsilon, "Epsilon must be 0 < min_epsilon < epsilon"
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.current_epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Set the game for the agent
        self.game = game
        self.action_count = {
            "total": np.zeros(self.game.action_space_size),
            "episode": np.zeros(self.game.action_space_size)
        }

        # ...

        # Create a logger and set the name
        self.name = save_name
        self.save_frequency = save_frequency
        self.logger = DeepQLog(log_path=save_name)


    def __update_replay_memory(self, transition: tuple):
        """[summary]

        Args:
            transition (tuple): tuple(state, action, reward, state_prime, done)
        """
        self.replay_memory.append(transition)

    def __get_pred_main(self, state: np.array) -> np.array:
        """[summary]

        Args:
            state (np.array):

        Returns:
            np.array: [description]
        """
        return self.main_model.predict((state / 255.0).astype(np.float32))

    def __get_pred_target(self, state: np.array) -> np.array:
        """[summary]

        Args:
            state (np.array):

        Returns:
            np.array: [description]
        """
        return self.target_model.predict((state / 255.0).astype(np.float32))

    def train(self, max_episodes, max_frames, frames_per_epoch=0, show_game=False, verbose=1):
        """[summary]

        Args:
            max_episodes ([type]): [description]
            show_game (bool, optional): [description]. Defaults to False.
            verbose (int, optional): [description]. Defaults to 1.
        """

        if verbose > 0:
            print("Training Initiated:")

        pbar = tqdm(total=max_frames)
        pbar.update(self.step_count)
        for episode in range(1, max_episodes + 1):

            # Set episode values
            done = False
            current_step = 1
            current_state = self.game.reset()
            total_episode_reward = 0

            # update the logger
            labels = ["episode", "epsilon", "step_count", "reward", "replay_memory_size"]
            rows = [episode, self.current_epsilon, self.step_count, total_episode_reward, len(self.replay_memory)]
            self.logger(labels, rows)

            while not done:
                # Choose an action and step with it
                action = self.get_action(current_state)

                # Count number of actions
                self.action_count["episode"][action] += 1
                self.action_count["total"][action] += 1

                # Get the result of taking our action, which returns a stacked state
                new_state, reward, done, info = self.game.step(action, False)

                # Add to the total reward for the episode
                total_episode_reward += reward

                # Render the game if we want to
                if show_game and episode % 10 == 0:
                    self.game.render()

                # Update the replay memory with a new item
                memory_item = (current_state, action, reward, new_state, done)
                self.__update_replay_memory(memory_item)

                # If we have taken an action (with frame skip), update the network.
                self.__train_network(verbose)

                # Update the current state and current step
                current_state = new_state
                current_step += 1

                # Decay current epsilon
                self.__eps_linear_decay()

                # Update the frame counter
                self.step_count += 1

            # Update the progress bar with the number of steps in the episode
            pbar.update(current_step)

            print(f"Episode {episode} Complete")
            print(f"  Total Episode Rewards: {total_episode_reward}")
            print(f"  Replay Memory Size: {len(self.replay_memory)}")
            print(f"  Frame Count: {self.step_count}")
            print(f"  Epsilon: {self.current_epsilon}")
            print(f"  Actions Episode:", self.action_count["episode"])
            print(f"  Actions Total:", self.action_count["total"])
            print("-"*50)

            # Save the agent at their checkpoint
            if episode % self.save_frequency == 0:
                self.save("agent_checkpoint")

            # Reset the action count
            self.action_count["episode"] = np.zeros(self.game.action_space_size)

            if self.step_count > max_frames:
                print("Done: Saving Models")
                self.main_model.Model.save("main_model")
                self.target_model.Model.save("target_model")
                pbar.close()
                return

    def get_action(self, state: np.array, inference_epsilon=None) -> int:
        """[summary]

        Args:
            state (np.array): [description]
            inference_epsilon ([type], optional): [description]. Defaults to None.

        Returns:
            int: [description]
        """

        # Check if we are running inference
        if inference_epsilon is None:
            eps = self.current_epsilon
        else:
            eps = inference_epsilon

        # Run epsilon greedy
        if np.random.random() > eps:
            return np.argmax(self.__get_pred_main(np.expand_dims(state, axis=0)))
        return np.random.randint(0, self.game.action_space_size)

    def __train_network(self, verbose):
        """[summary]

        Args:
            terminal_state ([type]): [description]
        """
        # Allow the agent to explore before training
        if len(self.replay_memory) <= self.min_replay_memory_size or self.step_count <= self.exploration_steps:
            return

        # Get a mini batch from the memory
        idx = np.random.choice(len(self.replay_memory), self.main_model.batch_size)
        batch = []
        states = []
        states_prime = []
        for i in idx:
            memory_item = self.replay_memory[i]
            batch.append(memory_item)
            states.append(memory_item[0])
            states_prime.append(memory_item[3])

        states_prime = np.array(states_prime)
        states = np.array(states)

        # Get the current qs
        current_qs = self.__get_pred_main(states)

        # Get all of the next states
        future_qs = self.__get_pred_target(states_prime)

        # Create out
        x, y = [], []
        for index, (current_state, action, reward, _, done) in enumerate(batch):

            # Get our new q value
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # Update the desired q value
            current_q_value = current_qs[index]
            current_q_value[action] = new_q

            # Add to the dataset
            x.append(current_state)
            y.append(current_q_value)

        # Fit our model
        self.main_model.fit(np.array(x), np.array(y), verbose=0)

        # Once we have completed enough frames, reset the counter and update the target
        if self.step_count % self.target_update_horizon == 0:
            self.target_model.Model.set_weights(self.main_model.Model.get_weights())

            if verbose > 0:
                print(f"Target Model Updated At Frame {self.step_count} with {len(self.replay_memory)} memory items.")

    def reset_all(self):
        """
        Resets the current value of epsilon to the max value of epsilon
        and the frame counter. Along with other things which could be added later.
        """
        self.reset_epsilon()
        self.step_count = 0

    def reset_epsilon(self):
        """
        Resets the current value of epsilon to the max value of epsilon
        """
        self.current_epsilon = self.max_epsilon

    def save(self, filename):
        """
        Saves the model and overwrites any file or directory with the same name

        Args:
            filename ([type]): [description]
        """
        # Check if we have a dir already for saving
        if not os.path.exists(filename):
            # Make the dir for the save
            os.mkdir(filename)

        # Save the params
        info = {}
        with open(f"{filename}/params.json", "w") as f:
            # Polulate a json
            info["min_eps"] = self.min_epsilon
            info["epsilon"] = self.current_epsilon
            info["max_epsilon"] = self.max_epsilon
            info["epsilon_decay"] = self.epsilon_decay

            info["frames"] = self.step_count

            info["gamma"] = self.gamma

            info["min_replay_memory_size"] = self.min_replay_memory_size
            info["target_update_horizon"] = self.target_update_horizon
            info["main_model_train_horizon"] = self.main_model_train_horizon
            json.dump(info, f)

        # Save the replay memory
        with open(f"{filename}/replay_memory", "wb") as dq:
            pickle.dump(self.replay_memory, dq)

        # And the models
        self.main_model.Model.save(f"{filename}/main_model.h5")
        self.target_model.Model.save(f"{filename}/target_model.h5")

    def load(self, filepath):
        """
        Loads parameters from previous run.

        Args:
            filepath ([type]): [description]
        """
        info = json.load(open(f"{filepath}/params.json", "r"))
        self.min_epsilon = info["min_eps"]
        self.current_epsilon = info["epsilon"]
        self.max_epsilon = info["max_epsilon"]
        self.step_count = info["frames"]
        self.gamma = info["gamma"]
        self.min_replay_memory_size = info["min_replay_memory_size"]
        self.target_update_horizon = info["target_update_horizon"]
        self.main_model_train_horizon = info["main_model_train_horizon"]
        self.epsilon_decay = info["epsilon_decay"]

        # Load the replay memory
        with open(f"{filepath}/replay_memory", "rb") as dq:
            self.replay_memory = pickle.load(dq)

        # Save the models in .h5 format (to stay consistent)
        self.main_model.Model.set_weights(keras.models.load_model(f"{filepath}/main_model.h5").get_weights())
        self.target_model.Model.set_weights(keras.models.load_model(f"{filepath}/target_model.h5").get_weights())

        # Set up the labels to avoid overwriting any saved data
        self.logger.labels = ["episode", "epsilon", "step_count", "reward", "replay_memory_size"]

    def __eps_linear_decay(self):
        self.current_epsilon -= self.epsilon_decay
        self.current_epsilon = max(self.min_epsilon, self.current_epsilon)
