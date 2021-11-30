import numpy as np
from tqdm import tqdm
from collections import deque
from src.atari_env.atari import Atari
from src.deep_q_agent.agent_logger import DeepQLog
from src.deep_q_network.deep_q_network import DeepQNetwork


class DeepQAgent(object):
    def __init__(self,
                 game: Atari,
                 model: DeepQNetwork,
                 gamma: float,
                 alpha: float,
                 epsilon: float,
                 replay_memory_size: int,
                 main_model_train_horizon: int,
                 target_update_horizon: int,

                 min_replay_memory_size: int = None,
                 target_model: DeepQNetwork = None,
                 epsilon_decay: float = 0.99975,
                 min_epsilon: float = 0.001,
                 name = "deep_q"
                 ):
        """[summary]

        Args:
            game (Atari): [description]
            model (DeepQNetwork): [description]
            gamma (float): [description]
            alpha (float): [description]
            epsilon (float): [description]
            replay_memory_size (int): [description]
            main_model_train_horizon (int): [description]
            target_update_horizon (int): [description]
            min_replay_memory_size (int, optional): [description]. Defaults to replay_memory_size / 50.
            target_model (DeepQNetwork, optional): [description]. Defaults to model.clone().
            epsilon_decay (float, optional): [description]. Defaults to 0.99975.
            min_epsilon (float, optional): [description]. Defaults to 0.001.
            name (str, optional): [description]. Defaults to "deep_q".
        """

        super().__init__()

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
        self.frame_count = 0

        # Store gamma and alpha
        self.gamma = gamma
        self.alpha = alpha

        # Setup our epsilon values
        self.max_epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.current_epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Set the game for the agent
        self.game = game
        # ...

        # Create a logger and set the name
        self.logger = DeepQLog()
        self.name = name


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

        return self.main_model.predict(state)

    def __get_pred_target(self, state: np.array) -> np.array:
        """[summary]

        Args:
            state (np.array):

        Returns:
            np.array: [description]
        """
        return self.target_model.predict(state)

    def train(self, max_episodes, show_game=False, verbose=1):
        """[summary]

        Args:
            max_episodes ([type]): [description]
            show_game (bool, optional): [description]. Defaults to False.
            verbose (int, optional): [description]. Defaults to 1.
        """

        if verbose > 0:
            print("Training Initiated:")

        for episode in tqdm(range(1, max_episodes + 1)):

            # Set episode values
            done = False
            current_step = 1
            current_state = self.game.reset(frame_skip=self.main_model_train_horizon)
            total_episode_reward = 0
            while not done and self.game.max_steps > current_step:
                # Choose an action and step with it
                action = self.get_action(current_state)

                # Get the result of taking our action, which returns a stacked state
                new_state, reward, done = self.game.step(action, self.main_model_train_horizon)

                # Add to the total reward for the episode
                total_episode_reward += reward

                # Render the game if we want to
                if show_game:
                    self.game.render()

                # Update the replay memory with a new item
                memory_item = (current_state, action, reward, new_state, done)
                self.__update_replay_memory(memory_item)

                # If we have taken an action (with frame skip), update the network.
                self.__train_network(verbose)

                # Update the current state and current step
                current_state = new_state
                current_step += 1

                # update the logger
                log_entry = f"episode-{episode}"
                self.logger(log_name=log_entry,
                            step=current_step,
                            total_reward=total_episode_reward,
                            epsilon=self.current_epsilon)

                # Decay current epsilon
                self.__eps_linear_decay()

                # Update the frame counter
                self.frame_count += 1

            print(f"Total Episode Rewards: {total_episode_reward}")
            print(f"Replay Memory Size: {len(self.replay_memory)}")
            print(f"Frame Count: {self.frame_count}")
            print(f"Epsilon: {self.current_epsilon}")
            print()

    def get_action(self, state: np.array) -> int:
        """[summary]

        Args:
            state (np.array): [description]

        Returns:
            int: [description]
        """
        if np.random.random() > self.current_epsilon:
            return np.argmax(self.__get_pred_main(np.expand_dims(state, axis=0)))
        return np.random.randint(0, self.game.action_space_size)

    def __train_network(self, verbose):
        """[summary]

        Args:
            terminal_state ([type]): [description]
        """

        if len(self.replay_memory) <= self.min_replay_memory_size:
            return

        # Get a mini batch from the memory
        idx = np.random.choice(len(self.replay_memory), self.main_model.batch_size)
        batch = np.array([self.replay_memory[i] for i in idx])

        # Get the current qs
        states = np.array([self.replay_memory[i][0] for i in idx])
        current_qs = self.__get_pred_main(states)

        # Get all of the next states
        states_prime = np.array([self.replay_memory[i][3] for i in idx])
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
        self.main_model.fit(np.array(x), np.array(y))

        # Once we have completed enough frames, reset the counter and update the target
        if self.frame_count % self.target_update_horizon == 0:
            self.target_model.set_weights(self.main_model)

            if verbose > 0:
                print(f"Target Model Updated At Frame {self.frame_count} with {len(self.replay_memory)} samples using a batch size of {self.main_model.batch_size}")

    def reset_all(self):
        """
        Resets the current value of epsilon to the max value of epsilon
        and the frame counter. Along with other things which could be added later.
        """
        self.reset_epsilon()
        self.frame_count = 0

    def reset_epsilon(self):
        """
        Resets the current value of epsilon to the max value of epsilon
        """
        self.current_epsilon = self.max_epsilon

    def __eps_exponential_decay(self):
        """[summary]
        """
        self.current_epsilon *= self.epsilon_decay
        self.current_epsilon = max(self.min_epsilon, self.current_epsilon)

    def __eps_linear_decay(self):
        self.current_epsilon -= self.epsilon_decay
        self.current_epsilon = max(self.min_epsilon, self.current_epsilon)
