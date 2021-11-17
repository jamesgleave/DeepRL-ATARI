import numpy as np
from tqdm import tqdm
from collections import deque
from atari_env.atari import Atari
from deep_q_agent.agent_logger import DeepQLog
from deep_q_network.deep_q_network import DeepQNetwork


class DeepQAgent(object):
    def __init__(self,
                 game: Atari,
                 model: DeepQNetwork,
                 gamma: float,
                 alpha: float,
                 epsilon: float,
                 replay_memory_size: int,
                 target_update_horizen: int,

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
            target_update_horizen (int): [description]
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
        self.replay_memory = deque(replay_memory_size)
        # Set the replay memory size (defaulted to replay_memory_size / 50)
        self.min_replay_memory_size = replay_memory_size / 50 if replay_memory_size is None else min_replay_memory_size

        # The number of episodes that pass before we update the target model
        self.target_update_horizen = target_update_horizen
        # Private value used to tra
        self.__target_update_counter = 0

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
            k = round(np.log(self.min_epsilon) / np.log(self.epsilon_decay))
            print(f" - Minimum epsilon ({self.min_epsilon}) will be reached in {k} steps...")
            print(f" - Total steps are {self.game.max_steps * max_episodes}")

        for episode in tqdm(range(1, max_episodes + 1)):

            # Set episode values
            done = False
            current_step = 1
            current_state = self.game.reset()
            total_episode_reward = 0

            while not done:
                # Choose an action and step with it
                action = self.get_action(current_state)
                new_state, reward, done = self.game.step(action)
                # Add to the total reward for the episode
                total_episode_reward += reward

                # Render the game if we want to
                if show_game:
                    self.game.render()

                # Update the replay memory with a new item
                memory_item = (current_state, action, reward, new_state, done)
                self.__update_replay_memory(memory_item)

                # Train the main network (possible update target)
                self.__train_network(done)
                current_state = new_state
                current_step += 1

                # update the logger
                log_entry = f"episode-{episode}"
                self.logger(log_name=log_entry,
                            step=current_step,
                            total_reward=total_episode_reward,
                            epsilon=self.current_epsilon)

                # Decay current epsilon
                self.__eps_exponential_decay()

    def get_action(self, state: np.array) -> int:
        """[summary]

        Args:
            state (np.array): [description]

        Returns:
            int: [description]
        """
        if np.random.random() > self.current_epsilon:
            return np.argmax(self.__get_pred_main(state))
        return np.random.randint(0, self.game.action_space_size)

    def __train_network(self, terminal_state):
        """[summary]

        Args:
            terminal_state ([type]): [description]
        """

        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Get a mini batch from the memory
        batch = np.random.choice(self.replay_memory, self.main_model.batch_size)

        states = batch[:, 0]
        current_qs = self.__get_pred_main(states)

        # Get all of the next states
        states_prime = batch[:, 3]
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

        # After each episode, update the target
        if terminal_state:
            self.__target_update_counter += 1

        # Once we have completed enough episodes, reset the counter and update the target
        if self.target_update_horizen < self.__target_update_counter:
            self.target_model.set_weights(self.main_model)
            self.__target_update_counter = 0

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
