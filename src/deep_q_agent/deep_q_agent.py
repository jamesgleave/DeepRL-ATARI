import numpy as np
from collections import deque
from atari_env.atari import Atari
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
                 epsilon_decay_func: str = "exp",
                 name = "deep_q"
                 ):
        super().__init__()
        
        # Save the main model
        self.main_model = model
        # Create a clone of the main model for the target model
        self.target_model = model.clone() if target_model is None else target_model
        # Create a deque with size replay_memory_size
        self.replay_memory = deque(replay_memory_size)
        # Set the replay memory size (defaulted to replay_memory_size / 50)
        self.min_replay_memory_size = replay_memory_size / 50 if replay_memory_size is None else replay_memory_size
        
        # The number of episodes that pass before we update the target model
        self.target_update_horizen = target_update_horizen
        # Private value used to tra
        self.__target_update_counter = 0
        
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        # ...
        
    def __update_replay_memory(self, transition: tuple):
        """[summary]

        Args:
            transition (tuple): tuple(state, action, reward, state_prime, action, done)
        """
        self.replay_memory.append(transition)
        
    def __get_pred_main(self, state: np.array) -> np.array:
        """[summary]

        Args:
            state (np.array): np.array[tuple(state, action, reward, state_prime, action)]

        Raises:
            NotImplementedError: [description]

        Returns:
            np.array: [description]
        """
        # self.main_model.predict(state)
        raise NotImplementedError
    
    def __get_pred_target(self, state: np.array) -> np.array:
        """[summary]

        Args:
            state (np.array): np.array[tuple(state, action, reward, state_prime, action)]

        Raises:
            NotImplementedError: [description]

        Returns:
            np.array: [description]
        """
        # self.main_model.predict(state)
        raise NotImplementedError
    
    def train(self, terminal_state, step):
        
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        
        # Get a mini batch from the memory
        batch = np.random.choice(self.replay_memory, self.main_model.batch_size)
        
        states = batch[:, 0]
        current_qs = self.__get_pred_main(states)

        # Get all of the next states
        states_prime = batch[:, 3]
        future_qs = self.__get_pred_target(states)
        
        # Create out
        x, y = [], []
        for index, (current_state, action, reward, new_current_state, new_action, done) in enumerate(batch):
            
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
            self.

    