# DeepRL-ATARI
Original Paper: https://arxiv.org/pdf/1312.5602.pdf

Our Review: https://drive.google.com/file/d/1H-jJGkWA-M2kUMFBSNuFxQU3BJ6rujxg/view?usp=sharing

# About
To stay true to the DeepMind paper, we implemented their Deep Q-learning method with the same convolutional neural network (CNN) architecture for state-value function approximation. Furthermore, the actual agent itself is a Deep Q-learning agent (DQA) that receives the 84x84 images of the Atari game as inputs and utilizes the network to make decisions on what actions to perform.

![Group 7 (1)](https://user-images.githubusercontent.com/14239415/144766160-c314b329-e5d8-4787-979e-e8c55b651241.png)


# Installation requirements
To run all dependencies run the code below
```sh
pip install -r requirements.txt
```
If any issues occur, make sure all these dependencies are installed
```sh
- pip install gym[atari]
- pip install gym[accept-rom-license]
- pip install numpy
- pip install tensorflow
- pip install matplotlib
- pip install opencv-python
- pip install pickle
- pip install tqdm
- pip install seaborn
- pip install scipy
- pip install pandas
```

# Training
To train our model, run the train_brick_breaker.py.
You can configure the agent with different parameters. The list below are the parameters we used during training

Parameters | Configuration |
--- | --- |
game | -
model | -
gamma | 0.99
epsilon| 1
epsilon_decay | 0.9/500_000
replay_memory_size| 500_000
exploration_steps |100_000
target_update_horizon| 10_000
main_model_train_horizon| 4
min_replay_memory_size |32
save_frequency |250

# Evaluation
We have 2 different models that can be evaluated ours and the transfered weights one. They can be run with these calls:
```sh
python3 evaluate_brick_breaker.py  --model {ours,transfer}  [--games GAMES] [--render]
```
For example, to evaluate our model and get the average reward for 5 games while rendering each game:
```sh
python3 evaluate_brick_breaker.py  --model ours  --games 5 --render
```
