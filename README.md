# DeepRL-ATARI
DRIVE: https://drive.google.com/drive/folders/1u_tzZSIlvi1BjVqRYS4ewcKxap9kMaDJ?usp=sharing

PAPER: https://arxiv.org/pdf/1312.5602.pdf

# About
Algorithm and Implementation:
To stay true to the DeepMind paper, we implement their Deep Q-learning method with the same convolutional neural network (CNN) architecture for state-value function approximation. Furthermore, the actual agent itself is a Deep Q-learning agent (DQA) that receives the 84x84 images of the Atari game as inputs and utilizes the network to make decisions on what actions to perform. 

![Group 7 (1)](https://user-images.githubusercontent.com/14239415/144766160-c314b329-e5d8-4787-979e-e8c55b651241.png)


# Installation requirements

```sh
- pip instal gym[atari] 
- pip install numpy
- pip install python3.7.3
- pip install tensorflow
- pip install matplotlib
- pip install opencv
```

# Attain Sample Images and Stacking 
- Run atari.py to get single frames, and stacked frames from the Atari breakout-v0 enviroment.


# Main Model
We have 3 different models that can be run. 
```sh
python evaluate_brick_breaker.py  --model {ours,transfer,kerasrl}
```
