import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os

from scipy.ndimage.filters import gaussian_filter1d


"""
James Gleave
"""



def display(filepath: str,
            x: str,
            y: str,
            smoothing: float,
            palette=None,
            background_alpha=1,
            x_y_label: tuple = None,
            title: str = None,
            show: bool=False):
    """

    Display a nice plot!

    Args:
        filepath (str): path to json or csv
        x (str): the values on the x axis
        y (str): the values on the y axis
        smoothing (float): smoothing factor
        palette ([type], optional): seaborn palette. Defaults to None.
        background_alpha (int, optional): alpha value of the filled bounds. Defaults to 1.
        x_y_label (tuple, optional): labels for x and y axis. Defaults to None.
        title (str, optional): title for the plot. Defaults to None.
        show (bool, optional): whether or now to call plt.show(). Defaults to False.

    Raises:
        ValueError: If a file is not a csv or json it will cause issues

    Returns:
        [ax]: a plt axes instance
    """

    # Grab the filename and extension
    filename, file_extension = os.path.splitext(filepath)

    # Check if we have been passed a json or csv
    if file_extension == ".json":
        with open(filepath, 'r') as f:
            data = json.load(f)
        data = pd.DataFrame(data)
        data.fillna(0, inplace=True)

    elif file_extension == ".csv":
        data = pd.read_csv(filepath)
    else:
        raise ValueError("File must be json or csv")

    if x == "index":
        x = list(range(len(data)))

    # Get STD
    std = data[y].std(0)

    # Filter and smooth the y
    smoothed_y = gaussian_filter1d(data[y][:], sigma=smoothing)

    # Set the theme and pallet
    sns.set_theme(style="dark")
    sns.set_style("darkgrid", {"axes.facecolor": ".75"})
    if palette is None:
        palette = sns.color_palette("mako_r", 6)

    # Calculate the upper and lower bound
    chunk_size = 150
    upper_bound = [smoothed_y[0]]
    lower_bound = [smoothed_y[0]]
    ys = data[y].tolist()
    for i in range(1, len(data[y])-1):
        # Get the max and min of the chunk region
        region = ys[i:i+chunk_size+1]
        upper_bound.append(max(region))
        lower_bound.append(min(region))

    upper_bound.append(smoothed_y[-1])
    lower_bound.append(smoothed_y[-1])

    # Smooth out the values
    smoothed_upper_bound = gaussian_filter1d(upper_bound, sigma=smoothing)
    smoothed_lower_bound = gaussian_filter1d(lower_bound, sigma=smoothing)

    # Best fit line
    smoothed_ax = sns.lineplot(x=x, y=smoothed_y, data=data, palette=palette)

    smoothed_ax.fill_between(x, smoothed_upper_bound, smoothed_lower_bound, color=palette[1], alpha=background_alpha)
    sns.lineplot(x=x, y=smoothed_upper_bound, data=data, color=palette[2], alpha=background_alpha, ax=smoothed_ax)
    sns.lineplot(x=x, y=smoothed_lower_bound, data=data, color=palette[2], alpha=background_alpha, ax=smoothed_ax)

    if x_y_label:
        xlabel, ylabel = x_y_label
        smoothed_ax.set(xlabel=xlabel, ylabel=ylabel)

    if title:
        smoothed_ax.set_title(title)

    # Show the stuff
    if show:
        plt.show()

    return smoothed_ax



json_file = "/Users/martingleave/Desktop/School Work/UNIVERSITY/fourth_year/first_sem/CISC474/Projects/DeepRL-ATARI/src/extras/logs/dqn_Breakout-v0_log-paper.json"
csv_file = "/Users/martingleave/Desktop/School Work/UNIVERSITY/fourth_year/first_sem/CISC474/Projects/DeepRL-ATARI/src/extras/deep_q_agent.csv"
# a1 = display(csv_file, "index", "episode_reward", 50, background_alpha=0.75, x_y_label=("Episode", "Reward"), title="Keras-rl", show=True)
a1 = display(json_file, "index", "nb_episode_steps", 25, background_alpha=0.75, x_y_label=("Episode", "Steps Per Episode"), title="Deep Q Learning: Paper Implementation", show=True)
