# Multi Agent Autonomous On-Demand Free Flight Operations in Urban Air Mobility using Monte Carlo Tree Search

A Python implementation of the algorithm proposed in paper "Multi Agent Autonomous On-Demand Free Flight Operations in Urban Air Mobility using Monte Carlo Tree Search"

A short video demo of this algorithm: https://www.youtube.com/watch?v=Q6AkCCt-9nI&t=

## Requirements

* python 3.6
* numpy
* gym
* collections
* ...


## Getting Started

Make sure that you have the above requirements taken care of, then download all the files. You can run it using

```
python Agent_vertiport.py
```

Optional arguments:

`--save_path` the path where you want to save the output

`--seed` seed of the experiment

`--render` if render the env while running exp


## MCTS Algorithm
The whole MCTS algorithm files are under the directory `MCTS\`

`common.py` defines the general MCTS node class and state class

`nodes_multi.py` defines the MCTS node class and state class specifically for Multi Agent Aircraft Guidance proble, e.g., given current aircraft state and current action, how to decided the next aircraft state

`search_multi.py` describes the search process of MCTS algorithm

## Simulator
The simulator codes are under `Simulators\`

`config_vertiport.py` defines the configurable parameters of the simulator. For example, airspace width/length, number of aircraft, scale (1 pixel = how many meters), conflict/NMAC distance, cruise/max speed of aircraft, heading angle change rate of aircraft, number simulations and search depth of MCTS algorithm, vertiport location, ...

`MultiAircraftVertiportEnv.py` is the main simulator. Some main functions include:

* `__init__()` initilize the simulator by generating vertiports, centralized controller, loading configuration parameters, generating aircraft.

* `reset()` will reset the number of conflicts/NMACs to 0 and reset the aircraft dictionary. Note here all the aircraft objects are stored in the `AircraftDict` class, where we can add/remove aircraft from it, and get aircraft object by id.

* `_get_ob()` will return the current state, which is n by 8 matrix, where n is the number of aircraft. Each aircraft has (x, y, vx, vy, speed, heading, gx, gy) state information.

* `_get_normalized_ob()` will return the normalized current state, which will potential be useful if we want to feed state into a neural network.

* `step()` will return next state, reward, terminal, info given current state and current action. Each aircraft will fly according the given action. We have a clock at each vertiport to decide whether to generate new flight request/aircraft.

* `_terminal_reward()` will return the reward function for current state. This function will check if there is any conflict/NMAC between any two aircraft and update conflict/NMAC number. It will also remove aircraft that reaches goal position and aircraft pair that has NMAC.

* `render()` will visualize all of the current aircraft and vertiport. I used this to generate the demo video.



If you have any questions or comments, don't hesitate to send me an email! I am looking for ways to make this code even more computationally efficient.

Email: xuxiyang@iastate.edu
