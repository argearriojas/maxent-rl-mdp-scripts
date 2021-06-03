# %%
import numpy as np
import matplotlib.pyplot as plt

from solvers import u_solver
from utils import get_maze, get_environment, make_greedy

from visualization import plot_dist

# %%
# Take the variant with holes, to avoid the issue of getting unwanted traps between walls
desc = get_maze(maze='Tiomkin2wider', variant='holes')
# Using a reducible MDP
env = get_environment(desc, max_steps=1000, stay_at_goal_prob=1.)

# %%
beta = 1
u_sol = u_solver(env, steps=env._max_episode_steps, beta=beta, tolerance=0.01)

plot_dist(desc,
    u_sol.get('policy'),
    make_greedy(u_sol.get('policy')),
    titles=['Boltzmann policy', 'Greedy policy']
)
