# %%
import numpy as np
import matplotlib.pyplot as plt

from solvers import u_solver

from utils import get_maze, get_environment, make_greedy

from visualization import plot_dist
# %%
# Setup the layout and the environment. Using irreducible MDP
desc = get_maze(maze='2x9ridge', variant='holes').T
desc = np.flip(desc, axis=1)
env = get_environment(desc, max_steps=100, stay_at_goal_prob=0.)

# %%
beta = 5
u_sol = u_solver(env, steps=env._max_episode_steps, beta=beta, tolerance=0.01)

plot_dist(desc,
    u_sol.get('policy'),
    make_greedy(u_sol.get('policy')),
    titles=['Boltzmann policy', 'Greedy policy']
)

# %%
perron_eigenvalue = u_sol.get('eigenvalue')
offset = np.log(perron_eigenvalue) / beta

V_soft_values = u_sol.get('V')

plt.figure(figsize=(8, 5))
v_vectr = V_soft_values / env._max_episode_steps
plt.scatter(*zip(*enumerate(v_vectr)), label='Soft V / N')
plt.hlines(offset, 0, env.nS, linestyle='dashed', color='black', label=r'- $\theta$ / $\beta$')
ymin, ymax = offset -1.2/ env._max_episode_steps, offset + 1.2/ env._max_episode_steps
plt.vlines([i * desc.shape[1] - 0.5 for i in range(1, desc.shape[0])], ymin, ymax, color='gray', alpha=0.2)
plt.ylim(ymin, ymax)
plt.legend()
plt.ylabel('State values')
plt.xlabel('State label')
plt.show()

# %%
plt.figure(figsize=(8, 5))
q_table = u_sol.get('Q') / env._max_episode_steps
plt.scatter(*zip(*enumerate(q_table[:, 0])), alpha=0.8, label='Soft Q left / N')
plt.scatter(*zip(*enumerate(q_table[:, 1])), alpha=0.8, label='Soft Q down / N')
plt.scatter(*zip(*enumerate(q_table[:, 2])), alpha=0.8, label='Soft Q right / N')
plt.scatter(*zip(*enumerate(q_table[:, 3])), alpha=0.8, label='Soft Q up / N')
plt.hlines(offset, 0, env.nS, linestyle='dashed', color='black', label='Constant offset')
ymin, ymax = offset - 1.2/ env._max_episode_steps, offset + 1.2/ env._max_episode_steps
plt.vlines([i * desc.shape[1] - 0.5 for i in range(1, desc.shape[0])], ymin, ymax, color='gray', alpha=0.2)
plt.ylim(ymin, ymax)
plt.legend()
plt.ylabel('State values')
plt.xlabel('State label')
plt.show()

# %%
