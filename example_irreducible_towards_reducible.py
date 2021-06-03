# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import get_environment
from solvers import u_solver
from visualization import plot_dist

# %%
desc = np.array([
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFSFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'HHHHHHHHFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFGFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
    'FFFFFFFFFFFFFFF',
], dtype='c')

plot_dist(desc)
# %%
x = np.linspace(0, 1, 50)
for beta in [0.1, 0.2, 0.5, 1, 2]:
    policies = []
    for prob in x:
        env = get_environment(desc, max_steps=1000, stay_at_goal_prob=prob)
        u_sol = u_solver(env, beta=beta, steps=env._max_episode_steps, tolerance=0.01)
        policies.append(u_sol.get('policy').flatten().tolist())
    poli = np.array(policies)
    klwrt1 = (poli * (np.log(poli) - np.log(poli[-1]))).sum(axis=1)
    plt.plot(x, klwrt1, label=fr"$\beta=${beta}")
plt.xlabel('Probability of staying at goal')
plt.ylabel('KL divergence wrt. reducible case')
plt.legend()
plt.show()
# %%
