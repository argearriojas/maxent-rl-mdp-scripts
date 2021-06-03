# %%
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

from utils import get_maze, get_environment

from solvers import u_solver
from model_free import ULearning, gather_experience

from visualization import plot_dist, draw_paths

# %%
desc = get_maze(maze='7x7holes', variant='holes')
env = get_environment(desc, max_steps=1000, stay_at_goal_prob=0.)
plot_dist(desc)

# %%
beta = 10
u_sol = u_solver(env, steps=env._max_episode_steps, beta=beta)

# %%
target_theta = -np.log(u_sol.get('eigenvalue'))
target_policy = u_sol.get('policy')

# %%
n_repl = 3
alpha_scale = 0.01
decay = 2e6
batch_size = 100

exploration_policy = np.ones((env.nS, env.nA)) / env.nA

agents = [ULearning(env) for _ in range(n_repl)]
results = [dict(step=[], ev=[], kl=[]) for _ in range(n_repl)]

step = 0
# %%
max_it = 50
bar = Bar('Traning', max=max_it)
for it in range(max_it):
    for agent, result in zip(agents, results):
        alpha = decay / (decay + step) * alpha_scale

        sarsa_experience = gather_experience(env, exploration_policy, batch_size=batch_size, n_jobs=4)
        policy = agent.train(exploration_policy, sarsa_experience, alpha, beta, 1)

        kl = - (policy * (np.log(policy) - np.log(target_policy))).sum()
        ev = agent.l_valu

        result['step'].append(step + batch_size)
        result['ev'].append(ev)
        result['kl'].append(kl)
    
    step += batch_size
    bar.next()
bar.finish()
# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes = axes.ravel()
draw_paths(desc, axes[0], desc)
axes[1].hlines(target_theta, 0, step, color='k', linestyle='dashed', label=r'Target $\theta$')
label_set = False
for rpl_id, res in enumerate(results):
    if not label_set:
        label_a = r'Learned $\theta$'
        label_b = 'Policy KL divergence with respect to DP SoftQ'
        label_set = True
    else:
        label_a = label_b = None
    axes[1].plot(res['step'], -np.log(res['ev']), color='C0', alpha=0.5, label=label_a)
    axes[2].plot(res['step'], -np.array(res['kl']), color='C1', alpha=0.5, label=label_b)

axes[1].set_yscale('log')
# axes[1].set_ylim(9.5, 9.7)
axes[1].set_xlabel('Steps trained')
axes[1].legend()

axes[2].set_yscale('log')
axes[2].set_xlabel('Steps trained')
axes[2].legend()

axes[0].get_xaxis().set_visible(False)
axes[0].get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()