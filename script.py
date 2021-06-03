# %%
import matplotlib.pyplot as plt

from utils import get_maze, get_environment

from solvers import maze_solver, softq_solver, u_solver, z_solver

from visualization import plot_dist
# %%

desc = get_maze()
env = get_environment(desc, stay_prob=1)

solution = maze_solver(env)
softq_sol = softq_solver(env, tolerance=1e-15)

# %%
z_sol = z_solver(env, steps=1000, tolerance=1e-15)
u_sol = u_solver(env, steps=1000, tolerance=0.0001)

x = z_sol.get('V')
y = u_sol.get('V')

plt.scatter(x, y)
plt.show()

# %%
plot_dist(desc, solution.get('policy'), softq_sol.get('policy'), z_sol.get('policy'), u_sol.get('policy'))

# %%
