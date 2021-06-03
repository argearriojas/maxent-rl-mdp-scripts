import numpy as np
from scipy.sparse import csr_matrix

from gym.wrappers import TimeLimit
from environment import MAPS, ModifiedFrozenLake


def get_maze(maze='8x8zigzag', variant='walls'):
    desc = np.array(MAPS[maze], dtype='c')

    # make sure to normalize to walls
    desc[desc == b'H'] = b'W'

    # then apply the variant
    if variant == 'nails':
        desc[desc == b'W'] = b'N'
    elif variant == 'holes':
        desc[desc == b'W'] = b'H'

    return desc


def get_zigzag_maze(maze_size=1):
    tile1 = np.array([
        'FFFFFFW',
        'FFFWFFW',
        'WWWWFFW',
        'WFFFFFW',
        'FFWFFFF',
        'FFWWWWW',
        'FFFFFFW',
        'FFFWFFW',
        'WWWWFFW',
        'WFFFFFW',
        'FFWFFFF',
        'FFWWWWW',
        'FFFFFFF',
        'FFFFFFF',
    ], dtype='c')
    tile2 = np.flip(tile1, axis=0)
    nr, nc = tile1.shape
    tile = np.empty((nr, nc*2), dtype='c')
    tile[:, :nc] = tile1
    tile[:, nc:] = tile2

    nr, nc = tile.shape
    desc = np.empty((nr, nc*maze_size), dtype='c')
    
    for i in range(maze_size):
        desc[:, i*nc:(i+1)*nc] = tile

    desc[0, 0] = b'S'
    desc[1, -2] = b'G'
    return desc[:, :-1]


def get_environment(desc, max_steps=1000, stay_at_goal_prob=0.):
    env_src = ModifiedFrozenLake(
        n_action=4, hot_edges=False, max_reward=0., min_reward=-1.5, step_penalization=1,
        desc=desc, never_done=True, cyclic_mode=True, slippery=0, goal_attractor=stay_at_goal_prob)
    env = TimeLimit(env_src, max_episode_steps=max_steps)

    return env


def get_transition_dynamics(env, absorbing=False):

    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                if absorbing and r_j == 0:
                    s_j = s_i
                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j)
                

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    assert (dynamics.sum(axis=0) == 1).all()

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)), shape=shape)

    return dynamics, rewards


def get_mdp_generator(env, transition_dynamics, policy):
    td_coo = transition_dynamics.tocoo()

    rows, cols, data = [], [], []
    for s_j, col, prob in zip(td_coo.row, td_coo.col, td_coo.data):
        for a_j in range(env.nA):
            row = s_j * env.nA + a_j
            rows.append(row)
            cols.append(col)
            data.append(prob * policy[s_j, a_j])

    nrow = ncol = env.nS * env.nA
    shape = (nrow, ncol)
    mdp_generator = csr_matrix((data, (rows, cols)), shape=shape)

    return mdp_generator
