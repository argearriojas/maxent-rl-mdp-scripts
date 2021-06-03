import numpy as np
import matplotlib.pyplot as plt


def get_layout(desc, show_all=False):
    desc = np.asarray(desc, dtype='c')

    zeros = np.zeros_like(desc, dtype=np.float)
    walls = (desc == b'W').astype(np.float)
    holes = (desc == b'H').astype(np.float)
    candy = (desc == b'C').astype(np.float)
    nails = (desc == b'N').astype(np.float)
    goals = (desc == b'G').astype(np.float)
    start = (desc == b'S').astype(np.float)

    out = np.array([zeros.T, zeros.T, zeros.T]).T

    # Walls: blue
    out += np.array([zeros.T, zeros.T, walls.T]).T
    # Holes: red
    out += np.array([holes.T, zeros.T, zeros.T]).T
    # Candy: orange
    out += np.array([candy.T, candy.T * 0.5, zeros.T]).T
    # Nails: silver
    out += np.array([nails.T, nails.T, nails.T]).T * 0.75

    if show_all:
        # Start: green
        out += np.array([zeros.T, start.T, zeros.T]).T
        # Goal: yellow
        out += np.array([goals.T, goals.T, zeros.T]).T

    return out


def display_map(desc):
    poten = (desc == b'S').astype(np.float)
    poten = np.array([float(l)/10 if l in b'0123456789' else 0.
                      for r in desc for l in r]).reshape(desc.shape)
    out = get_layout(desc, show_all=True)
    # Potential: grayscale
    out += np.array([poten.T, poten.T, poten.T]).T

    plt.imshow(out)
    plt.show()

def draw_paths(desc, axi, paths, title=None, show_values=False):
    if paths is None:
        return
    nrow, ncol = desc.shape
    nsta = nrow * ncol
    zeros = np.zeros_like(desc, dtype=np.float)

    show_whole_maze = (desc.shape == paths.shape) and (desc == paths).all()
    out = get_layout(desc, show_all=show_whole_maze)

    if paths.shape in [desc.shape, (nsta,)] and not show_whole_maze:
        if paths.max() > 0:
            paths = paths / paths.max()
        paths = np.abs(paths.reshape(desc.shape))

        # Path: magenta
        out += np.array([paths.T, zeros.T, paths.T]).T
        out = np.minimum(out, 1.)

    axi.imshow(out)

    # show start and goal points
    axi.scatter(*np.argwhere(desc.T == b'S').T, color='#00FF00')
    axi.scatter(*np.argwhere(desc.T == b'G').T, color='#FFFF00')

    if len(paths.shape) == 2 and paths.shape[0] == nsta:
        # looks like a policy, lets try to illustrate it with arrows
        # axi.scatter(*np.argwhere(desc.T == b'F').T, color='#FFFFFF', s=10)

        nact = paths.shape[1]

        if nact in [2, 3]:
            direction = ['left', 'right', 'stay']
        elif nact in [4, 5]:
            direction = ['left', 'down', 'right', 'up', 'stay']
        elif nact in [8, 9]:
            direction = ['left', 'down', 'right', 'up', 'leftdown', 'downright', 'rightup', 'upleft', 'stay']
        else:
            raise NotImplementedError

        for state, row in enumerate(paths):
            for action, prob in enumerate(row):
                action_str = direction[action]
                if action_str == 'stay':
                    continue
                if action_str == 'left':
                    d_x, d_y = -prob, 0
                if action_str == 'down':
                    d_x, d_y = 0, prob
                if action_str == 'right':
                    d_x, d_y = prob, 0
                if action_str == 'up':
                    d_x, d_y = 0, -prob
                if action_str == 'leftdown':
                    d_x, d_y = -prob / np.sqrt(2), prob / np.sqrt(2)
                if action_str == 'downright':
                    d_x, d_y = prob / np.sqrt(2), prob / np.sqrt(2)
                if action_str == 'rightup':
                    d_x, d_y = prob / np.sqrt(2), -prob / np.sqrt(2)
                if action_str == 'upleft':
                    d_x, d_y = -prob / np.sqrt(2), -prob / np.sqrt(2)
                if desc[state // ncol, state % ncol] not in [b'W', b'G']:
                    axi.arrow(state % ncol, state // ncol, d_x*0.6, d_y*0.6,
                             width=0.001, head_width=0.2*prob, head_length=0.2*prob,
                             fc='w', ec='w')

    elif paths.shape == desc.shape and show_values:
        for i, row in enumerate(paths):
            for j, value in enumerate(row):
                # if desc[state // ncol, state % ncol] not in [b'W', b'G']:
                if value != 0:
                    axi.text(j-0.4, i-0.15, f"{value:.2f}", c='w', fontsize=10.)

    if title is not None:
        axi.set_title(title)

    axi.set_xlim(-0.5, ncol - 0.5)
    axi.set_ylim(nrow - 0.5, -0.5)


def plot_dist(desc, *paths_list, ncols=4, filename=None, titles=None, main_title=None, figsize=None, show_values=False, show_plot=True):
    if len(paths_list) == 0:
        paths_list = [desc]
        axes = [plt.gca()]
    elif len(paths_list) == 1:
        fig = plt.figure(figsize=figsize)
        axes = [plt.gca()]
    elif len(paths_list) > 1:
        n_axes = len(paths_list)

        ncols = min(ncols, n_axes)
        nrows = (n_axes-1)//ncols+1

        figsize = (5*ncols, 5*nrows) if figsize is None else figsize
        fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)
        axes = axes.ravel()
    else:
        raise ValueError("Missing required parameter: path")

    if titles is not None:
        assert type(titles) == list
        assert len(titles) == len(paths_list)
    else:
        titles = [None] * len(paths_list)

    for axi, paths, title in zip(axes, paths_list, titles):
        if paths is None:
            fig.delaxes(axi)
            continue
        draw_paths(desc, axi, paths, title, show_values)

    if main_title is not None:
        plt.suptitle(main_title)
    if filename is not None:
        plt.savefig(filename, dpi=300)
        return plt.gcf()
    elif show_plot:
        plt.show()
    else:
        return plt.gcf()
