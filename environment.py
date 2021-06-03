"""Customized Frozen lake enviroment"""
import sys
from contextlib import closing

from gym.envs.toy_text import discrete
from gym import utils
import numpy as np

from six import StringIO

class ModifiedFrozenLake(discrete.DiscreteEnv):
    """Customized version of gym environment Frozen Lake"""

    def __init__(
            self, desc=None, map_name="4x4", slippery=0, n_action=4,
            cyclic_mode=False, hot_edges=False, never_done=False, 
            goal_attractor=False,
            max_reward=0., min_reward=-1., step_penalization=0.):

        goal_attractor = float(goal_attractor)

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (min_reward, max_reward)

        if n_action == 2:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = None
        elif n_action == 3:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = 2
        elif n_action in [8, 9]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_leftdown = 4
            a_downright = 5
            a_rightup = 6
            a_upleft = 7
            a_stay = 8
        elif n_action in [4, 5]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_stay = 4
        else:
            raise NotImplementedError(f'n_action:{n_action}')

        all_actions = set(list(range(n_action)))
        self.n_state = n_state = nrow * ncol
        self.n_action = n_action

        if step_penalization is None:
            step_penalization = 1. / n_state

        isd = np.array(desc == b'S').astype('float64').ravel()
        if isd.sum() == 0:
            isd = np.array(desc == b'F').astype('float64').ravel()
        isd /= isd.sum()
        self.isd = isd

        transition_dynamics = {s : {a : [] for a in all_actions}
                               for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, action):
            if action == a_left:
                col = max(col - 1, 0)
            elif action == a_down:
                row = min(row + 1, nrow - 1)
            elif action == a_right:
                col = min(col + 1, ncol - 1)
            elif action == a_up:
                row = max(row - 1, 0)
            elif action == a_leftdown:
                col = max(col - 1, 0)
                row = min(row + 1, nrow - 1)
            elif action == a_downright:
                row = min(row + 1, nrow - 1)
                col = min(col + 1, ncol - 1)
            elif action == a_rightup:
                col = min(col + 1, ncol - 1)
                row = max(row - 1, 0)
            elif action == a_upleft:
                row = max(row - 1, 0)
                col = max(col - 1, 0)
            elif action == a_stay:
                pass
            else:
                raise ValueError("Invalid action provided")
            return (row, col)

        def compute_transition_dynamics(action_set, action_intended):

            restart = letter in b'H' and cyclic_mode

            diagonal_mode = n_action in [8, 9]

            for action_executed in action_set:
                prob = 1. / (len(action_set) + slippery)
                prob = (slippery + 1) * prob if action_executed == action_intended else prob

                if not restart:
                    newrow, newcol = inc(row, col, action_executed)
                    newletter = desc[newrow, newcol]
                    newstate = to_s(newrow, newcol)
                    edge_hit = action_executed != a_stay and state == newstate
                    got_burned = edge_hit and hot_edges

                    if letter == b'G' and goal_attractor:
                        newletter = letter
                        newstate = state

                    wall_hit = newletter == b'W'
                    if wall_hit:
                        newletter = letter
                        newstate = state
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'
                    ate_candy = letter == b'C'
                    step_nail = letter == b'N'

                    numbers = b'0123456789'
                    newpotential = np.int(newletter) if newletter in numbers else 0

                    # making diagonal steps costlier makes the agent to avoid them at all,
                    # even if the overall cost of trajectories would be less.
                    # can't yet explain why
                    is_diagonal_step = diagonal_mode and action_executed in [4, 5, 6, 7]
                    diagonal_adjust = 1.4 if is_diagonal_step else 1.

                    done = is_in_goal #or is_in_hole or got_burned
                    rew = 0.
                    rew -= step_penalization * (1. - done) * diagonal_adjust
                    rew -= step_penalization * newpotential / 10.
                    rew -= step_nail * step_penalization / 2.
                    rew += ate_candy * step_penalization / 2.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward
                    rew += got_burned * min_reward

                    done = done and not never_done
                    if goal_attractor > 0 and is_in_goal:
                        sat_li.append((prob * goal_attractor, newstate, rew, done))
                        for ini_state, start_prob in enumerate(isd):
                            if start_prob > 0.0:
                                sat_li.append((start_prob * prob * (1 - goal_attractor), ini_state, rew, done))
                    else:
                        sat_li.append((prob, newstate, rew, done))
                else:
                    done = False
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'

                    rew = 0.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward

                    for ini_state, start_prob in enumerate(isd):
                        if start_prob > 0.0:
                            sat_li.append((start_prob * prob, ini_state, rew, done))

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)

                for action_intended in all_actions:
                    sat_li = transition_dynamics[state][action_intended]
                    letter = desc[row, col]

                    if slippery != 0:
                        if action_intended == a_left:
                            action_set = set([a_left, a_down, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_down:
                            action_set = set([a_left, a_down, a_right])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_right:
                            action_set = set([a_down, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_up:
                            action_set = set([a_left, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_stay:
                            action_set = set([a_stay])
                        else:
                            raise ValueError(f"encountered undefined action: {action_intended}")

                    else:
                        action_set = set([action_intended])

                    compute_transition_dynamics(action_set, action_intended)

        super(ModifiedFrozenLake, self).__init__(n_state, n_action, transition_dynamics, isd)


    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        else:
            return None


MAPS = {
    "2x9ridge": [
        "FFFFFFFFF",
        "FSFHHHFGF"
    ],
    "3x2uturn": [
        "SF",
        "HF",
        "GF",
    ],
    "3x3uturn": [
        "SFF",
        "HHF",
        "GFF",
    ],
    "3x9ridge": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FSFHFHFGF"
    ],
    "5x4uturn": [
        "SFFF",
        "FFFF",
        "HHFF",
        "FFFF",
        "GFFF",
    ],
    "3x4uturn": [
        "SFFF",
        "HHHF",
        "GFFF",
    ],
    "3x5uturn": [
        "SFFFF",
        "HHHHF",
        "GFFFF",
    ],
    "3x6uturn": [
        "SFFFFF",
        "HHHHHF",
        "GFFFFF",
    ],
    "3x7uturn": [
        "SFFFFFF",
        "HHHHHHF",
        "GFFFFFF",
    ],
    "3x12ridge": [
        "FFFHHHHHHFFF",
        "FSFFFFFFFFGF",
        "FFFHHHHHHFFF"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "4x4empty": [
        "FFFF",
        "FSFF",
        "FFGF",
        "FFFF"
    ],
    "5x5empty": [
        "FFFFF",
        "FSFFF",
        "FFFFF",
        "FFFGF",
        "FFFFF"
    ],
    "5x12ridge": [
        "FFFHHHHHHFFF",
        "FFFFFFFFFFFF",
        "FSFFFFFFFFGF",
        "FFFFFFFFFFFF",
        "FFFHHHHHHFFF"
    ],
    "6x6empty": [
        "FFFFFF",
        "FSFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFGF",
        "FFFFFF"
    ],
    "7x7wall": [
        "FFFFFFF",
        "FFFSFFF",
        "FFFFFFF",
        "FFWWWFF",
        "FFFFFFF",
        "FFFGFFF",
        "FFFFFFF"
    ],
    "7x7holes": [
        "FFFFFFF",
        "FFFSFFF",
        "FFFFFFF",
        "FFHHHFF",
        "FFFFFFF",
        "FFFGFFF",
        "FFFFFFF"
    ],
    "7x7wall-mod": [
        "FFFFFFF",
        "FFFSFFF",
        "FFFFFFF",
        "FFWWWFF",
        "FFFCFFF",
        "FFFGFFF",
        "FFFFFFF"
    ],
    "7x8wall": [
        "FFFFFFFF",
        "FFFSFFFF",
        "FFFFFFFF",
        "FFWWWWFF",
        "FFFFFFFF",
        "FFFFGFFF",
        "FFFFFFFF"
    ],
    "7x7zigzag": [
        "FFFFFFF",
        "FSFFFFF",
        "WWWWWFF",
        "FFFFFFF",
        "FFWWWWW",
        "FFFFFGF",
        "FFFFFFF"
    ],
    "8x8empty": [
        "FFFFFFFF",
        "FSFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFGF",
        "FFFFFFFF"
    ],
    "9x9empty": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFSFFFFF",
        "FFFFFFFFF",
        "FFFFFGFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF"
    ],
    "9x9wall": [
        "FFFFFFFFF",
        "FFFFSFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFWWWWWFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFGFFFF",
        "FFFFFFFFF"
    ],
    "9x9zigzag": [
        "FFFFFFFFF",
        "FSFFFFFFF",
        "WWWWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFWWWWWW",
        "FFFFFFFGF",
        "FFFFFFFFF"
    ],
    "9x9zigzag2h": [
        "FFFFFFFFF",
        "FSFFFFFFF",
        "WWWWWWFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFWWWWWW",
        "FFFFFFFGF",
        "FFFFFFFFF"
    ],
    "8x8zigzag": [
        "FFFFFFFF",
        "FSFFFFFF",
        "WWWWWFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFWWWWW",
        "FFFFFFGF",
        "FFFFFFFF"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "8x8a_relearn":[ # http://alexge233.github.io/relearn/
        "SFFFFFFF",
        "FFFFWWFF",
        "FFFWFFFF",
        "WWWFFFFF",
        "FFFFFWFF",
        "FFFWWWFF",
        "FFWFFFFF",
        "GFWFFFFF",
    ],
    "8x8b_relearn":[ # http://alexge233.github.io/relearn/
        "SFFFFFFF",
        "FFFFWWFF",
        "HHHWFFFF",
        "WWWFFFFF",
        "FFFFFWFF",
        "FFFWWWFF",
        "FFWFFFFF",
        "GFWHHHHH",
    ],
    "5x15empty": [
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
    ],
    "10x10empty": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "9x9channel": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFSFFFFFF",
        "FFWHFHWFF",
        "FFWHFHWFF",
        "FFWHFHWFF",
        "FFFFFFGFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
    ],
    "10x10channel": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFFFFFF",
        "FFFFFFFFFF",
        "FFWHFFHWFF",
        "FFWHFFHWFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "8x8candy": [

        "FFFFFFFF",
        "FSFFFFFF",
        "FFFFFCFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFFF",
        "FFFFFFGF",
        "FFFFFFFF",
    ],
    "10x10candy": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFCFCFF",
        "FFFFFFFFFF",
        "FFFFFFFCFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFFFFFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "10x10candy-x2": [
        "FFFFFFFFFF",
        "FFFFFFFFFF",
        "FFSFFCFCFF",
        "FFFFFFFFFF",
        "FFFFFFFCFF",
        "FFCFFFFFFF",
        "FFFFFFFFFF",
        "FFCFCFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "10x10candy-x2-nails": [
        "FFFFFFFFFF",
        "FFFFNFNFNF",
        "FFSFFCFCFF",
        "FFFFNFNFNF",
        "FFFFFFFCFF",
        "FFCFFFNFNF",
        "FFFFFFFFFF",
        "FFCFCFFGFF",
        "FFFFFFFFFF",
        "FFFFFFFFFF",
    ],
    "11x11gradient": [
        "22222223432",
        "21112234543",
        "21012345654",
        "21112234543",
        "22222123432",
        "22221S12322",
        "22222122222",
        "33333333333",
        "44444444444",
        "55555555555",
        "66666666666",
    ],
    "11x11gradient-x2": [
        "98489444444",
        "84248445754",
        "42024447974",
        "84244445754",
        "98444244444",
        "44442S24444",
        "44444244444",
        "55555555555",
        "66666666666",
        "77777777777",
        "88888888888",
    ],
    "11x11zigzag": [
        "FFFFFFFFFFF",
        "FSFFFFFFFFF",
        "FFFFFFFFFFF",
        "WWWWWWWWFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFWWWWWWWW",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "11x11empty": [
        "FFFFFFFFFFF",
        "FSFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "11x11wall": [
        "FFFFFFFFFFF",
        "FFFFFSFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFWWWWWWWFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFGFFFFF",
        "FFFFFFFFFFF",
    ],
    "11x11dzigzag": [
        "FFFFFFFWFFF",
        "FSFFFFWFFFF",
        "FFFFFWFFFFF",
        "FFFFWFFFFFF",
        "FFFWFFFFFFF",
        "FFWFFFFFWFF",
        "FFFFFFFWFFF",
        "FFFFFFWFFFF",
        "FFFFFWFFFFF",
        "FFFFWFFFFGF",
        "FFFWFFFFFFF",
    ],
    "5x11ridgex2": [
        "FFFHHHHHFFF",
        "FFFFFFFFFFF",
        "FSFHHHHHFGF",
        "FFFFFFFFFFF",
        "FFFHHHHHFFF",
    ],
    "7x11ridgex4": [
        "FFFFFFFFFFF",
        "FFFHHHHHFFF",
        "FFFFFFFFFFF",
        "FSFHHHHHFGF",
        "FFFFFFFFFFF",
        "FFFHHHHHFFF",
        "FFFFFFFFFFF",
    ],
    "7x11uturn": [
        "FFFFFFFFFFF",
        "FFFFFFFFFSF",
        "FFFFFFFFFFF",
        "FFFFWWWWWWW",
        "FFFFFFFFFFF",
        "FFFFFFFFFGF",
        "FFFFFFFFFFF",
    ],
    "7x7hot_uturn": [
        "FFFFFFF",
        "FFFFFSF",
        "FFFFFFF",
        "FFFHWWW",
        "FFFFFFF",
        "FFFFFGF",
        "FFFFFFF",
    ],
    "9x9ridgex4": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FFFHHHFFF",
        "FFFFFFFFF",
        "FSFHHHFGF",
        "FFFFFFFFF",
        "FFFHHHFFF",
        "FFFFFFFFF",
        "FFFFFFFFF",
    ],
    "15x15empty": [
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
    ],
    "15x15zigzag": [
        "FFFFFFFFFFFFFFF",
        "FSFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "WWWWWWWWWWWFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFWWWWWWWWWWW",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGF",
        "FFFFFFFFFFFFFFF",
    ],
    "16x16empty": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16candy": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFCFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFCFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16candyx2": [
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFCFFFFFFFCFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFCFFFFFFFCFFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFCFFFFFFFFF",
        "FFFFFFFFFCFFFFFF",
        "FFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFF",
    ],
    "16x16bigS" : [
        "WWWWWWWWWWWWWWWW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFSFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFWWWWWWWWWWW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WWWWWWWWWWWFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WFFGFFFFFFFFFFFW",
        "WFFFFFFFFFFFFFFW",
        "WWWWWWWWWWWWWWWW",
    ],
    "5x17empty": [
        "FFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFF",
    ],
    "5x24empty": [
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFSFFFFFFFFFFFFFFFFFFGFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "7x32empty": [
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFSFFFFFFFFFFFFFFFFFFFFFFGFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "15x45": [
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFSFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF",
    ],
    "17x17center": [
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFGFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFF',
    ],
    "5x15zigzag": [
        'FFSFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFGFF',
    ],
    "8x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "11x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'HHHHHHHHHHHHFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFHHHHHHHHHHHH',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "23x15zigzag": [
        'FFFFFFFFFFFFFFF',
        'FFSFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'WWWWWWWWWWWWFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFWWWWWWWWWWWW',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'WWWWWWWWWWWWFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFWWWWWWWWWWWW',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFGFF',
        'FFFFFFFFFFFFFFF',
    ],
    "15x15mixed": [
        'FFFFFFFSFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFWWWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'WWWWFFFWWWWWFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFWWWWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFGFFFFFFF',
    ],
    "11x11mixed": [
        'FFFFFSFFFFF',
        'FFFFFWFFFFF',
        'FFFFWWFFFFF',
        'FFFFFWFFFFF',
        'FFFFFWFFFFF',
        'WWWFFWWWWFF',
        'FFFFFWFFFFF',
        'FFFFFWFFFFF',
        'FFFWWWFFFFF',
        'FFFFFWFFFFF',
        'FFFFFGFFFFF',
    ],
    "9x9mixed": [
        'FFFWFWFFF',
        'FFFFSFFFF',
        'FFFWFWFFF',
        'FFFWWWFFF',
        'WWFFWWWWF',
        'FFFFWFFFF',
        'FFWWWFFFF',
        'FFFFGFFFF',
        'FFFFWFFFF',
    ],
    "9x15asymmetric": [
        'FFFFFFWFWFFFFFF',
        'FWWFWFFSFFWWWWF',
        'FWFWFWWFWWWWWWF',
        'FFFFFFFWFFFFFFF',
        'FWFWFWFWFWWWWWW',
        'WFWFWWFWFWWWWWW',
        'FFFFFFFWFFFFFFF',
        'FWWWWWWWWWWWWWF',
        'FFFFFFGFFFFFFFF',
    ],
    "10x11asymmetric": [
        'FFFFWFWFFFF',
        'FFWFFSFFWWF',
        'FWFWWFWWWWF',
        'FFFFFWFFFFF',
        'FWFWFWFWWWW',
        'WFWFFWFWWWW',
        'FFFFFWFFFFF',
        'FWFWFWWWWWF',
        'FFWFWWWWWWF',
        'FFFFGFFFFFF',
    ],
    "9x10asymmetric": [
        'FWFWWFWWWW',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWFWWWWWF',
        'WFFFGFFFFF',
        'FFWFWWWWWW',
    ],
    "9x10asymmetric-00": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWFWWWWWF',
        'WFFFGFFFFF',
        'FFWFWWWWWW',
    ],
    "9x10asymmetric-01": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-02": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWWWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-03": [
        'WWWWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-04": [
        'WWFWWFWWWF',
        'WFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-05": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'WFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WFWFWWWWWW',
    ],
    "9x10asymmetric-09": [
        'WWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'WFWWFSFWWF',
        'FFFWFFFWWF',
        'WFWFWWWWWF',
        'WFFFGFFFFF',
        'WWWFWWWWWW',
    ],
    "9x10asymmetric-20": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWFWWWWWF',
        'WFFFGFFFFF',
        'FFWWWWWWWW',
    ],
    "9x10asymmetric-21": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'FFWWWWWWWW',
    ],
    "9x10asymmetric-22": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFFWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-23": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFFWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-24": [
        'FWFWWFWWWF',
        'FFFFFFFFFF',
        'WFWFWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-25": [
        'WWFWWFWWWF',
        'WFFFFFFFFF',
        'WFWFWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-26": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWFWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-27": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWWWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'FFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-28": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWWWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'FFWWFFFWWF',
        'WFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "9x10asymmetric-29": [
        'WWWWWFWWWF',
        'WFFFFFFFFF',
        'WFWWWFWWWF',
        'FFWWFFFWWF',
        'FFWWFSFWWF',
        'WFWWFFFWWF',
        'WFWWWWWWWF',
        'WFFFGFFFFF',
        'WWWWWWWWWW',
    ],
    "empty-quad" :[
        'FGFWFFFFF',
        'FFFWFFFGF',
        'FFFWFFFFF',
        'FFFWFWWWW',
        'FFFFSFFFF',
        'WWWWFWFFF',
        'FFFFFWFGF',
        'FFFGFWFFF',
        'FFFFFWFFF',
    ],
    "Todorov3A":[
        'FFFFF',
        'FFWFF',
        'FWWWF',
        'FFWFF',
        'FFFFG',
    ],
    "Todorov3B":[
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFF',
        'FFFFFFFFFFWWWFF',
        'FFFFFFFFFFFWFFF',
        'FFFFFFFWFFFWFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFFFFFWFFFFFFF',
        'FFFWWWWWFFFFFFF',
        'FFFWFFFFFFWFFFF',
        'FFFWFFFFFFWFFFF',
        'FFFWFFFFWWWFFFF',
        'FFFFFFFFWFFFFFF',
        'FFFFFFFFFFFFFFF',
    ],
    "Tiomkin2":[
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
        'WSFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWWWWWWWWWFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWFWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFGFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
    ],
    "Tiomkin2wider":[
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
        'WSFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWWWWWWWWWFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWFWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFGFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
    ],
    "Tiomkin2zigzag":[
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
        'WSFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFWWWWWWWWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFGFW',
        'WFFFFFFFFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWWWWWWWWW',
    ],
    '20x20burger': [
        'WWWWWWWWWWWWWWWWWWWW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFWWWWWWFFFFFFW',
        'WFFFFFWFFFFFFWFFFFFW',
        'WFFFFWFFFFFFFFWFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFWWWWWWWWWWFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WFFFFWWWWWWWWWWFFFFW',
        'WFSFFFFFFFFFFFFFFGFW',
        'WFFFFWWWWWWWWWWFFFFW',
        'WFFFFFFFFFFFFFFFFFFW',
        'WWWWWWWWWWWWWWWWWWWW',
    ],
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0,0))
        while frontier:
            r, c = frontier.pop()
            if not (r,c) in discovered:
                discovered.add((r,c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if res[r_new][c_new] not in '#H':
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]

