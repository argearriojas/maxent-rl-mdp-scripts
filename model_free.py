import itertools
from joblib import Parallel, delayed

import numpy as np

class ULearning:
    def __init__(self, env) -> None:
        self.l_valu = 1.
        self.u_table = np.ones((env.nS, env.nA)) / env.nA / env.nS

    def train(self, expl_poli, sarsa_experience, alpha, beta, gamma):

        l_alpha = alpha / np.prod(self.u_table.shape)

        for (state, action, reward, next_state, next_action), _ in sarsa_experience:

            exp_re = np.exp(reward * beta)

            u_valu = self.u_table[state, action]
            u_next = self.u_table[next_state, next_action]
            with np.errstate(divide='ignore', over='ignore'):
                ratio =  u_next / u_valu

            u_valu = (1. - alpha) * u_valu + alpha * exp_re / self.l_valu * u_next
            self.u_table[state, action] = u_valu

            if ratio != np.inf:
                self.l_valu = (1. - l_alpha) * self.l_valu + l_alpha * exp_re * ratio
                self.l_valu = min(self.l_valu, 1.)

        self.u_table = self.u_table / self.u_table.sum()

        eval_poli = self.u_table * expl_poli
        eval_poli = eval_poli / eval_poli.sum(axis=1).reshape((-1, 1))
        
        return eval_poli


class SARSA:
    def __init__(self, env) -> None:
        self.q_table = np.ones((env.nS, env.nA)) / env.nA / env.nS

    def train(self, expl_poli, sarsa_experience, alpha, beta, gamma):
        for (state, action, reward, next_state, next_action), _ in sarsa_experience:
            q_valu = self.q_table[state, action]
            q_next = self.q_table[next_state, next_action]

            q_valu = (1. - alpha) * q_valu + alpha * (reward + gamma * q_next)

            self.q_table[state, action] = q_valu

        delta = (self.q_table.min(axis=1) + self.q_table.max(axis=1)).reshape((-1, 1)) / 2.
        eval_poli = np.exp(beta * (self.q_table - delta))
        eval_poli = eval_poli / eval_poli.sum(axis=1).reshape((-1, 1))
        
        return eval_poli


class QLearning:
    def __init__(self, env) -> None:
        self.q_table = np.ones((env.nS, env.nA)) / env.nA / env.nS

    def train(self, expl_poli, sarsa_experience, alpha, beta, gamma):
        for (state, action, reward, next_state, _), _ in sarsa_experience:

            next_greedy_action = np.argmax(self.q_table[next_state])

            q_valu = self.q_table[state, action]
            q_next = self.q_table[next_state, next_greedy_action]

            q_valu = (1. - alpha) * q_valu + alpha * (reward + gamma * q_next)

            self.q_table[state, action] = q_valu

        eval_poli = np.exp(beta * self.q_table)
        eval_poli = eval_poli / eval_poli.sum(axis=1).reshape((-1, 1))
        
        return eval_poli


class SoftQLearning:
    def __init__(self, env) -> None:
        self.q_table = np.ones((env.nS, env.nA)) / env.nA / env.nS
        self.v_vectr = np.zeros((env.nS, 1))

    def train(self, expl_poli, sarsa_experience, alpha, beta, gamma):
        for (state, action, reward, next_state, _), _ in sarsa_experience:

            q_valu = self.q_table[state, action]
            v_next = self.v_vectr[next_state]

            q_valu = (1. - alpha) * q_valu + alpha * (reward + gamma * v_next)
            self.q_table[state, action] = q_valu

            delta = (self.q_table[state].min() + self.q_table[state].max()) / 2.
            if delta in [np.inf, -np.inf]:
                v_valu = delta
            else:
                with np.errstate(over='ignore'):
                    v_valu = delta + np.log(np.exp(beta * (self.q_table[state] - delta)).sum()) / beta
            self.v_vectr[state, 0] = v_valu

        eval_poli = np.exp(beta * (self.q_table - self.v_vectr))
        eval_poli = eval_poli / eval_poli.sum(axis=1).reshape((-1, 1))
        
        return eval_poli


class ZLearning:
    def __init__(self, env) -> None:
        self.q_table = np.ones((env.nS, env.nA)) / env.nA / env.nS
        self.z_vectr = np.ones((env.nS, ))

    def train(self, expl_poli, sarsa_experience, alpha, beta, gamma):
        for (state, action, reward, next_state, _), _ in sarsa_experience:
            z_valu = self.z_vectr[state]
            z_next = self.z_vectr[next_state]
            exp_re = np.exp(beta * reward)

            z_valu = (1. - alpha) * z_valu + alpha * exp_re * z_next
            self.z_vectr[state] = z_valu

            self.q_table[state, action] = np.log(z_next) / beta

        eval_poli = np.exp(beta * self.q_table)
        eval_poli = eval_poli / eval_poli.sum(axis=1).reshape((-1, 1))

        return eval_poli


######################
## Training related ##
######################

def training_episode(env, training_policy):
    sarsa_experience = []

    state = env.reset()
    action = np.random.choice(env.nA, p=training_policy[state])
    done = False
    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = np.random.choice(env.nA, p=training_policy[next_state])
        sarsa_experience.append(((state, action, reward, next_state, next_action), done))
        state, action = next_state, next_action

    return sarsa_experience

def gather_experience(env, training_policy, batch_size, n_jobs=1):
    if n_jobs > 1:
        split_experience = Parallel(n_jobs=n_jobs, backend='loky')(delayed(training_episode)(env, training_policy) for _ in range(batch_size))
    elif n_jobs == 1:
        split_experience = [training_episode(env, training_policy) for _ in range(batch_size)]

    return list(itertools.chain.from_iterable(split_experience))


########################
## Evaluation related ##
########################

def evaluation_episode(env, evaluation_policy):
    episode_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = np.random.choice(env.nA, p=evaluation_policy[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
    return episode_reward

def evaluate(env, evaluation_policy, n_episodes_eval, n_jobs=1):
    if n_jobs > 1:
        rewards_list = Parallel(n_jobs=n_jobs, backend='loky')(delayed(evaluation_episode)(env, evaluation_policy) for _ in range(n_episodes_eval))
    elif n_jobs == 1:
        rewards_list = [evaluation_episode(env, evaluation_policy) for _ in range(n_episodes_eval)]

    return np.mean(rewards_list)
