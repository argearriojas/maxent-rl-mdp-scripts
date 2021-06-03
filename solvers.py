import numpy as np

from utils import get_transition_dynamics, get_mdp_generator


def maze_solver(env, policy=None, steps=100, update_interval=20, gamma=1, tolerance=1e-2):
    print(f'Solving maze')

    dynamics, rewards = get_transition_dynamics(env)
    rewards = dynamics.multiply(rewards).sum(axis=0)
    policy = np.ones((env.nS, env.nA)) / env.nA if policy is None else policy
    mdp_generator = get_mdp_generator(env, dynamics, policy)

    Qi = np.zeros((1, env.nS * env.nA))

    for i in range(1, steps+1):
        Qj = mdp_generator.T.dot(Qi.T).T
        Qi_k = rewards + gamma * Qj
        err = np.abs(Qi_k - Qi).max()
        Qi = Qi_k

        if i % update_interval == 0:
            policy = Qi.reshape((env.nS, env.nA))
            policy = (policy == policy.max(axis=1)).astype(float)
            policy = policy / policy.sum(axis=1)
            mdp_generator = get_mdp_generator(env, dynamics, policy)

        if err <= tolerance:
            break

    if i == steps:
        print(f'Reached max steps. Err:{err}')
    else:
        print(f"Done in {i} steps")

    Vi = np.multiply(Qi.reshape((env.nS, env.nA)), policy).sum(axis=1)

    return dict(
        Q=np.array(Qi).reshape((env.nS, env.nA)),
        V=np.array(Vi).reshape((env.nS, 1)),
        policy=np.array(policy)
    )


def softq_solver(env, prior_policy=None, steps=100, update_interval=20, beta=1, gamma=1, tolerance=1e-2):

    dynamics, rewards = get_transition_dynamics(env)
    rewards = dynamics.multiply(rewards).sum(axis=0)
    prior_policy = np.ones((env.nS, env.nA)) / env.nA if prior_policy is None else prior_policy
    mdp_generator = get_mdp_generator(env, dynamics, prior_policy)

    Qi = np.zeros((1, env.nS * env.nA))

    for i in range(1, steps+1):
        Qj = np.log(mdp_generator.T.dot(np.exp(beta * Qi.T)).T) / beta
        Qi_k = rewards + gamma * Qj
        err = np.abs(Qi_k - Qi).max()
        Qi = Qi_k

        if err <= tolerance:
            break

    if i == steps:
        print(f'Reached max steps. Err:{err}')
    else:
        print(f"Done in {i} steps")
    
    Vi = np.log(
        np.multiply(prior_policy, np.exp(beta * Qi.reshape((env.nS, env.nA)))).sum(axis=1)
    ) / beta

    policy = np.multiply(prior_policy, np.exp(beta * (Qi.reshape((env.nS, env.nA)) - Vi)))

    return dict(
        Q=np.array(Qi).reshape((env.nS, env.nA)),
        V=np.array(Vi).reshape((env.nS, 1)),
        policy=np.array(policy)
    )


def z_solver(env, prior_policy=None, steps=100, rho=1, tolerance=1e-2):

    if prior_policy is None:
        prior_policy = np.ones((env.nS, env.nA)) / env.nA
    dynamics, rewards = get_transition_dynamics(env)
    rewards = dynamics.multiply(rewards).sum(axis=0)

    Pbar = np.matrix((np.array(dynamics.toarray()).reshape((env.nS, env.nS, env.nA)) * prior_policy.reshape((1, env.nS, env.nA))).sum(axis=2))
    G = np.matrix(np.diag(np.exp(rho * np.array(rewards.reshape((env.nS, env.nA)).mean(axis=1)).flatten())))
    M = Pbar.dot(G)

    z = np.matrix(np.ones((env.nS, 1)))

    max_it = steps
    for i in range(1, max_it + 1):
        zk = z.T.dot(M).T
        lz = np.linalg.norm(zk)
        zk = zk / lz
        err = np.abs((np.log(zk) - np.log(z))/rho).max()
        z = zk
        if err <= tolerance:
            break

    if i == max_it:
        print(f'Reached max steps. Err:{err}')
    else:
        print(f"Done in {i} steps")

    zV = np.matrix(np.log(z) / rho)

    # we can generate a Boltzmann policy from the value function
    policy = dynamics.multiply(np.exp(rho * zV)).sum(axis=0).reshape((env.nS, env.nA))
    policy /= policy.sum(axis=1)

    return dict(
        V=np.array(zV).reshape((env.nS, 1)),
        policy=np.array(policy),
        eigenvalue=lz,
        left_eigenvector=np.array(z)
    )


def u_solver(env, prior_policy=None, steps=100, beta=1, tolerance=1e-2):

    if prior_policy is None:
        prior_policy = np.ones((env.nS, env.nA)) / env.nA

    # dynamics is a rectangular matrix. One row for each state (s,), one column for each (s, a)
    dynamics, rewards = get_transition_dynamics(env)
    rewards = dynamics.multiply(rewards).sum(axis=0)

    # the mdp generator is a square matrix. One row for each (s, a), one column for each (s, a)
    P = get_mdp_generator(env, dynamics, prior_policy)

    # diagonal matrix
    T = np.matrix(np.diag(np.exp(beta * np.array(rewards).flatten())))

    # the twisted matrix
    M = P.dot(T)

    # finding left and right eigenvectors
    u = np.matrix(np.ones((env.nS * env.nA, 1)))
    v = np.matrix(np.ones((env.nS * env.nA, 1)))

    max_it = steps
    for i in range(1, max_it+1):
        uk = u.T.dot(M).T
        lu = np.linalg.norm(uk)
        uk = uk / lu

        # computing an error for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        err = np.abs((np.log(uk[mask]) - np.log(u[mask]))/ beta).max() + np.logical_xor(uk > 0, u > 0).sum()

        vk = M.dot(v)
        lv = np.linalg.norm(vk)
        vk = vk / lv

        # computing an error for convergence estimation
        mask = np.logical_and(vk > 0, v > 0)
        err += np.abs((np.log(vk[mask]) - np.log(v[mask]))).max() + np.logical_xor(vk > 0, v > 0).sum()

        # update the eigenvectors
        u = uk
        v = vk

        if err <= tolerance:
            break

    if i == max_it:
        print(f'Reached max steps. Err:{err}')
    else:
        print(f"Done in {i} steps")

    theta = - np.log(lu)
    v = v / v.sum()
    u = u / u.T.dot(v)
    
    uQ = - steps * theta / beta + np.log(u) / beta
    uV = - steps * theta / beta + np.matrix(
        np.log(
            np.multiply(
                np.array(u).reshape((env.nS, env.nA)),
                prior_policy
            ).sum(axis=1)
        )
    ).T / beta

    policy = np.multiply(u.reshape(prior_policy.shape), prior_policy)
    policy = policy / policy.sum(axis=1)

    return dict(
        Q=np.array(uQ).reshape((env.nS, env.nA)),
        V=np.array(uV).reshape((env.nS, 1)),
        policy=np.array(policy),
        eigenvalue=lu,
        left_eigenvector=np.array(u),
        right_eigenvector=np.array(v)
    )
