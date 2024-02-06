import itertools
import math

import numpy as np

from mdp import CTMDP


def inner_maximization(p_sa_hat, confidence_bound_p_sa, rank):
    '''
    Find the best local transition p(.|s, a) within the plausible set of transitions as bounded by the confidence bound for some state action pair.
    Arg:
        p_sa_hat : (n_states)-shaped float array. MLE estimate for p(.|s, a).
        confidence_bound_p_sa : scalar. The confidence bound for p(.|s, a) in L1-norm.
        rank : (n_states)-shaped int array. The sorted list of states in descending order of value.
    Return:
        (n_states)-shaped float array. The optimistic transition p(.|s, a).
    '''
    # print('rank', rank)
    p_sa = np.array(p_sa_hat)
    p_sa[rank[0]] = min(1, p_sa_hat[rank[0]] + confidence_bound_p_sa / 2)
    rank_dup = list(rank)
    last = rank_dup.pop()
    # Reduce until it is a distribution (equal to one within numerical tolerance)
    while sum(p_sa) > 1 + 1e-9:
        # print('inner', last, p_sa)
        p_sa[last] = max(0, 1 - sum(p_sa) + p_sa[last])
        last = rank_dup.pop()
    # print('p_sa', p_sa)
    return p_sa


def extended_value_iteration(n_states, n_actions, r, p_hat, confidence_bound_p, 
                             holding_rate_hat, confidence_bound_holding_time, 
                             holding_rate_min, holding_rate_max, epsilon):
    '''
    The extended value iteration which finds an optimistic MDP within the plausible set of MDPs and solves for its near-optimal policy.
    '''
    # Initial values (an optimal 0-step non-stationary policy's values)
    state_value_hat = np.zeros(n_states)
    next_state_value_hat = np.zeros(n_states)
    diff_value = np.zeros(n_states)
    diff_value[0], diff_value[-1] = math.inf, -math.inf

    # Optimistic MDP and its epsilon-optimal policy
    p_tilde = np.zeros((n_states, n_actions, n_states))
    pi_tilde = np.zeros(n_states, dtype='int')
    holding_rate_tilde = np.zeros((n_states, n_actions))

    while not (diff_value.max() - diff_value.min()) < epsilon:
        # Sort the states by their values in descending order
        rank = np.argsort(-state_value_hat)
        for st in range(n_states):
            best_ac, best_q = None, -math.inf
            for ac in range(n_actions):
                # print('opt', st, ac)
                # print(state_value_hat)
                
                # Optimistic transitions
                p_sa_tilde = inner_maximization(p_hat[st, ac], confidence_bound_p[st, ac], rank)
                q_sa = r[st, ac] + (p_sa_tilde * state_value_hat).sum() - state_value_hat[st]
                p_tilde[st, ac] = p_sa_tilde

                # Optimistic holding rates
                if q_sa > 0:
                    rev_lambda_tilde = 1 / holding_rate_hat[st, ac] - confidence_bound_holding_time
                    holding_rate_tilde[st, ac] = min(holding_rate_max, 1 / rev_lambda_tilde)
                else:
                    rev_lambda_tilde = 1 / holding_rate_hat[st, ac] + confidence_bound_holding_time
                    holding_rate_tilde[st, ac] = max(holding_rate_min, 1 / rev_lambda_tilde)

                # Optimistic uniformed Q-value
                unif_q_sa = q_sa * holding_rate_tilde[st, ac] / holding_rate_max

                # Value function
                if best_q < unif_q_sa:
                    best_q = unif_q_sa
                    best_ac = ac
                    pi_tilde[st] = best_ac
            next_state_value_hat[st] = best_q + state_value_hat[st]
            # print(state_value_hat)
        diff_value = next_state_value_hat - state_value_hat
        state_value_hat = next_state_value_hat
        next_state_value_hat = np.zeros(n_states)
        # print('u', state_value_hat, du.max() - du.min(), epsilon)
    return pi_tilde, (p_tilde, holding_rate_tilde)


def ct_ucrl(ct_mdp: CTMDP, delta, initial_state=None):
    '''
    CT_UCRL algorithm
    See _Logarithmic regret bounds for continuous-time average-reward Markov decision processes_ by Gao, X., & Zhou, X. Y. (2022)
    '''
    n_states, n_actions = ct_mdp.n_states, ct_mdp.n_actions
    n = 1
    # Initial state
    st = ct_mdp.reset(initial_state)
    # Model estimates
    total_visitations = np.zeros((n_states, n_actions))
    total_holding_time = np.zeros((n_states, n_actions))
    total_transitions = np.zeros((n_states, n_actions, n_states))
    vi = np.zeros((n_states, n_actions))
    for k in itertools.count():
        # Initialize episode k
        t_k = n
        # Per-episode visitations
        vi = np.zeros((n_states, n_actions))
        # MLE estimates
        p_hat = total_transitions / np.clip(total_visitations.reshape((n_states, n_actions, 1)), 1, None)
        # print('p_hat', p_hat)
        r_hat = total_rewards / np.clip(total_visitations, 1, None)
        # print('r_hat', r_hat)

        # Compute near-optimal policy for the optimistic MDP
        confidence_bound_r = np.sqrt(7 * np.log(2 * n_states * n_actions * t_k / delta) / (2 * np.clip(total_visitations, 1, None)))
        confidence_bound_p = np.sqrt(14 * np.log(2 * n_actions * t_k / delta) / np.clip(total_visitations, 1, None))
        # print('cb_p', confidence_bound_p)
        # print('cb_r', confidence_bound_r)
        pi_k, mdp_k = extended_value_iteration(n_states, n_actions, p_hat, confidence_bound_p, r_hat, confidence_bound_r, 1 / np.sqrt(t_k))
        # print(pi_k, mdp_k)

        # Execute policy
        ac = pi_k[st]
        # End episode when we visit one of the state-action pairs "often enough"
        while vi[st, ac] < max(1, total_visitations[st, ac]):
            next_st, reward = mdp.step(ac)
            # print('step', t, st, ac, next_st, reward)
            yield (t, st, ac, next_st, reward)
            # Update statistics
            vi[st, ac] += 1
            total_rewards[st, ac] += reward
            total_transitions[st, ac, next_st] += 1
            # Next tick
            n += 1
            st = next_st
            ac = pi_k[st]

        total_visitations += vi


if __name__ == '__main__':
    eps = 0.1
    alpha = 0.1
    n_states = n_actions = 2
    p = [
            [
                [1, 0],
                [1 - eps, eps]
            ],
            [
                [0, 1],
                [eps, 1 - eps]
            ],
         ]
    r = [
            [1 - alpha, 1 - alpha],
            [1, 1],
        ]
    mdp = SimpleMDP(n_states, n_actions, p, r)

    transitions = ucrl2(mdp, delta=0.1, initial_state=0)
    tr = []
    for _ in range(4000000):
        (t, st, ac, next_st, r) = transitions.__next__()
        tr.append((t, st, ac, next_st, r))
