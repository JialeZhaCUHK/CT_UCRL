import numpy as np
import mdp
from ct_ucrl import ct_ucrl, extended_value_iteration
import matplotlib.pyplot as plt

# Test of CT-MDP
n_states = n_actions = 2
p = np.array([[[0, 1],[0, 1]],
     [[1, 0],[1, 0]]])
r = np.array([[5, 8], [-4, -12]])
holding_lambda = np.array([[3, 5], [2, 7]])
ct_mdp = mdp.CTMDP(2, 2, p, r, holding_lambda)
holding_time_record = {0: {0: [], 1:[]}, 1: {0: [], 1:[]}}
transit_record = {0: {0: np.array([0, 0]), 1:np.array([0, 0])}, 
                  1: {0: np.array([0, 0]), 1:np.array([0, 0])}}

ct_mdp.reset(0)
for i in range(50000):
    ac = np.random.randint(2)
    state = ct_mdp.state
    next_state, reward, holding_time = ct_mdp.step(ac)
    holding_time_record[state][ac].append(holding_time)
    transit_record[state][ac][next_state] += 1

print('Test of CT-MDP: \n')
print('True Parameter: ')
for st in [0, 1]:
    for ac in [0, 1]:
        print(r'state: {}, action: {}, $\lambda(s, a)$: {}'.format(st, ac, holding_lambda[st, ac]))

print('\n Estimated Parameter: ')
for st in [0, 1]:
    for ac in [0, 1]:
        print(r'state: {}, action: {}, $\hat\lambda(s, a)$: {}'.format(
            st, ac, 1/np.mean(holding_time_record[st][ac])))
        print(r'$p(s\' | s, a)$: {}'.format(
            transit_record[state][ac]/np.sum(transit_record[st][ac])))

print('Test of CT-MDP is passed!')