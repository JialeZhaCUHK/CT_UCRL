import numpy as np
import mdp
from ct_ucrl import ct_ucrl, extended_value_iteration
import matplotlib.pyplot as plt
import tqdm
import pickle as pkl

# Test of CT-MDP
eps = 1e-4
n_actions = 2
# Two-state example
n_states = 2
p = np.array([[[0, 1],[0, 1]],
     [[1, 0],[1, 0]]])
r = np.array([[5, 8], [-4, -12]])
holding_lambda = np.array([[3, 5], [2, 7]])

# Three-state example
# n_states = 3
# p = np.array([[[0, 1/2, 1/2], [0, 1/3, 2/3]],
#               [[2/3, 0, 1/3], [1/4, 0, 3/4]],
#               [[1/2, 1/2, 0], [3/5, 2/5, 0]]])
# r = np.array([[5, 8], [5, 2], [4, 10]])/10
# holding_lambda = np.array([[5, 2], [2, 1], [7, 3]])

state_val, best_ac, best_mdp = extended_value_iteration(
    n_states, n_actions, np.array(r), np.array(p), np.zeros(np.shape(p)[:2]), 
    1 / np.array(holding_lambda), np.zeros(np.shape(holding_lambda)), 
    holding_lambda.min(), holding_lambda.max(), eps)
st = 1
ac = best_ac[st]
rho_star = ((p[st][ac] * state_val).sum() - state_val[st] + r[st, ac]) * holding_lambda[st, ac]
rho_star


num_dec_epoch = int(8e6)
num_sim = 50
avg_regret = [0, np.zeros(num_dec_epoch)]
avg_reward = [0, np.zeros(num_dec_epoch)]
avg_rhoT = [0, np.zeros(num_dec_epoch)]
for sim in range(num_sim):
    print(sim, end='\r')
    opt_regret = []
    ct_mdp = mdp.CTMDP(n_states, n_actions, p, r, holding_lambda)

    transitions = ct_ucrl(ct_mdp, np.max(holding_lambda), np.min(holding_lambda), 
                        r, 0.05, 0)
    sum_reward = 0

    for _ in tqdm.tqdm(range(num_dec_epoch)):
        (t, st, ac, next_st, holding_time, reward) = transitions.__next__()
#         opt_tr[sim].append((t, st, ac, next_st, holding_time, reward))
        sum_reward += reward
        opt_regret.append(rho_star * ct_mdp.timer - sum_reward)
    
    opt_regret = np.array(opt_regret)
        
    with open('../../data/CT-UCRL/EX1/delta_001/regret_sim' + str(sim), 'wb') as file:
        pkl.dump(opt_regret, file)
        file.close()

    plt.plot(avg_regret[1], label = 'CT-UCRL')
    plt.show()