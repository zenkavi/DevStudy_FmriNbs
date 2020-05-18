import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import pandas as pd
import seaborn as sns
import scipy

import pymc3 as pm
import theano
import theano.tensor as tt
from theano import tensor as T
from theano.ifelse import ifelse as tifelse
import arviz as az

def generate_data(alpha_neg= np.nan, alpha_pos= np.nan, exp_neg= 1, exp_pos= 1, beta= np.nan,
                  n=120,
                  p_r={'high_var': [.95, .05], 'low_var': [.5,.5]},
                  rs = np.array(([5.0, -495.0],[-5.0, 495.0],[10.0, -100.0],[-10.0, 100.0])),
                  sQ = np.zeros((4, 2))
                 ):

    # Need to denote both machine type and action

    # Pre-specify machines for each trial in a randomly balanced manner
    if n%4 != 0:
        print("Number of trials is not divisable by 4.\nCreating trials for %s trials."%(str(n-(n%4))))
        n = n-(n%4)

    machs = np.array([0,1,2,3])
    machs = np.tile(machs, int(n/4))
    np.random.shuffle(machs)

    # Initialize empty array that will be populated in the loop based on Q values
    acts = np.zeros(n, dtype=np.int)

    # Generate by coin flip for machine with differing probabilities and outcomes
    rews = np.zeros(n, dtype=np.int)

    # Stores the expected value for each of 4 machines in each trial for each action
    Qs = np.zeros((n, 4, 2))

    # Initialize Q table
    # Denotes expected value of each action
    # Should look like [0, 0] for each machine
    # *** The expected value of not playing should not change from 0! ***
    # Could these initial expected values/beliefs also be estimated from data?
    # E.g. what if kids have more optimistic priors about each machine though they learn at the same rate
    Q = sQ.copy()

    for i in range(n):

        cur_machine = machs[i]

        # Apply the Softmax transformation
        exp_Q = np.exp(np.multiply(beta, Q[cur_machine]))
        prob_a = exp_Q / np.sum(exp_Q)

        # Simulate choice
        a = np.random.choice([0, 1], p=prob_a)

        # Simulate reward if machine is played
        if a == 1:

            # Before sampling reward determine which variance condition machine is in
            if cur_machine>1:
                cur_p = 'low_var'
            else:
                cur_p = 'high_var'

            # Sample reward for current machine given its reward probs and outcome options
            r = np.random.choice(rs[cur_machine], p = p_r[cur_p])

            # Update Q table only if the machine is played
            # And only the value of playing NOT if not playing
            rpe = (r - Q[cur_machine][a])

            if rpe < 0:
                Q[cur_machine][a] = Q[cur_machine][a] + alpha_neg * abs(rpe)**exp_neg * (-1)

            if rpe >= 0:
                Q[cur_machine][a] = Q[cur_machine][a] + alpha_pos * rpe**exp_pos

        # If the machine is not played then Q remains unchanged and no reward is received
        else:
            r = 0.0

        # Store values
        acts[i] = a
        rews[i] = r
        #Qs[i] = Q.copy()
        Qs[i] = Q

    return machs, acts, rews, Qs

def llik_td_vectorized(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    # alpha, beta = x
    alpha_neg, alpha_pos, exp_neg, exp_pos, beta= x

    machines, actions, rewards = args
    n = len(actions)

    # Create a list with the Q values of each trial
    Qs = np.zeros((n, 4, 2), dtype=np.float)

    # The last Q values were never used, so there is no need to compute them
    for t, (m, a, r) in enumerate(zip(machines[:-1], actions[:-1], rewards[:-1])):
        Qs[t+1] = Qs[t]
        rpe = (r - Qs[t, m, a])
        if rpe < 0:
            Qs[t+1, m, a] = Qs[t, m, a] + alpha_neg * abs(rpe)**exp_neg * (-1)
        if rpe >= 0:
            Qs[t+1, m, a] = Qs[t, m, a] + alpha_pos * rpe**exp_pos
        #Qs[t+1, m, a] = Qs[t, m, a] + alpha * (r - Qs[t, m, a])
        Qs[t+1, m, 1-a] = Qs[t, m, 1-a]
        #print('t: %s, m: %s, a: %s, r: %s, Q:[%s, %s]'%(str(t), str(m), str(a), str(r), str(Qs[t,m,0]), str(Qs[t,m, 1])))

    # Apply the softmax transformation in a vectorized way
    idx = list(zip(range(n),machines))
    obs_Qs = [Qs[i] for i in idx]
    Qs_ = np.array(obs_Qs) * beta
    log_prob_actions = Qs_ - scipy.special.logsumexp(Qs_, axis=1)[:, None]

    # Return the log_prob_actions for the observed actions
    log_prob_obs_actions = log_prob_actions[np.arange(n), actions]
    return -np.sum(log_prob_obs_actions[1:])

def update_Q(machine, action, reward,
             Q,
            alpha_neg, alpha_pos, exp_neg, exp_pos):
    
    theano.config.compute_test_value = 'ignore'
    rpe = reward- Q[machine, action]

    Q_upd = tt.switch(rpe<0, Q[machine, action] + alpha_neg * np.sqrt((rpe)**2)**exp_neg * (-1), Q[machine, action] + alpha_pos * rpe**exp_pos)

    Q = tt.set_subtensor(Q[machine, action], Q_upd)
    return Q

def theano_llik_td(alpha_neg, alpha_pos, exp_neg, exp_pos, beta, machines, actions, rewards):

    #For single learning set alpha_neg = alpha_pos
    #For value distortion set exp_neg != 1, exp_pos != 1
    #For domain specificity set alpha_neg != alpha_pos, exp_neg != exp_pos

    # Transform the variables into appropriate Theano objects
    machines_ = theano.shared(np.asarray(machines, dtype='int16'))
    actions_ = theano.shared(np.asarray(actions, dtype='int16'))
    rewards_ = theano.shared(np.asarray(rewards, dtype='int16'))

    # Initialize the Q table
    Qs = tt.zeros((4,2), dtype='float64')

    alpha_neg = tt.scalar("alpha_neg")
    alpha_pos = tt.scalar("alpha_pos")
    exp_neg = tt.scalar("exp_neg")
    exp_pos = tt.scalar("exp_pos")
    beta = tt.scalar("beta")

    # Compute the Q values for each trial
    Qs, updates = theano.scan(
        fn=update_Q,
        sequences=[machines_, actions_, rewards_],
        outputs_info=[Qs],
        non_sequences=[alpha_neg, alpha_pos, exp_neg, exp_pos])
    
    int_Qs = tt.zeros((1, 4, 2), dtype='float64')

    Qs = tt.concatenate((int_Qs, Qs), axis=0)

    # Apply the softmax transformation
    n=len(actions)
    idx = list(zip(range(n),machines)) #list of tuples
    obs_Qs = [Qs[i] for i in idx]
    Qs_ = obs_Qs * beta
    log_prob_actions = Qs_ - pm.math.logsumexp(Qs_, axis=1)

    # Calculate the negative log likelihod of the observed actions
    log_prob_actions = log_prob_actions[tt.arange(actions_.shape[0]), actions_]
    theano.config.compute_test_value = 'ignore'
    return tt.sum(log_prob_actions[1:])


# Wrap all the steps pre output into a function
def get_nuts_est(t_alpha_neg = np.nan, t_alpha_pos = np.nan, t_exp_neg = 1, t_exp_pos = 1, t_beta = np.nan, n=120):

    # Generate data
    machines, actions, rewards, all_Qs = generate_data(alpha_neg= t_alpha_neg, alpha_pos= t_alpha_pos, exp_neg= t_exp_neg, exp_pos= t_exp_pos, beta= t_beta, n = n)
    true_llik = llik_td_vectorized([t_alpha_neg, t_alpha_pos, t_exp_neg, t_exp_pos, t_beta], *(machines, actions, rewards))

    # NUTS estimate
    actions_ = theano.shared(np.asarray(actions, dtype='int16'))
    with pm.Model() as m:
        s_alpha_neg = pm.Beta('alpha_neg', 1, 1)
        s_alpha_pos = pm.Beta('alpha_pos', 1, 1)
        s_exp_neg = pm.Beta('exp_neg', 1, 1)
        s_exp_pos = pm.Beta('exp_pos', 1, 1)
        s_beta = pm.HalfNormal('beta', 10)
        like = pm.Potential('like', theano_llik_td(s_alpha_neg, s_alpha_pos, s_exp_neg, s_exp_pos, s_beta, machines, actions, rewards, n))
        tr = pm.sample()

    nuts_alpha_neg_ave = np.mean(tr.alpha_neg)
    nuts_alpha_pos_ave = np.mean(tr.alpha_pos)
    nuts_exp_neg_ave = np.mean(tr.exp_neg)
    nuts_exp_pos_ave = np.mean(tr.exp_pos)
    nuts_beta_ave = np.mean(tr.beta)
    nuts_alpha_neg_std = np.mean(tr.alpha_neg)
    nuts_alpha_pos_std = np.mean(tr.alpha_pos)
    nuts_exp_neg_std = np.mean(tr.exp_neg)
    nuts_exp_pos_std = np.mean(tr.exp_pos)
    nuts_beta_std = np.mean(tr.beta)
    nuts_llik = llik_td_vectorized([nuts_alpha_neg, nuts_alpha_pos, nuts_exp_neg, nuts_exp_pos, nuts_beta], *(machines, actions, rewards))

    # Output:
    est_df = pd.DataFrame(data={"true_alpha_neg": t_alpha_neg,
                                "true_alpha_pos": t_alpha_pos,
                                "true_exp_neg": t_exp_neg,
                                "true_exp_pos": t_exp_pos,
                                "true_beta": t_beta,
                                "true_llik": true_llik,
                                "nuts_alpha_neg_ave": nuts_alpha_neg_ave,
                                "nuts_alpha_pos_ave": nuts_alpha_pos_ave,
                                "nuts_exp_neg_ave": nuts_exp_neg_ave,
                                "nuts_exp_pos_ave": nuts_exp_pos_ave,
                                "nuts_beta_ave": nuts_beta_ave,
                                "nuts_alpha_neg_std": nuts_alpha_neg_std,
                                "nuts_alpha_pos_std": nuts_alpha_pos_std,
                                "nuts_exp_neg_std": nuts_exp_neg_std,
                                "nuts_exp_pos_std": nuts_exp_pos_std,
                                "nuts_beta_std": nuts_beta_std,
                                "nuts_llik": nuts_llik}, index=[0])

    return (est_df)
