import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import pandas as pd
import seaborn as sns
import scipy

import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az

def generate_data(alpha, beta, n=100, 
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
            # And only the value of playing NOT of not playing
            Q[cur_machine][a] = Q[cur_machine][a] + alpha * (r - Q[cur_machine][a])
        
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
    alpha, beta = x
    machines, actions, rewards = args
    n = len(actions)

    # Create a list with the Q values of each trial
    Qs = np.zeros((n, 4, 2), dtype=np.float)
    
    # The last Q values were never used, so there is no need to compute them
    for t, (m, a, r) in enumerate(zip(machines[:-1], actions[:-1], rewards[:-1])):
        Qs[t+1] = Qs[t]
        Qs[t+1, m, a] = Qs[t, m, a] + alpha * (r - Qs[t, m, a])
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
             alpha):
    Q = tt.set_subtensor(Q[machine, action], Q[machine, action] + alpha * (reward - Q[machine, action]))
    return Q

def theano_llik_td(alpha, beta, machines, actions, rewards, n=120):
    # Transform the variables into appropriate Theano objects
    machines_ = theano.shared(np.asarray(machines, dtype='int16')) 
    actions_ = theano.shared(np.asarray(actions, dtype='int16'))
    rewards_ = theano.shared(np.asarray(rewards, dtype='int16'))
    
    # Initialize the Q table
    Qs = tt.zeros((4,2), dtype='float64')

    # Compute the Q values for each trial
    Qs, updates = theano.scan(
        fn=update_Q,
        sequences=[machines_, actions_, rewards_],
        outputs_info=[Qs],
        non_sequences=[alpha])

    int_Qs = tt.zeros((1, 4,2), dtype='float64')

    Qs = tt.concatenate((int_Qs, Qs), axis=0)

    # Apply the softmax transformation
    idx = list(zip(range(n),machines)) #list of tuples
    obs_Qs = [Qs[i] for i in idx]
    Qs_ = obs_Qs * beta
    log_prob_actions = Qs_ - pm.math.logsumexp(Qs_, axis=1)

    # Calculate the negative log likelihod of the observed actions
    log_prob_actions = log_prob_actions[tt.arange(actions_.shape[0]), actions_]
    return tt.sum(log_prob_actions[1:])