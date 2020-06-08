print("Importing eGLM helpers...")
from collections import OrderedDict
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("white")
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import scale

sys.path.append('../')
sys.path.append('../../utils/')
# Primary module with most model functions
import model

def phi(x): 
    return(np.tanh(x))

def run_ucr_glm(all_nodes_ts, task_reg, standardize=False):
    
    """
    Runs classical GLM looping through each node of a network

    Parameters:
        all_nodes_ts = time series of all nodes in the network (DVs for GLM). 2D array with nodes for rows and time points for columns
        task_reg = task regressor (IV for GLM)

    Returns: 
        Dictionary with two items
        ucr_task_betas = uncorrected task parameter estimates
        ucr_mods = sm.OLS objects from which the task parameters come from
    """
    
    nregions = all_nodes_ts.shape[0]
    ucr_task_betas = np.zeros((nregions))
    ucr_mods = []
    
    for region in range(0, nregions):
        cur_y = all_nodes_ts[region,:]
        if standardize:
            ucr_mod = sm.OLS(scale(cur_y), scale(task_reg)) 
        else:
            ucr_mod = sm.OLS(cur_y, task_reg)
        ucr_res = ucr_mod.fit()
        ucr_task_betas[region] = ucr_res.params[0]
        ucr_mods.append(ucr_mod)
    
    return ({"ucr_task_betas":ucr_task_betas,
             "ucr_mods": ucr_mods})

def run_ext_glm(all_nodes_ts, task_reg, weight_matrix, dt, tau, g, s, standardize=False): 
    
    """
    Runs extended GLM looping through each node of a network

    Parameters:
        all_nodes_ts = time series of all nodes in the network (DVs for GLM). 2D array with nodes for rows and time points for columns
        task_reg = task regressor (IV for GLM)
        weight_matrix = weight matrix containing the connectivity weights for the network
        dt = sampling rate
        tau = time constant
        g = global information transfer strength
        s = local/self information transfer strength

    Returns: 
        Dictionary with two items
        ext_task_betas = corrected task parameter estimates
        ext_mods = sm.OLS objects from which the task parameters come from
    """
    
    nregions = all_nodes_ts.shape[0]
    ext_task_betas = np.zeros((nregions))
    ext_mods = []
       
    for region in range(0, nregions):
        cur_y = all_nodes_ts[region,:]
        incoming_connections = weight_matrix[region, :]
        incoming_connections = np.delete(incoming_connections,region)
        drop_region = [region]
        
        #DV
        next_y = cur_y[1:] #shift column up to predict activity in next time point

        #IV 1
        cur_y = cur_y[:-1] #drop last time point
        cur_y = ((2*(tau**2)+dt*(dt-2*tau))/(2*(tau**2)))*cur_y
        
        #IV 2
        other_ns_cur_spont = np.delete(all_nodes_ts, drop_region, axis=0)[:,:-1] #dropping last col/timepoint
        other_ns_cur_spont = other_ns_cur_spont.T
        other_ns_cur_spont = np.apply_along_axis(phi, 0, other_ns_cur_spont)
        other_ns_cur_spont = np.sum(other_ns_cur_spont*incoming_connections, axis = 1) 
        other_ns_cur_spont = (dt/(2*tau))*((tau-dt)/tau)*g*other_ns_cur_spont
        
        #IV 3
        cur_y_phi = all_nodes_ts[region,:]
        cur_y_phi = cur_y_phi[:-1]
        cur_y_phi = (dt/(2*tau))*((tau-dt)/tau)*s*phi(cur_y_phi)
        
        #IV 4
        cur_n_task = (dt/(2*tau))*((tau-dt)/tau)*task_reg[:-1]
        
        #IV 5
        other_ns_next_spont = np.delete(all_nodes_ts, drop_region, axis=0)[:,1:] #dropping first col/timepoint
        other_ns_next_spont = other_ns_next_spont.T
        other_ns_next_spont = np.apply_along_axis(phi, 0, other_ns_next_spont)
        other_ns_next_spont = np.sum(other_ns_next_spont*incoming_connections, axis = 1)
        other_ns_next_spont = (dt/(2*tau))*g*other_ns_next_spont
        
        #IV 6
        cur_n_first_appr = all_nodes_ts[region,:]
        cur_n_first_appr = cur_n_first_appr[:-1]
        cur_n_first_appr = ((tau-dt)/tau)*cur_n_first_appr
        tmp = np.delete(all_nodes_ts, drop_region, axis=0)[:,:-1] #dropping last col/timepoint
        tmp = tmp.T
        tmp = np.apply_along_axis(phi, 0, tmp)
        tmp = np.sum(tmp*incoming_connections, axis = 1) 
        tmp = (dt/tau)*g*tmp
        cur_n_first_appr = cur_n_first_appr+tmp
        tmp = all_nodes_ts[region,:]
        tmp = tmp[:-1]
        tmp = s*phi(tmp)
        cur_n_first_appr = cur_n_first_appr+tmp
        cur_n_first_appr = cur_n_first_appr+task_reg[:-1]
        cur_n_first_appr = (dt/(2*tau))*s*phi(cur_n_first_appr)
        
        #IV 7
        cur_n_next_task = (dt/(2*tau))*task_reg[1:]
        
        #All IVs in design matrix
        df = pd.DataFrame(data = {"next_y": next_y,
                                  "cur_y" : cur_y,
                                  "other_ns_cur_spont": other_ns_cur_spont,
                                  "cur_y_phi": cur_y_phi,
                                  "cur_n_task": cur_n_task,
                                  "other_ns_next_spont": other_ns_next_spont,
                                  "cur_n_first_appr": cur_n_first_appr,
                                  "cur_n_next_task": cur_n_next_task})
        
        s_df = pd.DataFrame(scale(df))
        s_df.rename(columns={i:j for i,j in zip(s_df.columns,df.columns)}, inplace=True)
        
        if standardize:
            ext_mod = smf.ols(formula = 'next_y ~ -1 + cur_y + other_ns_cur_spont + cur_y_phi + cur_n_task + other_ns_next_spont + cur_n_first_appr + cur_n_next_task', data = s_df)
        else:
             ext_mod = smf.ols(formula = 'next_y ~ -1 + cur_y + other_ns_cur_spont + cur_y_phi + cur_n_task + other_ns_next_spont + cur_n_first_appr + cur_n_next_task', data = df)
        
        ext_res = ext_mod.fit()
        ext_params = ext_res.params

        # What was the *(dt/(2*tau)) for?
        # This is the operation applied to the task regressor in the design matrix
        # 
        ext_task_betas[region] = (dt/(2*tau))*ext_params["cur_n_next_task"]
        ext_mods.append(ext_mod)
        
    else:
        return ({"ext_task_betas": ext_task_betas,
                 "ext_mods": ext_mods})

def make_stimtimes(Tmax, dt, stim_nodes, stim_mag, tasktiming=None, ncommunities = 3, nodespercommunity = 35,  sa = 500, ea = 1000, iv = 2000):
    
    """
    Creates timeseries for all nodes in network

    Parameters:
        Tmax = task length
        dt = sampling rate
        stim_nodes = nodes that are stimulated by the task
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        ncommunities = number of communities in network
        nodespercommunity = number of nodes per community in network
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval

    Returns: 
        2D array with nodes in rows and time points in columns
    """
    
    totalnodes = nodespercommunity*ncommunities
    T = np.arange(0,Tmax,dt)
    # Construct timing array for convolution 
    # This timing is irrespective of the task being performed
    # Tasks are only determined by which nodes are stimulated
    if tasktiming is None:
        tasktiming = np.zeros((len(T)))
        for t in range(len(T)):
            if t%iv>sa and t%iv<ea:
                tasktiming[t] = 1.0
    
    stimtimes = np.zeros((totalnodes,len(T)))
    
    # When task is ON the activity for a stim_node at that time point changes the size of stim_mag
    for t in range(len(T)):
        if tasktiming[t] == 1:
            stimtimes[stim_nodes,t] = stim_mag
            
    return(tasktiming, stimtimes)

def sim_network_task_glm(ncommunities = 3, 
                         innetwork_dsity = .60, 
                         outnetwork_dsity = .08, 
                         hubnetwork_dsity = .25, 
                         nodespercommunity = 35, 
                         plot_network = False,
                         dt = 1, tau = 1, g = 1, s = 1, 
                         topdown = True, bottomup = False, 
                         local_com = 1, 
                         Tmax = 100000, 
                         plot_task = False, 
                         stimsize = np.floor(35/3.0), 
                         noise = None,
                         noise_loc = 0, 
                         noise_scale = 0,
                         stim_mag = .5,
                         W = None,
                         taskdata = None,
                         tasktiming = None,
                         sa = 500,
                         ea = 1000,
                         iv = 2000,
                         standardize=False):
    
    """
    Simulates task activity in network and runs both uncorrected and corrected GLMs to estimate task parameters

    Parameters:
        ncommunities = number of communities in network (needed even if W is provided)
        innetwork_dsity = probability of a node being connected to another in the same community
        outnetwork_dsity = probability of a node being connected to another in a different (non-hub) community
        hubnetwork_dsity = probability of a node being connected to another in a hub community
        nodespercommunity = number of nodes per community in network (needed even if W is provided)
        plot_network = make plot showing the network connectivity weight matrix
        dt = sampling rate
        tau = time constany
        g = global information transfer strength
        s = self information transfer strength
        topdown = Topdown task that stimulates hub network only
        bottomup = Bottom up task that stimulates local network only
        local_com = local community index
        Tmax = task length
        plot_task = Make plot of task timing
        stimsize = number of nodes that will be stimulated by the task
        noise = Will noise be added to the timeseries
        noise_loc = Mean of normal distribution noise will be drawn from
        noise_scale = SD of normal distribution noise will be drawn from
        stim_mag = magnitude of stimulation
        W = alternative pre-specified weight matrix
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval
        standardize = Should the timeseries and the design matrix be standardized for the GLMs
        
    Returns: 
        Dictionary with weight matrix, corrected and uncorrected task parameters and model objects, stimulated nodes, timeseries for all nodes
    """
    
    #############################################
    # Create network
    #############################################

    if W is None:        
        totalnodes = nodespercommunity*ncommunities
        # Construct structural matrix
        S = model.generateStructuralNetwork(ncommunities=ncommunities,
                                            innetwork_dsity=innetwork_dsity,
                                            outnetwork_dsity=outnetwork_dsity,
                                            hubnetwork_dsity=hubnetwork_dsity,
                                            nodespercommunity=nodespercommunity,
                                            showplot=plot_network)
        # Construct synaptic matrix
        W = model.generateSynapticNetwork(S, showplot=plot_network)
        
    else:
        totalnodes = W.shape[0]

    if plot_network:
        plt.rcParams["figure.figsize"][0] = 5
        plt.rcParams["figure.figsize"][1] = 4
        sns.heatmap(W, xticklabels=False, yticklabels=False)
        plt.xlabel('Regions')
        plt.ylabel('Regions')
        plt.title("Synaptic Weight Matrix")

    #############################################
    # Get stim nodes
    #############################################
    
    # Construct a community affiliation vector
    Ci = np.repeat(np.arange(ncommunities),nodespercommunity) 
    # Identify the regions associated with the hub network (hub network is by default the 0th network)
    hub_ind = np.where(Ci==0)[0] 

    if topdown:
        stim_nodes_td = np.arange(0, stimsize, dtype=int)
    else:
        stim_nodes_td = None
    
    if bottomup:
        # Identify indices for one of the local communities
        local_ind = np.where(Ci==local_com)[0] 
        # Identify efferent connections from local network to hub network
        W_mask = np.zeros((W.shape))
        W_mask[local_ind,hub_ind] = 1.0
        local2hub_connects = np.multiply(W,W_mask)
        local_regions_wcon = np.where(local2hub_connects!=0)[0]
        local_regions_ncon = np.setdiff1d(local_ind,local_regions_wcon)
        #If there are enough nodes in the local community with hub connections:
        if len(local_regions_wcon)>= np.floor(stimsize/2):
            #Half of the stimulated local community nodes have hub connections while the other does not
            stim_nodes_bu = np.hstack((np.random.choice(local_regions_ncon, int(np.floor(stimsize/2)), replace=False),
                                np.random.choice(local_regions_wcon, int(stimsize-np.floor(stimsize/2)), replace=False)))
        else:
            stim_nodes_bu = np.hstack((np.random.choice(local_regions_wcon, len(local_regions_wcon), replace=False),
                                np.random.choice(local_regions_ncon, int(stimsize-len(local_regions_wcon)), replace=False)))
    else:
        stim_nodes_bu = None
    
    if stim_nodes_td is not None and stim_nodes_bu is not None:
        stim_nodes = np.hstack((stim_nodes_td, stim_nodes_bu))
    elif stim_nodes_td is not None and stim_nodes_bu is None:
        stim_nodes = stim_nodes_td
    else:
        stim_nodes = stim_nodes_bu
    
    #############################################
    # Make task and node stimulus timing
    #############################################
   
    tasktiming, stimtimes = make_stimtimes(Tmax=Tmax, dt=dt, stim_nodes=stim_nodes, stim_mag=stim_mag, 
                   tasktiming=tasktiming, ncommunities = ncommunities, nodespercommunity = nodespercommunity,  
                   sa = sa, ea = ea, iv = iv)

    if plot_task:
        if len(T)>9999:
            plt.plot(T[:10000], tasktiming[0,:10000])
            plt.ylim(top = 1.2, bottom = -0.1)
        else:
            plt.plot(T, tasktiming[0,:])
            plt.ylim(top = 1.2, bottom = -0.1)

    #############################################
    # Make task data (timeseries for all nodes)
    #############################################
    
    if taskdata is None:
        taskdata, error = model.networkModel(W,Tmax=Tmax,dt=dt,g=g,s=s,tau=tau, I=stimtimes, noise=noise, noise_loc = noise_loc, noise_scale = noise_scale)
    
    #Use only a subset of data for GLM's if it's too long
    if taskdata.shape[1]>44999:
        short_lim = int(np.floor(taskdata.shape[1]/3))
        y = copy.copy(taskdata[:,:short_lim])
        I = copy.copy(stimtimes[:,:short_lim])
    else:
        y = copy.copy(taskdata)
        I = copy.copy(stimtimes)
        
    #############################################
    # Run uncorrected and extended GLM to compare task regressor
    #############################################
    
    ucr_model = run_ucr_glm(all_nodes_ts = y, task_reg = I[stim_nodes[0],:], standardize=standardize)
    ext_model = run_ext_glm(all_nodes_ts = y, task_reg = I[stim_nodes[0],:], 
                            weight_matrix = W, dt = dt, tau = tau, g = g, s = s,
                            standardize=standardize)
    
    ucr_betas = ucr_model["ucr_task_betas"]
    ext_betas = ext_model["ext_task_betas"]
    
    ucr_glms = ucr_model["ucr_mods"]
    ext_glms = ext_model["ext_mods"]
    
    return({"W":W, "ucr_betas": ucr_betas, "ucr_glms": ucr_glms, "ext_betas": ext_betas, "ext_glms": ext_glms,
            "stim_nodes": stim_nodes, "taskdata": taskdata})

def get_res_ts(sim):
    
    """
    Residualized the time series accouting for connectivity effects and leaving only the remaining effect of the task regressor

    Parameters:
        sim = simulation dictionary output from sim_network_task_glm

    Returns: 
        res_ts = residualized timeseries 2D array with nodes for nodes and time points for columns
    """
    
    num_nodes = sim['taskdata'].shape[0]
    num_ts = len(sim['ext_glms'][0].endog)
    
    res_ts = np.zeros((num_nodes, num_ts))
    
    for cur_node in range(res_ts.shape[0]):
        raw_y = sim['ext_glms'][cur_node].endog
        # Drop task regressor column from the design matrix
        res_x = sim['ext_glms'][cur_node].exog[:,:-1]
        res_mod = sm.OLS(raw_y, res_x).fit()
        res_ts[cur_node,:] = res_mod.resid
    
    return(res_ts)

def get_res_taskreg(sim, node, dt=1, tau=1):
    
    """
    Residualized time series and projected task regressor for one node

    Parameters:
        sim = simulation dictionary output from sim_network_task_glm
        node = network node that will be used for residualization
        dt = sampling rate
        tau = time constant

    Returns: 
        res_y = residualized timeseries of given node
        m_task_reg = projected task regressor
    """
    
    raw_y = sim['ext_glms'][node].endog
    #design matrix without the task regressor
    X = sim['ext_glms'][node].exog[:,:-1]
    res_mod = sm.OLS(raw_y, X).fit()
    res_y = res_mod.resid
    
    #get projection matrix
    cur_p = np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T)
    
    #get residual making annihilator matrix
    cur_m = np.identity(cur_p.shape[0])-cur_p
    
    #task regressor in the design matrix is task*c where c = dt/(2*tau) is in the last column
    task_reg = sim['ext_glms'][node].exog[:,-1]
    
    #apply annihilator matrix on the raw task regressor
    m_task_reg = np.dot(cur_m, task_reg)
    
    #to capture the relationship between the projected but not transformed/true task regressor apply the inverse of the operation
    #applied to the original task regressor in the original design matrix
    m_task_reg = m_task_reg/(dt/(2*tau))
    
    return(res_y, m_task_reg)

# Baselines are calculated by 
#- residualizing the time series taking out the effect of the rest of the network and autocorrelated activity
#- for stimulated nodes: regressing the residualized timeseries onto the task regressor that's been projected and inverted
# --> is this the samething as getting the partial correlation? No. Partial correlation residualizes both x and y wrt the same 
# covariates and then looks at the relationship between them. This only projects the raw regressor into the prediction space of y
# it does NOT account for the transformations that have been applied to the task regressor. 
#- for non-stim nodes: append 0

def get_true_baseline(sim, stim_nodes = range(11), dt = 1, stim_mag=0.5, tasktiming = None, sa = 500, ea = 1000, iv = 2000):    
    
    """
    Get baselines for stimulated and non stimulated nodes against which GLM results will be compared

    Parameters:
        sim = simulation dictionary output from sim_network_task_glm
        stim_nodes = nodes that are stimulated by the task
        dt = sampling rate
        stim_mag = stimulus magnitude
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval

    Returns: 
        baseline_vec = baseline for all nodes
        
    """
    
    # needs fixing/more arguments for other types of tasks
    if tasktiming is None:
        task_timing = make_stimtimes(Tmax = sim['taskdata'].shape[1], dt=1, stim_nodes=stim_nodes, tasktiming = tasktiming, stim_mag=0.5, sa = sa, ea = ea, iv = iv)[0]
    task_indices = np.where(task_timing>0)
    
    # make sure the task indices aren't longer than the node timeseries
    # in case task regressor is cut in sim_network_task_glm to run faster
    if sim['taskdata'].shape[1]<len(task_indices):
        task_indices = task_indices[0][np.where(task_indices[0]<sim['taskdata'].shape[0])]
    
    baseline_vec = []
    
    # For large networks (if network has more than 10 nodes)
    # Most cases except for POC networks
    if sim['taskdata'].shape[0]>10:
        
        res_ts = get_res_ts(sim)
        
        # calculate residualized task regressor using only once for speed 
        # using the residualizing matrix from one stimulated node's regression only
        tmp_res_y, m_task_reg = get_res_taskreg(sim, stim_nodes[0])

        for cur_node in range(res_ts.shape[0]):
            if cur_node in stim_nodes:
                res_y = res_ts[cur_node,:]
                node_baseline = sm.OLS(res_y, m_task_reg).fit().params[0]
                baseline_vec.append(node_baseline)
            
            # For speed: not calculating proper baselines for non-stim nodes. Just adding 0 since they are not stimulated directly by task
            # This should be underestimating the true baseline for these nodes if they have any incoming connections from stim nodes
            else:
                baseline_vec.append(0)
        
        # If the network is small enough calculate the projected task regressor for all nodes
    else:
        for cur_node in range(sim['taskdata'].shape[0]):
                
            res_y, m_task_reg = get_res_taskreg(sim, cur_node)
            node_baseline = sm.OLS(res_y, m_task_reg).fit().params[0]
            baseline_vec.append(node_baseline)
            
    
    return(baseline_vec)

def plot_sim_network_glm(data,
                         width = 8,
                         height = 6,
                         ncoms = 3,
                         nnods = 35,
                         task_type = "td",
                         ucr_label = "cGLM (baseline)",
                         ext_label = "eGLM (baseline)",
                         base_label = None,
                         alp = 1,
                         stim_nodes = range(11),
                         tasktiming = None, sa = 500, ea = 1000, iv = 200):
    
    """
    Plotting wrapper comparing cGLM to eGLM results

    Parameters:
        data = simulation dictionary output from sim_network_task_glm
        width = width of figure
        height = height of figure
        ncoms = number of communities
        nnodes = number of nodes per community
        task_type = "td" for topdown (not extended to include "bu" yet)
        ucr_label = label for cGLM results
        ext_label = label for eGLM results
        base_label = label for baseline
        alp = opacity level 
        stim_nodes = nodes that are stimulated by the task
        tasktiming = block task array is created if not specified. If specified must be of length Tmax/dt
        sa = start point of stimulation
        ea = end point of stimulation
        iv = interstim interval

    Returns: 
        inline plot with nodes on the x axis and task parameter estimates on the y axis colored by GLM type
    """
    
    totalnodes = ncoms*nnods
    
    plt.rcParams["figure.figsize"][0] = width
    plt.rcParams["figure.figsize"][1] = height
    
    plt.plot(data['ucr_betas'], alpha = alp, color = "C0", label = ucr_label)
    plt.plot(data['ext_betas'], alpha = alp, color = "C1", label = ext_label)
    
    baseline_vec = get_true_baseline(data, stim_nodes = stim_nodes, tasktiming = tasktiming, sa = sa, ea = ea, iv = iv)
    

    all_nodes = list(range(totalnodes))
    stim_ind = [1 if x in data['stim_nodes'] else 0 for x in all_nodes]
    #baseline_vec = [stim_baseline if x == 1 else nonstim_baseline for x in stim_ind]
    plt.plot(baseline_vec, 
     color = "black", linestyle = '--', label = base_label, alpha = alp)
    
    plt.ylabel('Beta',fontsize=14)
    plt.xlabel('Node',fontsize=14)
    
    for n in range(1,ncoms):
        plt.axvline(x=nnods*n,linewidth=2, color='gray', ls = "--")
    
    plt.legend(loc="best")

print("All eGLM helpers imported!")
