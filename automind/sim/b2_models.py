### Brian2 models for network construction.
import brian2 as b2
import numpy as np
from automind.sim import b2_inputs


def adaptive_exp_net(all_param_dict):
    """Adaptive exponential integrate-and-fire network."""
    # separate parameter dictionaries
    param_dict_net = all_param_dict["params_net"]
    param_dict_settings = all_param_dict["params_settings"]

    # set random seeds
    b2.seed(param_dict_settings["random_seed"])
    np.random.seed(param_dict_settings["random_seed"])
    b2.defaultclock.dt = param_dict_settings["dt"]

    param_dict_neuron_E = all_param_dict["params_Epop"]
    # check if there is inhibition
    has_inh = False if param_dict_net["exc_prop"] == 1 else True
    if has_inh:
        param_dict_neuron_I = all_param_dict["params_Ipop"]

    #### NETWORK CONSTRUCTION ############
    ######################################
    ### define neuron equation
    adex_coba_eq = """dv/dt = (-g_L * (v - v_rest) + g_L * delta_T * exp((v - v_thresh)/delta_T) - w + I)/C : volt (unless refractory)"""
    adlif_coba_eq = (
        """dv/dt = (-g_L * (v - v_rest) - w + I)/C : volt (unless refractory)"""
    )

    network_eqs = """            
            dw/dt = (-w + a * (v - v_rest))/tau_w : amp
            dge/dt = -ge / tau_ge : siemens
            dgi/dt = -gi / tau_gi : siemens
            Ie = ge * (E_ge - v): amp
            Ii = gi * (E_gi - v): amp
            I = I_bias + Ie + Ii : amp
            """

    ### get cell counts
    N_pop, exc_prop = param_dict_net["N_pop"], param_dict_net["exc_prop"]
    N_exc, N_inh = int(N_pop * exc_prop), int(N_pop * (1 - exc_prop))

    ### make neuron populations, set initial values and connect poisson inputs ###
    # make adlif if delta_t is 0, otherwise adex
    neuron_eq = (
        adlif_coba_eq + network_eqs
        if param_dict_neuron_E["delta_T"] == 0
        else adex_coba_eq + network_eqs
    )
    E_pop = b2.NeuronGroup(
        N_exc,
        model=neuron_eq,
        threshold="v > v_cut",
        reset="v = v_reset; w+=b",
        refractory=param_dict_neuron_E["t_refrac"],
        method=param_dict_settings["integ_method"],
        namespace=param_dict_neuron_E,
    )
    E_pop.v = (
        param_dict_neuron_E["v_rest"]
        + np.random.randn(N_exc) * param_dict_neuron_E["v_0_offset"][1]
        + param_dict_neuron_E["v_0_offset"][0]
    )

    ### TO DO: also randomly initialize w to either randint(?)*b or randn*(v-v_rest)*a
    poisson_input_E = b2.PoissonInput(
        target=E_pop,
        target_var="ge",
        N=param_dict_neuron_E["N_poisson"],
        rate=param_dict_neuron_E["poisson_rate"],
        weight=param_dict_neuron_E["Q_poisson"],
    )


    if has_inh:
        # make adlif if delta_t is 0, otherwise adex
        neuron_eq = (
            adlif_coba_eq + network_eqs
            if param_dict_neuron_E["delta_T"] == 0
            else adex_coba_eq + network_eqs
        )
        I_pop = b2.NeuronGroup(
            N_inh,
            model=neuron_eq,
            threshold="v > v_cut",
            reset="v = v_reset; w+=b",
            refractory=param_dict_neuron_I["t_refrac"],
            method=param_dict_settings["integ_method"],
            namespace=param_dict_neuron_I,
        )
        I_pop.v = (
            param_dict_neuron_I["v_rest"]
            + np.random.randn(N_inh) * param_dict_neuron_I["v_0_offset"][1]
            + param_dict_neuron_I["v_0_offset"][0]
        )
        # check for poisson input
        if (
            param_dict_neuron_I["N_poisson"] == 0
            or param_dict_neuron_I["poisson_rate"] == 0
        ):
            poisson_input_I = []
        else:
            poisson_input_I = b2.PoissonInput(
                target=I_pop,
                target_var="ge",
                N=param_dict_neuron_I["N_poisson"],
                rate=param_dict_neuron_I["poisson_rate"],
                weight=param_dict_neuron_I["Q_poisson"],
            )

    ### make and connect recurrent synapses ###
    syn_e2e = b2.Synapses(
        source=E_pop,
        target=E_pop,
        model="q: siemens",
        on_pre="ge_post += Q_ge",
        delay=param_dict_neuron_E["tdelay2e"],
        namespace=param_dict_neuron_E,
    )
    syn_e2e.connect("i!=j", p=param_dict_net["p_e2e"])
    if has_inh:
        syn_e2i = b2.Synapses(
            source=E_pop,
            target=I_pop,
            model="q: siemens",
            on_pre="ge_post += Q_ge",
            delay=param_dict_neuron_E["tdelay2i"],
            namespace=param_dict_neuron_I,
        )
        syn_e2i.connect("i!=j", p=param_dict_net["p_e2i"])
        syn_i2e = b2.Synapses(
            source=I_pop,
            target=E_pop,
            model="q: siemens",
            on_pre="gi_post += Q_gi",
            delay=param_dict_neuron_I["tdelay2e"],
            namespace=param_dict_neuron_E,
        )
        syn_i2e.connect("i!=j", p=param_dict_net["p_i2e"])
        syn_i2i = b2.Synapses(
            source=I_pop,
            target=I_pop,
            model="q: siemens",
            on_pre="gi_post += Q_gi",
            delay=param_dict_neuron_I["tdelay2i"],
            namespace=param_dict_neuron_I,
        )
        syn_i2i.connect("i!=j", p=param_dict_net["p_i2i"])

    ### define monitors ###
    rate_monitors, spike_monitors, trace_monitors = [], [], []
    rec_defs = param_dict_settings["record_defs"]
    dt_ts = param_dict_settings["dt_ts"]  # recording interval for continuous variables
    zipped_pops = (
        zip(["exc", "inh"], [E_pop, I_pop]) if has_inh else zip(["exc"], [E_pop])
    )
    for pop_name, pop in zipped_pops:
        if pop_name in rec_defs.keys():
            if rec_defs[pop_name]["rate"] is not False:
                rate_monitors.append(
                    b2.PopulationRateMonitor(pop, name=pop_name + "_rate")
                )
            if rec_defs[pop_name]["spikes"] is not False:
                rec_idx = (
                    np.arange(rec_defs[pop_name]["spikes"])
                    if type(rec_defs[pop_name]["spikes"]) is int
                    else rec_defs[pop_name]["spikes"]
                )
                spike_monitors.append(
                    b2.SpikeMonitor(pop[rec_idx], name=pop_name + "_spikes")
                )
            if rec_defs[pop_name]["trace"] is not False:
                rec_idx = (
                    np.arange(rec_defs[pop_name]["trace"][1])
                    if type(rec_defs[pop_name]["trace"][1]) is int
                    else rec_defs[pop_name]["trace"][1]
                )
                trace_monitors.append(
                    b2.StateMonitor(
                        pop,
                        rec_defs[pop_name]["trace"][0],
                        record=rec_idx,
                        name=pop_name + "_trace",
                        dt=dt_ts,
                    )
                )

    ### collect into network object and return ###
    net_collect = b2.Network(b2.collect())  # magic collect all groups
    monitors = [rate_monitors, spike_monitors, trace_monitors]
    net_collect.add(monitors)
    return net_collect


def make_clustered_edges(N_neurons, n_clusters, clusters_per_neuron, sort_by_cluster):
    """Make clustered connection graph.

    Args:
        N_neurons (int): Number of neurons.
        n_clusters (int): Number of clusters.
        clusters_per_neuron (int): Number of clusters per neuron.
        sort_by_cluster (bool): Sort by cluster.

    Returns:
        tuple: Membership, shared membership, connections in, and connections out.
    """
    # assign cluster membership randomly
    if clusters_per_neuron > n_clusters:
        print("More clusters per neuron than exists.")
        return
    membership = np.random.randint(
        int(n_clusters), size=(N_neurons, int(clusters_per_neuron))
    )

    if sort_by_cluster:
        # sort to show block-wise membership
        membership = membership[np.argsort(membership[:, 0]), :]

    # find pairs that share membership
    shared_membership = (
        np.array([np.any(m_i == membership, axis=1) for m_i in membership])
    ).astype(int)

    # connections not in the same clusters
    conn_out = np.array((1 - shared_membership).nonzero())
    # connections in the same clusters
    conn_in = np.array((shared_membership).nonzero())
    # remove all i==j to get the correct # of edges as percentage of possible
    conn_in = conn_in[:, conn_in[0, :] != conn_in[1, :]]

    return membership, shared_membership, conn_in, conn_out


def draw_prob_connections(conn, p):
    """Draw probabilistic connections given connectivity matrix shape."""
    # randomly choose p * C connections
    # where C is the number of possible connections
    ij = np.sort(
        np.random.choice(
            conn.shape[1], np.ceil(conn.shape[1] * p).astype(int), replace=False
        )
    )
    return conn[:, ij]


def make_clustered_network(
    N_neurons, n_clusters, clusters_per_neuron, p_in, p_out, sort_by_cluster=False
):
    """Make clustered network.

    Args:
        N_neurons (int): Number of neurons.
        n_clusters (int): Number of clusters.
        clusters_per_neuron (int): Number of clusters per neuron.
        p_in (float): probability of incoming connections.
        p_out (float): probability of outgoing connections.
        sort_by_cluster (bool, optional): Sort by cluster. Defaults to False.

    Returns:
        tuple: Membership, shared membership, connections in, and connections out.
    """
    membership, shared_membership, conn_in, conn_out = make_clustered_edges(
        N_neurons, n_clusters, clusters_per_neuron, sort_by_cluster
    )
    conn_in = draw_prob_connections(conn_in, p_in)
    conn_out = draw_prob_connections(conn_out, p_out)
    return membership, shared_membership, conn_in, conn_out


#Modified function incorporating inputs to specific clusters 
def adaptive_exp_net_clustered_custom_input(all_param_dict):
    '''
    Adaptive exponential integrate-and-fire network with clustered connections.

    Args:
        all_param_dict (dict): Dictionary for simulation parameters.
        mode (string): Accepts 'default', 'single', or 'cluster'
        - 'default' mode - no input
        - 'single' mode - type: array. 
            Single input array with same length as sim time / dt
            User can define any input sequence. b2_inputs.DM_simple is used when no inputs are provided
        - 'cluster' mode - type: List (of arrays). 
            Each cluster gets different input, can be defined by user. DM_simple with different means is used for each cluster when no inputs are provided. 
            Can also select number of clusters to stimulate 

        custom_input (arr): Input array with same length as (simulation time / dt)
        stim_cluster (int): Number of clusters to stimulate
        custom_cluster_input (list of arrays): List of input arrays, each in the format as custom_input.
    '''

    # separate parameter dictionaries
    param_dict_net = all_param_dict["params_net"]
    param_dict_settings = all_param_dict["params_settings"]

    # set random seeds
    b2.seed(param_dict_settings["random_seed"])
    np.random.seed(param_dict_settings["random_seed"])
    b2.defaultclock.dt = param_dict_settings["dt"]
    param_dict_neuron_E = all_param_dict["params_Epop"]

    # check if there is inhibition
    has_inh = False if param_dict_net["exc_prop"] == 1 else True
    if has_inh:
        param_dict_neuron_I = all_param_dict["params_Ipop"]

    #### NETWORK CONSTRUCTION ############
    ######################################

    ### get cell counts
    N_pop, exc_prop = param_dict_net["N_pop"], param_dict_net["exc_prop"]
    N_exc = int(N_pop * exc_prop)
    N_inh = N_pop - N_exc

    ### define neuron equation
    adex_coba_eq = """dv/dt = (-g_L * (v - v_rest) + g_L * delta_T * exp((v - v_thresh)/delta_T) - w + I)/C : volt (unless refractory)"""

    adlif_coba_eq = """dv/dt = (-g_L * (v - v_rest) - w + I)/C : volt (unless refractory)"""

    network_eqs = """            
            dw/dt = (-w + a * (v - v_rest))/tau_w : amp
            dge/dt = -ge / tau_ge : siemens
            dgi/dt = -gi / tau_gi : siemens
            Ie = ge * (E_ge - v): amp
            Ii = gi * (E_gi - v): amp
            I_ext: amp 
            I = I_bias + Ie + Ii + I_ext: amp
            """

    ### make neuron populations, set initial values and connect poisson inputs ###
    # make adlif if delta_t is 0, otherwise adex
    neuron_eq = (
        adlif_coba_eq + network_eqs
        if param_dict_neuron_E["delta_T"] == 0
        else adex_coba_eq + network_eqs
    )
    E_pop = b2.NeuronGroup(
        N_exc,
        model=neuron_eq,
        threshold="v > v_cut",
        reset="v = v_reset; w+=b",
        refractory=param_dict_neuron_E["t_refrac"],
        method=param_dict_settings["integ_method"],
        namespace=param_dict_neuron_E,
        name="Epop",
    )
    E_pop.v = (
        param_dict_neuron_E["v_rest"]
        + np.random.randn(N_exc) * param_dict_neuron_E["v_0_offset"][1]
        + param_dict_neuron_E["v_0_offset"][0]
    )

    ### SUBSET INPUT
    # define subset of E cells that receive input, to model
    # spontaneously active cells (which are a small proportion)
    N_igniters = int(param_dict_neuron_E["p_igniters"] * N_exc)
    # has to be a contiguous chunk
    E_igniters = E_pop[:N_igniters]

    # connect poisson input only to igniter neurons
    poisson_input_E = b2.PoissonInput(
        target=E_igniters,
        target_var="ge",
        N=param_dict_neuron_E["N_poisson"],
        rate=param_dict_neuron_E["poisson_rate"],
        weight=param_dict_neuron_E["Q_poisson"],
    )

    #### resolve inhibitory population
    if has_inh:
        # make adlif if delta_t is 0, otherwise adex
        neuron_eq = (
            adlif_coba_eq + network_eqs
            if param_dict_neuron_E["delta_T"] == 0
            else adex_coba_eq + network_eqs
        )
        I_pop = b2.NeuronGroup(
            N_inh,
            model=neuron_eq,
            threshold="v > v_cut",
            reset="v = v_reset; w+=b",
            refractory=param_dict_neuron_I["t_refrac"],
            method=param_dict_settings["integ_method"],
            namespace=param_dict_neuron_I,
            name="Ipop",
        )
        I_pop.v = (
            param_dict_neuron_I["v_rest"]
            + np.random.randn(N_inh) * param_dict_neuron_I["v_0_offset"][1]
            + param_dict_neuron_I["v_0_offset"][0]
        )
        # check for poisson input
        if (param_dict_neuron_I["N_poisson"] == 0) or (
            param_dict_neuron_I["poisson_rate"] == 0
        ):
            poisson_input_I = []
        else:
            poisson_input_I = b2.PoissonInput(
                target=I_pop,
                target_var="ge",
                N=param_dict_neuron_I["N_poisson"],
                rate=param_dict_neuron_I["poisson_rate"],
                weight=param_dict_neuron_I["Q_poisson"],
            )

    ############## SYNAPSES ####################
    ### make and connect recurrent synapses ###
    if (
        ("n_clusters" not in param_dict_net.keys())
        or (param_dict_net["n_clusters"] < 2)
        or (param_dict_net["R_pe2e"] == 1)
    ):
        # make homogeneous connection if 0,1 (or unspecified) cluster, or R_pe2e (ratio between in:out connection prob) is 1
        # print('homogeneous')
        syn_e2e = b2.Synapses(
            source=E_pop,
            target=E_pop,
            model="q: siemens",
            on_pre="ge_post += Q_ge",
            delay=param_dict_neuron_E["tdelay2e"],
            namespace=param_dict_neuron_E,
            name="syn_e2e",
        )
        syn_e2e.connect("i!=j", p=param_dict_net["p_e2e"])

    else:
        # print(f'clustered: n_clusters={param_dict_net["n_clusters"]}')
        # scale connectivity probability and make clustered connections
        p_out = param_dict_net["p_e2e"]
        p_in = p_out * param_dict_net["R_pe2e"]

        # NOTE: if cluster membership is ordered (last arg), then it explicitly makes
        # block diagonal wrt neuron id, which coincides with input (first n)
        # otherwise, clusters id are randomly assigned
        # print('ordered cluster id' if param_dict_net['order_clusters'] else 'random cluster id')
        #
        # Also, given non-integer n_clusters, int() is applied, which floors it.
        membership, shared_membership, conn_in, conn_out = make_clustered_network(
            N_exc,
            param_dict_net["n_clusters"],
            param_dict_net["clusters_per_neuron"],
            p_in,
            p_out,
            param_dict_net["order_clusters"],
        )
        
        # Membership not needed outside of function aside from analysis and visualisation and only saved in param_dict_net
        param_dict_net["membership"] = membership

        # scale synaptic weight
        Q_ge_out = param_dict_neuron_E["Q_ge"]
        param_dict_neuron_E["Q_ge_out"] = Q_ge_out
        param_dict_neuron_E["Q_ge_in"] = Q_ge_out * param_dict_net["R_Qe2e"]

        # make synapses and connect
        # in-cluster synapses
        syn_e2e_in = b2.Synapses(
            source=E_pop,
            target=E_pop,
            model="q: siemens",
            on_pre="ge_post += Q_ge_in",
            delay=param_dict_neuron_E["tdelay2e"],
            namespace=param_dict_neuron_E,
            name="syn_e2e_in",
        )
        syn_e2e_in.connect(i=conn_in[0, :], j=conn_in[1, :])

        # across-cluster synapses
        syn_e2e_out = b2.Synapses(
            source=E_pop,
            target=E_pop,
            model="q: siemens",
            on_pre="ge_post += Q_ge_out",
            delay=param_dict_neuron_E["tdelay2e"],
            namespace=param_dict_neuron_E,
            name="syn_e2e_out",
        )
        syn_e2e_out.connect(i=conn_out[0, :], j=conn_out[1, :])

    if has_inh:
        # no clustered connections to or from inhibitory populations
        # e to i
        syn_e2i = b2.Synapses(
            source=E_pop,
            target=I_pop,
            model="q: siemens",
            on_pre="ge_post += Q_ge",
            delay=param_dict_neuron_E["tdelay2i"],
            namespace=param_dict_neuron_I,
            name="syn_e2i",
        )
        syn_e2i.connect("i!=j", p=param_dict_net["p_e2i"])

        # i to e
        syn_i2e = b2.Synapses(
            source=I_pop,
            target=E_pop,
            model="q: siemens",
            on_pre="gi_post += Q_gi",
            delay=param_dict_neuron_I["tdelay2e"],
            namespace=param_dict_neuron_E,
            name="syn_i2e",
        )
        syn_i2e.connect("i!=j", p=param_dict_net["p_i2e"])

        # i to i
        syn_i2i = b2.Synapses(
            source=I_pop,
            target=I_pop,
            model="q: siemens",
            on_pre="gi_post += Q_gi",
            delay=param_dict_neuron_I["tdelay2i"],
            namespace=param_dict_neuron_I,
            name="syn_i2i",
        )
        syn_i2i.connect("i!=j", p=param_dict_net["p_i2i"])

    ### Handle different input modes ### 
    #!!!Arguments for input modes etc need to be first generated in b2_inputs.generate_input_params!!!
    param_dict_input = all_param_dict.get("params_input", {"mode": "default"}) #In case params_input is not provided
    mode = param_dict_input.get("mode")
    
    if mode == 'default': #No input (aside from poisson) provided to the network
        pass
            
    elif mode == 'single':
    # Check if custom input is provided in the parameter dictionary
        stim_time_values = param_dict_input['single_input']   
        dt = param_dict_settings["dt"]
        stim_timed_array = b2.TimedArray(stim_time_values * b2.amp, dt=dt)

        # Define network operation to update I_ext
        @b2.network_operation(dt=dt)
        def update_test_input(t):
            E_pop.I_ext = stim_timed_array(t)

    elif mode == 'cluster': 
        # Condition check for clusters is moved to b2_inputs.generate_input_params_dict
        # Select clusters for stimulation 
        selected_clusters = param_dict_input["selected_clusters"]
        cluster_lists = [[c] for c in selected_clusters] 

        # Generate cluster-specific inputs for selected clusters - see b2_inputs
        stim_list = param_dict_input["cluster_input_list"]
        weight_list = param_dict_input["cluster_weight_list"]
            
        # Create input configurations
        input_configs = b2_inputs.get_input_configs(
                cluster_lists,
                stim_list,
                weight_list,
            )
        input_op = b2_inputs.create_input_operation(E_pop, input_configs, membership)

    ### define monitors ###
    rate_monitors, spike_monitors, trace_monitors = [], [], []
    rec_defs = param_dict_settings["record_defs"]
    dt_ts = param_dict_settings["dt_ts"]  # recording interval for continuous variables
    zipped_pops = (
        zip(["exc", "inh"], [E_pop, I_pop]) if has_inh else zip(["exc"], [E_pop])
    )
    for pop_name, pop in zipped_pops:
        if pop_name in rec_defs.keys():
            if rec_defs[pop_name]["rate"] is not False:
                rate_monitors.append(
                    b2.PopulationRateMonitor(pop, name=pop_name + "_rate")
                )
            if rec_defs[pop_name]["spikes"] is not False:
                if pop_name == "exc":
                    # special override for excitatory to record all first
                    # and later drop randomly before saving, otherwise
                    # recording only from first n neurons, which heavily overlap
                    # with those stimulated, and the first few clusters
                    rec_idx = np.arange(N_exc) 
                else:
                    rec_idx = (
                        np.arange(rec_defs[pop_name]["spikes"]) 
                        if type(rec_defs[pop_name]["spikes"]) is int
                        else rec_defs[pop_name]["spikes"] #Change param_settings.record_Defs to 2000
                    )
                spike_monitors.append(
                    b2.SpikeMonitor(pop[rec_idx], name=pop_name + "_spikes")
                )

            if rec_defs[pop_name]["trace"] is not False:
                rec_idx = (
                    np.arange(rec_defs[pop_name]["trace"][1])
                    if type(rec_defs[pop_name]["trace"][1]) is int
                    else rec_defs[pop_name]["trace"][1]
                )
                trace_monitors.append(
                    b2.StateMonitor(
                        pop,
                        rec_defs[pop_name]["trace"][0],
                        record=rec_idx,
                        name=pop_name + "_trace",
                        dt=dt_ts,
                    )
                )

    ### collect into network object and return ###
    net_collect = b2.Network(b2.collect())  # magic collect all groups
    monitors = [rate_monitors, spike_monitors, trace_monitors]
    net_collect.add(monitors)
    return net_collect
