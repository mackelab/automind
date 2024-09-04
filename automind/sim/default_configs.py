### Various default configurations for the simulation and analysis.
import brian2 as b2
import numpy as np


ADEX_NEURON_DEFAULTS_ZERLAUT = {
    ### parameters taken from Zerlaut et al. 2018, JCompNeurosci
    # these are free for each different type of population (i.e., E and I)
    #### NEURON PARAMS ####
    "C": 200 * b2.pF,  # capacitance
    "g_L": 10 * b2.nS,  # leak conductance
    "v_rest": -65 * b2.mV,  # resting (leak) voltage
    "v_thresh": -50 * b2.mV,
    # note v_thresh for adex is not technically the spike cutoff
    #   but where exp nonlinearity kicks in, i.e., spike initiation
    #   spike threshold can be defined as 5*risetime from v_thresh
    "delta_T": 2 * b2.mV,  # exponential nonlinearity scaling
    "a": 4 * b2.nS,
    "tau_w": 500 * b2.ms,
    "b": 20 * b2.pA,
    "v_reset": -65 * b2.mV,  # reset voltage
    "t_refrac": 5 * b2.ms,
    "v_cut": 0 * b2.mV,  # threshold voltage
    "v_0_offset": [
        0 * b2.mV,
        4 * b2.mV,
    ],  # starting voltage mean & variance from resting voltage
    #### SYNAPSE PARAMS ####
    "E_ge": 0 * b2.mV,
    "E_gi": -80 * b2.mV,
    "Q_ge": 1 * b2.nS,
    "Q_gi": 5 * b2.nS,
    "tau_ge": 5 * b2.ms,
    "tau_gi": 5 * b2.ms,
    "tdelay2e": 0 * b2.ms,
    "tdelay2i": 0 * b2.ms,
    #### external input ####
    "N_poisson": 500,
    "Q_poisson": 1 * b2.nS,
    "poisson_rate": 1 * b2.Hz,
    "I_bias": 0 * b2.pA,
    "p_igniters": 1.0,
}

ADEX_NET_DEFAULTS = {
    # cell counts
    "N_pop": 2000,
    "exc_prop": 0.8,
    # 2 population weight matrix
    "p_e2e": 0.05,
    "p_e2i": 0.05,
    "p_i2e": 0.05,
    "p_i2i": 0.05,
    # clustered connectivity scaling
    "n_clusters": 1,
    "clusters_per_neuron": 2,
    "R_pe2e": 1,
    "R_Qe2e": 1,
    "order_clusters": False,
}

SIM_SETTINGS_DEFAULTS = {
    ### simulation default settings
    "experiment": None,
    "network_type": "adex",
    "sim_time": 1.1 * b2.second,
    "dt": 0.2 * b2.ms,
    "dt_ts": 1.0 * b2.ms,
    "batch_seed": 0,
    "random_seed": 42,
    "record_defs": {
        "exc": {"rate": True, "spikes": False, "trace": (["I"], np.arange(10))}
    },
    "save_sigdigs": 6,
    "t_sigdigs": 4,
    "integ_method": "euler",
    "real_run_time": 0.0,
}

ANALYSIS_DEFAULTS = {
    ### early stop settings
    "t_early_stop": 1.1 * b2.second,
    "early_stop_window": np.array([0.1, 1.1]),
    "early_stopped": False,
    "stop_fr_norm": (0.0001, 0.99),
    ### spike and rate summary settings
    "do_spikes": True,
    "pop_sampler": {"exc": None},
    "analysis_window": np.array([0.1, None]),
    "smooth_std": 0.0005,
    "dt_poprate": 1.0 * b2.ms,
    "min_num_spikes": 3,
    ### bursts
    "do_bursts": True,
    "use_burst_prominence": True,
    "min_burst_height": 5,
    "min_burst_height_ratio": 0.5,
    "min_burst_distance": 1.0,
    "burst_win": [-0.5, 2.5],
    "burst_wlen": 10,
    "burst_rel_height": 0.95,
    ### PSD settings
    "do_psd": False,
    "nperseg_ratio": 0.5,
    "noverlap_ratio": 0.75,
    "f_lim": 500,
    ### PCA settings
    "do_pca": False,
    "n_pcs": 100,
    "pca_bin_width": 10.0 * b2.ms,
    "pca_smooth_std": 50.0 * b2.ms,
}

MKI_pretty_param_names = [
    "$\% E:I$",
    r"$p_{E\rightarrow E}$",
    r"$p_{E\rightarrow I}$",
    r"$p_{I\rightarrow E}$",
    r"$p_{I\rightarrow I}$",
    r"$R_{p_{E-clus}}$",
    r"$R_{Q_{E-clus}}$",
    r"$\% input_E$",
    r"$\# clus$",
    r"C",
    r"$g_L$",
    r"$V_{rest}$",
    r"$V_{thresh}$",
    r"$V_{reset}$",
    r"$t_{refrac}$",
    r"$\Delta T$",
    r"$g_{adpt}$ (a)",
    r"$Q_{adpt}$ (b)",
    r"$\tau_{adpt}$",
    r"$E_{I\rightarrow E}$",
    r"$Q_{E\rightarrow E}$",
    r"$Q_{I\rightarrow E}$",
    r"$\tau_{E\rightarrow E}$",
    r"$\tau_{I\rightarrow E}$",
    r"$\nu_{input\rightarrow E}$",
    r"$Q_{E\rightarrow I}$",
    r"$Q_{I\rightarrow I}$",
    r"$\nu_{input\rightarrow I}$",
]

MKI_3col_plot_order = np.array(
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        -1,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        20,
        21,
        25,
        26,
        19,
        22,
        23,
        24,
        27,
        -1,
    ]
)
