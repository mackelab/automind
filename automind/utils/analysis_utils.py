### Utility functions for computing summary features from simulations data.
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from brian2 import second
from ..analysis import spikes_summary
from . import data_utils


def compute_spike_features_only(spikes, params_dict):
    """Compute single unit ISI features.

    Args:
        spikes (dict): Dictionary of spike times.
        params_dict (dict): Dictionary containing analysis configurations.

    Returns:
        DataFrame: Dataframe of single unit ISI features.
    """
    # summary function for just SU features, usually for training restricted prior
    if "t_end" in spikes.keys():
        # this has to be popped because downstream analysis looks at all keys as population definitions
        t_end = spikes.pop("t_end")
    # compute single unit isi features
    df_spikes_SU = spikes_summary.return_df_summary(
        spikes,
        params_dict["params_analysis"]["analysis_window"],
        params_dict["params_analysis"]["min_num_spikes"],
    )
    df_spikes = spikes_summary.get_population_average(
        df_spikes_SU, params_dict["params_analysis"]["pop_sampler"]
    )
    return df_spikes


def compute_burst_features(rate, params_dict):
    """Compute burst features from population rate.

    Args:
        rate (array): Population rate.
        params_dict (dict): Dictionary containing analysis configurations.

    Returns:
        tuple: (DataFrame, dict) Dataframe of burst features and burst statistics.
    """
    # get burst event statistics
    burst_analysis_params = {
        k: v for k, v in params_dict["params_analysis"].items() if "burst_" in k
    }
    burst_stats = spikes_summary.get_popburst_peaks(
        rate,
        fs=second / params_dict["params_analysis"]["dt_poprate"],
        **burst_analysis_params,
    )
    # get subpeak features if required
    if "min_subpeak_prom_ratio" in params_dict["params_analysis"]:
        burst_stats = spikes_summary.get_burst_subpeaks(
            rate,
            second / params_dict["params_analysis"]["dt_poprate"],
            burst_stats,
            params_dict["params_analysis"]["min_subpeak_prom_ratio"],
            params_dict["params_analysis"]["min_subpeak_distance"],
        )
    # get stats in dataframe
    if len(burst_stats["burst_kernels"]) > 1:
        # compute burst features
        df_bursts = spikes_summary.compute_burst_summaries(burst_stats)
    else:
        # return empty df row otherwise it screws up the final dataframe
        col_names = [
            "burst_num",
            "burst_interval_mean",
            "burst_interval_std",
            "burst_interval_cv",
            "burst_peak_fr_mean",
            "burst_peak_fr_std",
            "burst_width_mean",
            "burst_width_std",
            "burst_onset_time_mean",
            "burst_onset_time_std",
            "burst_offset_time_mean",
            "burst_offset_time_std",
            "burst_corr_mean",
            "burst_corr_std",
            "burst_corr_interval2nextpeak",
            "burst_corr_interval2prevpeak",
            "burst_numsubpeaks_mean",
            "burst_numsubpeaks_std",
            "burst_mean_fr_mean",
            "burst_mean_fr_std",
        ]
        df_bursts = pd.DataFrame(columns=col_names)

    return df_bursts, burst_stats


def compute_spike_burst_features(spikes, params_dict):
    """Compute single unit ISI and burst features."""
    if "t_end" in spikes.keys():
        # this has to be popped because downstream analysis looks at all keys as population definitions
        t_end = spikes.pop("t_end")
    else:
        print("No end time in spikes_dict, defaulting to 200.1s.")
        t_end = 200.1

    # compute single unit isi features
    df_spikes_SU = spikes_summary.return_df_summary(
        spikes,
        params_dict["params_analysis"]["analysis_window"],
        params_dict["params_analysis"]["min_num_spikes"],
    )
    df_spikes = spikes_summary.get_population_average(
        df_spikes_SU, params_dict["params_analysis"]["pop_sampler"]
    )

    # compute population rate
    pop_rates = spikes_summary.compute_poprate_from_spikes(
        spikes,
        t_collect=(0, t_end),
        dt=params_dict["params_analysis"]["dt_poprate"] / second,
        dt_bin=params_dict["params_settings"]["dt"] / second,
        pop_sampler=params_dict["params_analysis"]["pop_sampler"],
        smooth_std=params_dict["params_analysis"]["smooth_std"],
    )
    pop_rates["avgpop_rate"] = spikes_summary.compute_average_pop_rates(
        pop_rates, params_dict["params_analysis"]["pop_sampler"]
    )
    # get burst event statistics
    burst_analysis_params = {
        k: v for k, v in params_dict["params_analysis"].items() if "burst_" in k
    }
    burst_stats = spikes_summary.get_popburst_peaks(
        pop_rates["avgpop_rate"],
        fs=second / params_dict["params_analysis"]["dt_poprate"],
        **burst_analysis_params,
    )

    if "min_subpeak_prom_ratio" in params_dict["params_analysis"]:
        burst_stats = spikes_summary.get_burst_subpeaks(
            pop_rates["avgpop_rate"],
            second / params_dict["params_analysis"]["dt_poprate"],
            burst_stats,
            params_dict["params_analysis"]["min_subpeak_prom_ratio"],
            params_dict["params_analysis"]["min_subpeak_distance"],
        )

    if len(burst_stats["burst_kernels"]) > 1:
        # compute burst features
        df_bursts = spikes_summary.compute_burst_summaries(burst_stats)

        # merge spike and network features
        return pd.concat((df_spikes, df_bursts), axis=1), pop_rates, burst_stats
    else:
        return df_spikes, pop_rates, burst_stats


def compute_summary_features(spikes, params_dict):
    """Compute all summary features from simulation data."""
    if "t_end" in spikes.keys():
        t_end = spikes["t_end"]
    else:
        t_end = 200.1

    result_collector = {}

    #### compute single unit isi features
    if params_dict["params_analysis"]["do_spikes"]:
        df_spikes_SU = spikes_summary.return_df_summary(
            # get just spikes in popsampler
            {
                k: v
                for k, v in spikes.items()
                if k.split("_")[0]
                in params_dict["params_analysis"]["pop_sampler"].keys()
            },
            params_dict["params_analysis"]["analysis_window"],
            params_dict["params_analysis"]["min_num_spikes"],
        )
        df_spikes = spikes_summary.get_population_average(
            df_spikes_SU, params_dict["params_analysis"]["pop_sampler"]
        )
        result_collector["summary_spikes"] = df_spikes

    # compute unsmoothed total population rate
    dt = params_dict["params_analysis"]["dt_poprate"] / second
    pop_rates_raw = spikes_summary.compute_poprate_from_spikes(
        spikes,
        t_collect=(0, t_end),
        dt=dt,
        dt_bin=dt,
        # pop_sampler=params_dict["params_analysis"]["pop_sampler"],
        pop_sampler={k.split("_")[0]: None for k in spikes.keys() if k != "t_end"},
    )
    pop_rates_raw["avgpop_rate"] = spikes_summary.compute_average_pop_rates(
        pop_rates_raw, params_dict["params_analysis"]["pop_sampler"]
    )

    ##### get PSD
    if params_dict["params_analysis"]["do_psd"]:
        df_psd = spikes_summary.compute_psd(
            pop_rates_raw, params_dict["params_analysis"]
        )
        result_collector["summary_psd"] = df_psd

    ###### get burst features
    if params_dict["params_analysis"]["do_bursts"]:
        # smooth population rates
        pop_rates_smo = pop_rates_raw.copy()
        for pop in params_dict["params_analysis"]["pop_sampler"].keys():
            pop_rates_smo[pop + "_rate"] = spikes_summary.smooth_with_gaussian(
                pop_rates_raw[pop + "_rate"],
                dt,
                params_dict["params_analysis"]["smooth_std"],
            )

        pop_rates_smo["avgpop_rate"] = spikes_summary.compute_average_pop_rates(
            pop_rates_smo, params_dict["params_analysis"]["pop_sampler"]
        )
        df_burst, burst_stats = compute_burst_features(
            pop_rates_smo["avgpop_rate"], params_dict
        )

        result_collector["summary_bursts"] = df_burst
        result_collector["summary_burst_stats"] = burst_stats
        result_collector["pop_rates"] = pop_rates_smo
    else:
        # if no bursts, just return the raw population rates, unsmoothed
        result_collector["pop_rates"] = pop_rates_raw

    ##### get PCA
    if params_dict["params_analysis"]["do_pca"]:
        df_pca = compute_pca_features(spikes, (0, t_end), params_dict)
        result_collector["summary_pca"] = df_pca
    return result_collector


def compute_correlations(np_array, method="pearson"):
    """Compute correlation coefficients and p-values for all pairs of variables in a numpy array."""
    # Get the number of variables (columns)
    n_vars = np_array.shape[1]

    # Initialize matrices to store correlation coefficients and p-values
    corr_matrix = np.zeros((n_vars, n_vars))
    p_values = np.ones((n_vars, n_vars))

    # Compute correlation coefficient and p-value for each pair of variables
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if method == "pearson":
                corr, pval = pearsonr(np_array[:, i], np_array[:, j])
            elif method == "spearman":
                corr, pval = spearmanr(np_array[:, i], np_array[:, j])

            corr_matrix[i, j] = corr_matrix[j, i] = corr
            p_values[i, j] = p_values[j, i] = pval

    return corr_matrix, p_values


def compute_participation(df_pca, up_to_npcs=10, norm_by=None):
    """Compute participation ratio for a given number of PCs."""
    # sum squared divided by sum of squares
    if norm_by is None:
        norm_by = up_to_npcs
    eigvals = df_pca.values[:, :up_to_npcs]
    pr = (eigvals.sum(1)) ** 2 / (eigvals**2).sum(1) / norm_by
    return pr


def check_ei_param_order(param_names):
    assert param_names.index("params_Epop.Q_ge") == 20
    assert param_names.index("params_Epop.tau_ge") == 22
    assert param_names.index("params_Epop.v_rest") == 11
    assert param_names.index("params_Epop.poisson_rate") == 24
    assert param_names.index("params_Epop.p_igniters") == 7

    assert param_names.index("params_Epop.Q_gi") == 21
    assert param_names.index("params_Epop.tau_gi") == 23
    assert param_names.index("params_Epop.E_gi") == 19
    assert param_names.index("params_Ipop.poisson_rate") == 27

    assert param_names.index("params_net.exc_prop") == 0
    assert param_names.index("params_net.p_e2e") == 1
    assert param_names.index("params_net.p_i2e") == 3
    assert param_names.index("params_net.R_pe2e") == 5
    assert param_names.index("params_net.R_Qe2e") == 6
    assert param_names.index("params_net.n_clusters") == 8


def check_tau_param_order(param_names):
    assert param_names.index("params_Epop.C") == 9
    assert param_names.index("params_Epop.g_L") == 10


def check_v_param_order(param_names):
    assert param_names.index("params_Epop.v_rest") == 11
    assert param_names.index("params_Epop.v_thresh") == 12
    assert param_names.index("params_Epop.v_reset") == 13
    assert param_names.index("params_Epop.E_gi") == 19


def compute_composite_tau(samples, param_names=None):
    """Computes single neuron membrane time constant in ms, i.e., tau = C/g_L"""
    if param_names is not None:
        check_tau_param_order(param_names)
    C, g_L = np.hsplit(samples[:, [9, 10]], 2)
    return C / g_L


def compute_composite_v_diff(samples, param_names=None):
    """Compute voltage difference between various and resting.
    v_thresh - v_rest, v_reset - v_rest, E_gi - v_rest
    """
    if param_names is not None:
        check_v_param_order(param_names)
    v_rest, v_thresh, v_reset, E_gi = np.hsplit(samples[:, [11, 12, 13, 19]], 4)
    return np.hstack([v_thresh - v_rest, v_reset - v_rest, E_gi - v_rest])


def compute_composite_ei(samples, param_names=None, make_clus_adjust=False):
    """quant, count, total rec, input"""
    # Two current issues with this:
    #    1. what to do when E_ie is above the resting potential, i.e., no inhibition
    #       another problem: cannot use v_rest for Vavg
    #       maybe check here? https://neuronaldynamics.epfl.ch/online/Ch13.S6.html

    #    2. DONE how to account for the cluster amplification
    #    (plus the variable checking)

    if param_names is not None:
        check_ei_param_order(param_names)
    N_neurons = 2000  # constant number of neurons per network
    N_poissons = 500  # hardcoded number of poisson inputs
    Q_poisson = 1.0  # hardcoded poisson synaptic conductance
    ### E parameters
    Q_ge, tau_ge, V_rest, poissonE_rate, p_igniters = np.hsplit(
        samples[:, [20, 22, 11, 24, 7]], 5
    )
    E_ge = 0  # hardcoded reversal potential

    ### I parameters
    Q_gi, tau_gi, E_gi, poissonI_rate = np.hsplit(samples[:, [21, 23, 19, 27]], 4)

    ### network parameters
    exc_prop, p_e2e, p_i2e, R_pe, R_Qe, n_clusters = np.hsplit(
        samples[:, [0, 1, 3, 5, 6, 8]], 6
    )
    clus_per_neuron = 2  # hardcoded, each neuron belongs to 2 clusters

    ### NOT SURE IF V_rest is the right thing to use here, in general NOT Vavg

    # E synapse quantile charge
    exc_quant = Q_ge * tau_ge  # * (E_ge - V_rest)
    # E recurrent synapse count per neuron
    exc_count = p_e2e * exc_prop * N_neurons
    # cluster amplification of conductance
    exc_quant_clus = exc_quant * R_Qe
    # cluster amplification of intra-cluster connectivity
    exc_count_clus = (
        (exc_count / np.floor(np.maximum(n_clusters, 1.0)))
        * clus_per_neuron
        * (R_pe - 1.0)
    )

    # total recurrent E charge: baseline + intra-cluster
    exc_rec_total = exc_quant * exc_count
    exc_rec_clus = exc_quant_clus * exc_count_clus
    if make_clus_adjust:
        # Adjust for the clustering
        exc_count += exc_count_clus
        exc_rec_total += exc_rec_clus
        exc_quant = exc_rec_total / exc_count

    # external input to E charge
    exc_ext = (
        poissonE_rate * p_igniters * N_poissons * (Q_poisson * tau_ge * (E_ge - V_rest))
    )

    # I synapse quantile charge
    inh_quant = Q_gi * tau_gi  # * (-(E_gi - V_rest))

    # print((inh_quant<0).sum(), len(inh_quant))

    # I recurrent synapse count
    inh_count = p_i2e * (1.0 - exc_prop) * N_neurons
    # total recurrent I charge
    inh_rec_total = inh_quant * inh_count
    # external input to I charge
    inh_ext = poissonI_rate * N_poissons * (Q_poisson * tau_ge * (E_ge - V_rest))

    return np.hstack([exc_quant, exc_count, exc_rec_total, exc_ext]), np.hstack(
        [inh_quant, inh_count, inh_rec_total, inh_ext]
    )


def discard_by_avg_power(
    df_summary, f_band=(2, 10), power_thresh=[1e-10, None], return_idx=False
):
    """Discard simulations based on average power in a given frequency range."""
    # discard by average power in a given frequency range
    cols_psd = data_utils.subselect_features(df_summary, ["psd"])
    f_axis = data_utils.decode_df_float_axis(cols_psd)
    f_sel_idx = (f_axis >= f_band[0]) & (f_axis <= f_band[1])
    avg_power = df_summary[cols_psd].iloc[:, f_sel_idx].mean(1)
    if (power_thresh[0] == "None") or (power_thresh[0] is None):
        power_thresh[0] = -np.inf
    if (power_thresh[1] == "None") or (power_thresh[1] is None):
        power_thresh[1] = np.inf
    idx_keep = (avg_power > power_thresh[0]) & (avg_power < power_thresh[1])
    # Return good indices if requested, otherwise the dataframe
    if return_idx:
        return idx_keep
    else:
        return df_summary[idx_keep]


def manual_filter_logPSD(
    log_psd,
    f_axis,
    f_bounds=[1, 490],
    bounds_hyperactive=(6.5, 13),
    bounds_silent=(7e-4, 0.14),
):
    """
    Filter out bad simulations based on per-sample PSD statistics.
    Bounds are hard-coded, not ideal but gets the job done.
    Returns indices of good ones.
    """
    f_idx = np.logical_and(f_axis >= f_bounds[0], f_axis <= f_bounds[1])
    f_axis = f_axis[f_idx]
    log_psd = log_psd[:, f_idx]

    # manual conditions for bad simulations
    logpsd_range = log_psd.max(1) - log_psd.min(1)

    bad_idx_hyperactive = (log_psd.var(1) > bounds_hyperactive[0]) & (
        (logpsd_range) > bounds_hyperactive[1]
    )  # crazy active
    bad_idx_silent = (log_psd.var(1) < bounds_silent[0]) & (
        (logpsd_range) < bounds_silent[1]
    )  # basically no activity
    bad_idx_infnans = np.isinf(log_psd).any(axis=1) | np.isnan(log_psd).any(
        axis=1
    )  # nans or infs

    good_idx = ~(
        bad_idx_infnans | bad_idx_silent | bad_idx_hyperactive
    )  # toss any bad ones
    return good_idx


def compute_pca_features(spikes, t_collect, params_dict):
    """Compute PCA features from spikes."""
    # aggregate all spikes in the sampled populations
    spikes_list = sum(
        [
            list(spikes[f"{pop}_spikes"].values())
            for pop in params_dict["params_analysis"]["pop_sampler"].keys()
        ],
        [],
    )
    # bin and smooth
    t_bins, SU_rate = spikes_summary.compute_smoothed_SU_rate(
        spikes_list,
        t_collect,
        params_dict["params_analysis"]["pca_bin_width"] / second,
        params_dict["params_analysis"]["pca_smooth_std"] / second,
    )
    # do PCA
    pca = spikes_summary.compute_PCA(SU_rate.T, params_dict["params_analysis"]["n_pcs"])
    # record mean variance
    var_total = (SU_rate).var(1).mean()
    # record variance explained ratios
    df_var_exp = pd.DataFrame(
        np.hstack([var_total, pca.explained_variance_ratio_]).T,
        index=["pca_total"]
        + [f"pca_{i+1}" for i in np.arange(params_dict["params_analysis"]["n_pcs"])],
    )

    return df_var_exp.T
