### Functions for computing various summary statistics from spike data.

import numpy as np
from numba import jit
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA


def _filter_spiketimes(spiketimes, analysis_win):
    """
    Filter spiketimes such that spikes outside of
    analysis_win are removed before analysis.

    Args:
        spiketimes (array): Array of spiketimes in seconds, (n_spikes,) or (n_spikes,1)
        analysis_win (array): Start and end of analysis window in seconds, i.e, [window_t_min, window_t_max].

    Returns:
        array: array of filtered spiketimes.
    """
    window_t_min, window_t_max = analysis_win
    if window_t_max is None:
        return spiketimes[spiketimes >= window_t_min]
    else:
        return spiketimes[(spiketimes >= window_t_min) & (spiketimes < window_t_max)]


@jit(nopython=True)
def compute_basic_isi_features(spiketimes, min_num_spikes=2):
    """
    Compute single unit spiketrain statistics.

    Args:
        spiketimes (array): Spiketimes in seconds.
        min_num_spikes (int, optional): Minimum number of spikes required to be considered spiking. Defaults to 2.

    Returns:
        out (array): [num_spikes, isi_mean, isi_std, isi_25q, isi_50q, isi_75q]
    """
    NUM_FEATS = 6
    basic_features = np.nan * np.ones(NUM_FEATS)

    # number of spikes
    basic_features[0] = spiketimes.shape[0]
    if basic_features[0] >= min_num_spikes:
        # compute ISI
        isi = np.diff(spiketimes)
        # NOTE that empty lists will return nans for mean and std, which propagates in the average
        basic_features[1] = isi.mean()  # mean ISI
        basic_features[2] = isi.std()  # std of ISI
        basic_features[3:] = np.percentile(isi, np.array([25, 50, 75]))  # isi quartiles

    return basic_features


def compute_smoothed_SU_rate(
    spikes_list,
    t_collect,
    dt=0.01,
    smooth_std=0.05,
    smooth_win_length=7,
    return_rate=True,
):
    """Compute smoothed population rate from a list of spike times.

    Args:
        spikes_list (list): list of spike times for each neuron.
        t_collect (tuple): (t_start, t_end) of the collection window.
        dt (float, optional): time step of the rate. Defaults to 0.01.
        smooth_std (float, optional): std of Gaussian smoothing window in seconds. Defaults to 0.05.
        smooth_win_length (int, optional): length of smoothing window in stds. Defaults to 7.
        return_rate (bool, optional): return bin length-normalized rate. Defaults to True.

    Returns:
        tuple: (time bins, smoothed population rate)
    """
    t_bins = np.arange(t_collect[0], t_collect[1] + dt, dt)
    binned_spikes = np.vstack(
        [np.histogram(s, t_bins - dt / 2)[0] for s in spikes_list]
    )
    smo_window = signal.windows.gaussian(
        int(smooth_win_length * smooth_std / dt), int(smooth_std / dt)
    )
    smo_window /= smo_window.sum()

    smoothed_SU_rate = np.apply_along_axis(
        lambda m: np.convolve(m, smo_window, mode="same"), axis=1, arr=binned_spikes
    )
    if return_rate:
        smoothed_SU_rate /= dt
    return t_bins[:-1], smoothed_SU_rate


def compute_PCA(x, n_pcs):
    """
    Compute PCA on a matrix of data.

    Args:
        x (array): Data matrix. x has to be in shape [n_samples, n_features], so [T, n_neurons] for pop rate.
        n_pcs (int): Number of principal components to compute.

    Returns:
        PCA: PCA object fitted to the data.
    """
    pca = PCA(n_components=n_pcs)
    pca.fit(x)
    return pca


def compute_spiketrains_summary(spiketrains_dict, analysis_win, min_num_spikes):
    """Compute summary statistics for a dictionary of spike trains.

    Args:
        spiketrains_dict (dict): Dictionary of spike times for each neuron.
        analysis_win (array): Start and end of analysis window in seconds, i.e, [window_t_min, window_t_max].
        min_num_spikes (int): Minimum number of spikes required to be considered spiking.

    Returns:
        tuple: (feature_names, spike_summary)
    """

    # this should be the final output function

    # run dummy to get number of features
    NUM_BASIC_FEATURES = compute_basic_isi_features(np.array([])).shape[0]

    # set flag for whether to filter spikes
    do_filt_spikes = type(analysis_win) != type(None)

    # get id of cells to put into array as ints
    cell_ids = np.array(list(spiketrains_dict.keys()), dtype=int)

    # compute for each cell
    basic_features = np.zeros((len(cell_ids), NUM_BASIC_FEATURES))
    for i_c, (cell, spikes) in enumerate(spiketrains_dict.items()):
        if do_filt_spikes:
            # filter spikes
            spikes = _filter_spiketimes(spikes, analysis_win)
        basic_features[i_c, :] = compute_basic_isi_features(spikes, min_num_spikes)

    # append with cell id
    spiketrains_summary = np.hstack((cell_ids[:, None], basic_features))
    feature_names = [
        "cell_id",
        "isi_numspks",
        "isi_mean",
        "isi_std",
        "isi_25q",
        "isi_50q",
        "isi_75q",
    ]
    return feature_names, spiketrains_summary


def return_df_summary(spikes_dict, analysis_win, min_num_spikes):
    """Return a dataframe of spike train summaries.

    Args:
        spikes_dict (dict): Dictionary of spike times for each neuron.
        analysis_win (array): Start and end of analysis window in seconds, i.e, [window_t_min, window_t_max].
        min_num_spikes (int): Minimum number of spikes required to be considered spiking.

    Returns:
        Dataframe: Dataframe of spike train summaries.
    """
    collect = []
    for pop, pop_spks in spikes_dict.items():
        # get population name
        pop_name = pop.split("_")[0]
        # compute summaries
        feat_names, pop_summary = compute_spiketrains_summary(
            pop_spks, analysis_win, min_num_spikes
        )

        # put into dataframe and relabel
        df_summary = pd.DataFrame(pop_summary, columns=feat_names).set_index("cell_id")
        df_summary.index = [pop_name + "_%i" % int(i) for i in df_summary.index]
        collect.append(df_summary)
    return pd.concat(collect)


def grab_subpopulation(df_cells, pop_name, pop_sel):
    """Grab the spike stats from select or random cells within a pre-defined subpopulation (or all).

    Args:
        df_cells (Dataframe): Pandas dataframe of spike stats, indexed by subpop_i.
        pop_name (str): Name of the subpopulation to grab, or 'all'.
        pop_sel (None, list, array, float, or int):
            selection criterion for cells within the population.
            None: return all within the subpopulation.
            list or array: return indices of the subpopulation in the list/array.
            float: return a random proportion of the subpopulation, between 0. and 1.
            int: return a random number of cells within the subpopulation, between 0 and number of cells.

    Returns:
        Dataframe: rows of the spike summary df that satisfy the selection criterion.
    """
    if pop_name == "all":
        # just copy over if population is 'all'
        df_sub_pop = df_cells
    else:
        # get the appropriate subset of cells
        df_sub_pop = df_cells[[pop_name in i for i in df_cells.index]]

    # grab subset of population
    if isinstance(pop_sel, type(None)):
        # return all cells
        df_return = df_sub_pop

    elif isinstance(pop_sel, list) or isinstance(pop_sel, np.ndarray):
        # return the requested indices
        try:
            df_return = df_sub_pop.loc[[pop_name + "_%i" % i for i in pop_sel]]
        except:
            # failed, either population or index not found
            print(
                "Population '%s' not found OR index exceeds its size. Choosing %i from '%s' randomly..."
                % (pop_name, len(pop_sel), pop_name)
            )
            df_return = df_sub_pop.sample(n=len(pop_sel))
    else:
        # return integer or float (proportion of random cells)
        # if size is too big, pandas will throw an adequate error message
        if type(pop_sel) == float:
            df_return = df_sub_pop.sample(frac=pop_sel)
        elif type(pop_sel) == int:
            df_return = df_sub_pop.sample(n=pop_sel)

    return df_return


def get_population_average(df_cells, pop_sampler):
    """Get the average and standard deviation of spike stats for a population.

    Args:
        df_cells (Dataframe): Pandas dataframe of spike stats, indexed by subpop_i.
        pop_sampler (dict): Dictionary of population names and selection criteria.

    Returns:
        Dataframe: Dataframe of population average (_mu) and standard deviation (_sigma).
    """
    # pop_sampler is dictionary where key denotes population name, has to be a string that matches in the data
    # and the value is either how many random ones of the population (integer or float),
    # a list of cell indices (list), or None, which is everything.
    # If 'all' is one of the population names, then it supersedes the other population names and just grab from all

    # first append compute isi cv
    df_cells["isi_cv"] = df_cells["isi_std"] / df_cells["isi_mean"]

    # now flip through the population definitions to sample
    if "all" in pop_sampler.keys():
        # 'all' supersedes all population definitions, so return immediately
        df_sub_pop = grab_subpopulation(df_cells, "all", pop_sampler["all"])
    else:
        df_sub_pop = [
            grab_subpopulation(df_cells, pop_name, pop_sel)
            for pop_name, pop_sel in pop_sampler.items()
        ]
        df_sub_pop = pd.concat(df_sub_pop, axis=0)

    df_summary = pd.concat((df_sub_pop.mean(), df_sub_pop.std()), axis=0).to_frame().T
    df_summary.columns = [c + "_mu" for c in df_cells.columns] + [
        c + "_sigma" for c in df_cells.columns
    ]
    return df_summary


def bin_population_spikes(
    spikes, t_start, t_end, dt, return_fr=True, smooth_std=None, downsample_factor=None
):
    """Binarize a collection of spikes into one population rate vector.

    Args:
        spikes (dict or ndarray): dictionary or np array where each item/element is spike times from one neuron / channel
        t_start (float): start time of binning, in seconds
        t_end (float): end time of binning, in seconds
        dt (float): time step to bin the spikes, in seconds
        return_fr (bool, optional): Whether to return in spike count or cell firing rate. Defaults to True.
        smooth_std (float, optional): width of Gaussian smoothing window in seconds. Defaults to None.
        downsample_factor (int, optional): Downsampling factor. Defaults to None.

    Returns:
        t (array): time series timestamps
        binned_spikes (array): time series of population spike count/rate
    """
    # merge spike times into one array
    if type(spikes) is np.ndarray:
        # spikes are collected in an array over channels, just squash
        # need to pad a dimension for concatenating
        pop_spiketimes = np.hstack([s for s in spikes if s is not None])

    elif type(spikes) is dict:
        # spikes are in dict
        pop_spiketimes = np.hstack([v for v in spikes.values()])

    if pop_spiketimes.size > 0:
        # make time vector
        t = np.arange(t_start, t_end + dt, dt)
        # bin spikes
        # NOTE: make bins that center on the time steps
        binned_spikes = np.histogram(pop_spiketimes, t - dt / 2)[0]
        t = t[:-1]

        if return_fr:
            # return firing rate, i.e., normalize by # of cells and dt
            binned_spikes = binned_spikes / dt / len(spikes)

        if smooth_std:
            # smoothing window standard deviation is in units of seconds
            binned_spikes = smooth_with_gaussian(binned_spikes, dt, smooth_std)

        if (downsample_factor is not None) & (downsample_factor != 1):
            # downsample
            binned_spikes = signal.decimate(binned_spikes, q=downsample_factor)
            t = t[::downsample_factor]

        return t, binned_spikes

    else:
        t = np.arange(t_start, t_end + dt, dt)[:-1]
        if downsample_factor:
            t = t[::downsample_factor]
        return t, np.zeros_like(t)


def smooth_with_gaussian(x, dt, smooth_std, smooth_win_length=7):
    """Smooth a time series with a Gaussian window.

    Args:
        x (array): time series to smooth.
        dt (float): time step of the time series (sampling period).
        smooth_std (float): standard deviation of the Gaussian window in seconds.
        smooth_win_length (int, optional): length of the smoothing window in stds. Defaults to 7.

    Returns:
        array: smoothed time series.
    """
    smo_window = signal.windows.gaussian(
        int(smooth_win_length * smooth_std / dt), int(smooth_std / dt)
    )
    smo_window /= smo_window.sum()
    x_smoothed = np.convolve(x, smo_window, "same")
    return x_smoothed


def compute_poprate_from_spikes(
    spikes, t_collect, dt, dt_bin, pop_sampler, smooth_std=None
):
    """Compute population firing rate from all spikes of each population.

    Args:
        spikes (dict): dictionary of spikes
        t_collect (tuple): (t_start, t_end)
        dt (float): final bin width of rate (i.e., sampling period).
        dt_bin (float): initial bin width, downsampled later.
        pop_sampler (dict): which populations to compute rates for
        smooth_std (float, Optional): std of Gaussian window to smooth spikes, in seconds.

    Returns:
        dict: dict of firing rates.
    """
    rates = {}
    for k, v in spikes.items():
        if type(v) is dict:
            # make sure it's a dict, and not float
            pop_name = k.split("_")[0]
            if pop_name in pop_sampler:
                rates["t_ds"], rates[pop_name + "_rate"] = bin_population_spikes(
                    spikes[k],
                    t_start=t_collect[0],
                    t_end=t_collect[1],
                    dt=dt_bin,
                    return_fr=True,
                    smooth_std=smooth_std,
                    downsample_factor=int(dt / dt_bin),
                )
    return rates


def compute_average_pop_rates(pop_rates_dict, pop_sampler):
    """Get average of all rates included.

    Args:
        pop_rates_dict (dict): rates dictionary.
        pop_sampler (dict): defines all populations to average over.

    Returns:
        numpy array: array of averaged rate.
    """
    n_avg = 0
    all_rates = np.zeros_like(pop_rates_dict["t_ds"])
    for pop in pop_sampler.keys():
        if f"{pop}_rate" in pop_rates_dict.keys():
            all_rates = all_rates + pop_rates_dict[f"{pop}_rate"]
            n_avg += 1
        else:
            print(f'"{pop}_rate" does not exist.')
    return all_rates / n_avg


def _collect_epochs(x, indices, window=[-500, 2500], verbose=True):
    """Collect epochs / windows around indices.

    Args:
        x (array): time series.
        indices (array): indices to collect epochs around.
        window (list, optional): window around indices to grab. Defaults to [-500, 2500].
        verbose (bool, optional): print out skipped indices. Defaults to True.

    Returns:
        list: list of epochs.
    """
    epochs = []
    for i, idx in enumerate(indices):
        if (idx + window[0] < 0) or (idx + window[1] > len(x)):
            if not verbose:
                print("index %i (%i) skipped because near boundary." % (i, idx))
        else:
            epochs.append(x[int(idx + window[0]) : int(idx + window[1])])

    return epochs


def get_popburst_peaks(
    pop_rate,
    fs,
    min_burst_height,
    min_burst_height_ratio,
    min_burst_distance,
    burst_win,
    use_burst_prominence=True,
    burst_wlen=None,
    burst_rel_height=0.95,
    verbose=True,
    burst_max_window=[0, None],
):
    """Get population burst peaks and their statistics.

    Args:
        pop_rate (array): population rate time series.
        fs (float): sampling frequency.
        min_burst_height (float): minimum height of burst peaks.
        min_burst_height_ratio (float): minimum height ratio of burst peaks.
        min_burst_distance (float): minimum distance between burst peaks.
        burst_win (list): window around burst peak to collect.
        use_burst_prominence (bool, optional): use peak prominence for burst detection. Defaults to True.
        burst_wlen (float, optional): window length for peak width calculation. Defaults to None.
        burst_rel_height (float, optional): relative height for peak width calculation. Defaults to 0.95.
        verbose (bool, optional): Defaults to True.
        burst_max_window (list, optional): window to compute burst threshold. Defaults to [0, None].

    Returns:
        dict: dictionary of burst statistics.
    """
    # peak detection
    # get burst threshold from this window, for skipping the beginning
    idx_beg = int(burst_max_window[0] * fs)
    if not burst_max_window[1]:
        idx_end = len(pop_rate)
    else:
        idx_end = int(burst_max_window[1] * fs)

    if (idx_end < idx_beg) or (pop_rate[idx_beg:idx_end].max() < min_burst_height):
        # basically no activity, set high threshold to fail
        peak_threshold = pop_rate.max() * 2
    else:
        peak_threshold = pop_rate[idx_beg:idx_end].max() * min_burst_height_ratio

    # add a tiny bit of noise to avoid identical peak heights
    noise = np.random.normal(0, 1e-7, size=len(pop_rate))

    if use_burst_prominence:
        pk_ind, pk_prop = signal.find_peaks(
            x=pop_rate + noise,
            prominence=peak_threshold,
            distance=min_burst_distance * fs,
            wlen=int(burst_wlen * fs),
        )

        # get peak widths
        pk_proms = (
            pk_prop["prominences"],
            pk_prop["left_bases"],
            pk_prop["right_bases"],
        )
        pk_widths = signal.peak_widths(
            pop_rate + noise,
            pk_ind,
            rel_height=burst_rel_height,
            prominence_data=pk_proms,
        )
    else:
        pk_ind, pk_prop = signal.find_peaks(
            x=pop_rate + noise,
            height=peak_threshold,
            distance=min_burst_distance * fs,
        )
        pk_widths = signal.peak_widths(
            pop_rate + noise, pk_ind, rel_height=burst_rel_height
        )

    burst_height = pop_rate[pk_ind]

    # get cut windows
    window = [int(burst_win[0] * fs), int(burst_win[1] * fs)]
    kernels = _collect_epochs(pop_rate, pk_ind, window, verbose)
    return {
        "burst_times": pk_ind / fs,
        "burst_heights": burst_height,
        "burst_kernels": kernels,
        "burst_widths": pk_widths[0] / fs,
        "burst_width_heights": pk_widths[1],
        "burst_width_ips": (
            np.vstack([ips for ips in zip(pk_widths[2], pk_widths[3])]) / fs
            if pk_widths[2].any()
            else []
        ),
    }


def get_burst_subpeaks(
    pop_rate, fs, burst_stats, min_subpeak_prom_ratio, min_subpeak_distance
):
    """Get subpeaks within each burst.

    Args:
        pop_rate (array): population rate time series.
        fs (float): sampling frequency.
        burst_stats (dict): dictionary of burst statistics.
        min_subpeak_prom_ratio (float): minimum prominence ratio of subpeaks.
        min_subpeak_distance (float): minimum distance between subpeaks.

    Returns:
        dict: dictionary of burst statistics with subpeak information
    """
    ips = burst_stats["burst_width_ips"]
    b_times = burst_stats["burst_times"]
    subpeaks_times, burst_kernel_refined = [], []
    for i in range(len(b_times)):
        ips_idx = np.arange(int(ips[i][0] * fs), int(ips[i][1] * fs), dtype=int)
        if len(ips_idx) > 0:
            # add a little bit of noise in case constant amplitude peak
            noise = np.random.normal(0, 1e-7, size=len(ips_idx))
            curr_burst = pop_rate[ips_idx] + noise
            subpeaks, _ = signal.find_peaks(
                curr_burst,
                prominence=curr_burst.max() * min_subpeak_prom_ratio,
                distance=fs * min_subpeak_distance,
            )
            subpeaks_times.append((subpeaks + ips_idx[0]) / fs)
            burst_kernel_refined.append(curr_burst)

    burst_stats["subpeak_times"] = subpeaks_times
    burst_stats["burst_kernels_refined"] = burst_kernel_refined
    burst_stats["burst_mean_fr"] = np.array(
        [bk.mean() for i_b, bk in enumerate(burst_stats["burst_kernels_refined"])]
    ).squeeze()
    return burst_stats


def compute_burst_summaries(burst_stats):
    """Compute aggregate burst summary statistics. Mean, std, CV, etc. over bursts.

    Args:
        burst_stats (dict): dictionary of burst statistics.

    Returns:
        DataFrame: summary statistics of bursts.
    """
    # compute interburst interval
    pk_ibi = np.diff(burst_stats["burst_times"])
    # compute pairwise kernel correlation and take the upper triangle
    pw_corr = np.corrcoef(burst_stats["burst_kernels"])[
        np.triu_indices(len(burst_stats["burst_kernels"]), 1)
    ]
    # compute all the summary stats
    features_dict = {
        "burst_num": len(burst_stats["burst_times"]),
        "burst_interval_mean": pk_ibi.mean(),
        "burst_interval_std": pk_ibi.std(),
        "burst_interval_cv": pk_ibi.std() / pk_ibi.mean(),
        "burst_peak_fr_mean": burst_stats["burst_heights"].mean(),
        "burst_peak_fr_std": burst_stats["burst_heights"].std(),
        "burst_width_mean": burst_stats["burst_widths"].mean(),
        "burst_width_std": burst_stats["burst_widths"].std(),
        "burst_onset_time_mean": -(
            burst_stats["burst_width_ips"] - burst_stats["burst_times"][:, None]
        ).mean(0)[0],
        "burst_onset_time_std": (
            burst_stats["burst_width_ips"] - burst_stats["burst_times"][:, None]
        ).std(0)[0],
        "burst_offset_time_mean": (
            burst_stats["burst_width_ips"] - burst_stats["burst_times"][:, None]
        ).mean(0)[1],
        "burst_offset_time_std": (
            burst_stats["burst_width_ips"] - burst_stats["burst_times"][:, None]
        ).std(0)[1],
        "burst_corr_mean": pw_corr.mean(),
        "burst_corr_std": pw_corr.std(),
        "burst_corr_interval2nextpeak": stats.spearmanr(
            pk_ibi, burst_stats["burst_heights"][1:]
        )[0],
        "burst_corr_interval2prevpeak": stats.spearmanr(
            pk_ibi, burst_stats["burst_heights"][:-1]
        )[0],
    }
    if "subpeak_times" in burst_stats:
        n_subpeaks = np.array(
            [len(st) - 1 for st in burst_stats["subpeak_times"]]
        )  # take 1 off for main peak
        features_dict["burst_numsubpeaks_mean"] = n_subpeaks.mean()
        features_dict["burst_numsubpeaks_std"] = n_subpeaks.std()
        features_dict["burst_mean_fr_mean"] = (
            burst_stats["burst_mean_fr"] / burst_stats["burst_heights"]
        ).mean()
        features_dict["burst_mean_fr_std"] = (
            burst_stats["burst_mean_fr"] / burst_stats["burst_heights"]
        ).std()

    return pd.DataFrame(features_dict, index=[0])


def compute_psd_arr(x, t, nperseg, noverlap_ratio, analysis_window=(0, None)):
    """Compute power spectral density of a time series.

    Args:
        x (array): time series.
        t (array): time stamps.
        nperseg (float): length of window in seconds.
        noverlap_ratio (float): overlap ratio.
        analysis_window (tuple, optional): window to analyze. Defaults to (0, None).

    Returns:
        tuple: (frequencies, psd)
    """
    # get parameters
    fs = 1 / (t[1] - t[0])
    nperseg = int(fs * nperseg)
    noverlap = int(nperseg * noverlap_ratio)

    if analysis_window[0] is None:
        idx_beg = 0
    else:
        idx_beg = int(analysis_window[0] * fs)
    if analysis_window[1] is None:
        idx_end = len(x)
    else:
        idx_end = int(analysis_window[1] * fs)

    # compute PSD
    f_axis, psd = signal.welch(
        x[idx_beg:idx_end], fs=fs, nperseg=nperseg, noverlap=noverlap
    )
    return f_axis, psd


def compute_psd(rates_dict, analysis_params):
    """Compute power spectral density of population rates.

    Args:
        rates_dict (dict): dictionary of population rates.
        analysis_params (dict): analysis parameters.

    Returns:
        DataFrame: dataframe of population rate PSDs.
    """
    t_ds = rates_dict["t_ds"]
    psd_dict = {}
    for pop, rate in rates_dict.items():
        if pop != "t_ds":
            f_axis, psd_dict[pop] = compute_psd_arr(
                rate,
                t_ds,
                analysis_params["nperseg_ratio"],
                analysis_params["noverlap_ratio"],
                analysis_params["analysis_window"],
            )

    return pd.DataFrame(psd_dict, index=[f"psd_{f}" for f in f_axis]).T
