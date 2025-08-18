# Utility functions for accessing and saving data, directories, etc.
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import second
import pickle
from os import listdir
import h5py as h5
from time import strftime
from pathlib import Path
from glob import glob
from ..analysis import spikes_summary
from ..sim import b2_interface
from ..utils import dist_utils


def _filter_spikes_random(spike_trains, n_to_save):
    """Filter a subset of spike trains randomly for saving."""
    np.random.seed(42)
    record_subset = np.sort(
        np.random.choice(len(spike_trains), n_to_save, replace=False)
    )
    return {k: spike_trains[k] for k in spike_trains.keys() if k in record_subset}


def collect_spikes(net_collect, params_dict):
    """Collect all spikes from all spike monitors, filter a subset if necessary."""
    spike_dict = {}
    for sm in net_collect.objects:
        if "spikes" in sm.name:
            spike_trains = sm.spike_trains()
            pop_save_def = params_dict["params_settings"]["record_defs"][
                sm.name.split("_")[0]
            ]["spikes"]
            n_to_save = pop_save_def if type(pop_save_def) == int else len(pop_save_def)
            #n_to_save = len(spike_trains)
            if n_to_save == len(spike_trains):
                # recorded and to-be saved is the same length, go on a per usual
                spike_dict[sm.name] = b2_interface._deunitize_spiketimes(spike_trains)
            else:
                # recorded more than necessary, subselect for saving
                spike_dict[sm.name] = b2_interface._deunitize_spiketimes(
                    _filter_spikes_random(spike_trains, n_to_save) 
                )
    return spike_dict


def collect_timeseries(net_collect, params_dict):
    """Collect all timeseries data from the network collector."""
    timeseries = b2_interface.parse_timeseries(
        net_collect, params_dict["params_settings"]["record_defs"]
    )
    return timeseries


def collect_raw_data(net_collect, params_dict):
    """Collect all raw data from the network collector."""
    # get spikes
    spikes = collect_spikes(net_collect, params_dict)
    timeseries = collect_timeseries(net_collect, params_dict)

    all_pop_sampler = {pop.split("_")[0]: None for pop in spikes.keys()}
    # compute rates
    rates = spikes_summary.compute_poprate_from_spikes(
        spikes,
        t_collect=(0, net_collect.t / second),
        dt=params_dict["params_settings"]["dt_ts"] / second,
        dt_bin=params_dict["params_settings"]["dt"] / second,
        pop_sampler=all_pop_sampler,
        smooth_std=params_dict["params_analysis"]["smooth_std"],
    )
    timeseries = {**timeseries, **rates}
    return spikes, timeseries


def check_before_save_h5(h5_file, dataset_name, data):
    """Check if dataset exists before saving to h5 file."""
    if dataset_name not in h5_file:
        h5_file.create_dataset(dataset_name, data=data)


def save_spikes_h5(h5_file, params_dict, spikes):
    """Save spikes to h5 file."""
    run_id = f"{params_dict['params_settings']['batch_seed']}/{params_dict['params_settings']['random_seed']}/"
    for pop_name, pop_spikes in spikes.items():
        for cell, spks in pop_spikes.items():
            dataset_name = run_id + "spikes/" + pop_name + f"/{cell}"
            check_before_save_h5(
                h5_file,
                dataset_name,
                np.around(spks, params_dict["params_settings"]["t_sigdigs"]),
            )


def get_spikes_h5(h5_file, run_id):
    """Get spikes of a specific run ID (random_seed) from h5 file."""
    spikes = {}
    if run_id in h5_file:
        # found file
        for pop in h5_file[run_id + f"spikes/"].keys():
            spikes[pop] = {
                cell: np.array(
                    h5_file[run_id + f"spikes/{pop}/"][cell][()], dtype=float
                )
                for cell in h5_file[run_id + f"spikes/{pop}/"].keys()
            }

    return spikes


def save_timeseries_h5(h5_file, params_dict, timeseries):
    """Save timeseries data to h5 file."""
    run_id = f"{params_dict['params_settings']['batch_seed']}/{params_dict['params_settings']['random_seed']}/"
    for k, v in timeseries.items():
        if (k == "t") or (k == "t_ds"):
            # save dt and t_end
            check_before_save_h5(h5_file, run_id + "timeseries/dt", v[1] - v[0])
            check_before_save_h5(h5_file, run_id + "timeseries/t_end", v[-1] + v[1])
        else:
            check_before_save_h5(h5_file, run_id + "timeseries/" + k, v)


def save_h5_and_plot_raw(sim_collector, do_plot_ts=True):
    """Save raw data to h5 file and plot network rate."""
    # grab batch_seed from first run bc they're all identical
    batch_seed = sim_collector[0][0]["params_settings"]["batch_seed"]
    # create or access existing h5 file
    h5_file_path = (
        sim_collector[0][0]["path_dict"]["data"] + f"{batch_seed}_raw_data.hdf5"
    )

    with h5.File(h5_file_path, "a") as h5_raw_file:
        # loop over collector
        for params_dict, spikes, timeseries in sim_collector:
            # check if there's anything going on
            # basically don't bother plotting or saving anything if there is no spikes at all
            if (timeseries) and ((timeseries["exc_rate"] > 0).any()):
                if do_plot_ts:
                    # plot network rate
                    run_id = (
                        f'{batch_seed}_{params_dict["params_settings"]["random_seed"]}_'
                    )
                    fig_name = (
                        params_dict["path_dict"]["figures"]
                        + run_id
                        + "exc_rate_%s.pdf"
                        % (
                            "short"
                            if params_dict["params_analysis"]["early_stopped"]
                            else "full"
                        )
                    )

                    if not Path(fig_name).exists():
                        # doesn't exist, plot and save
                        fig = plt.figure(figsize=(12, 1.5))
                        t_plot = timeseries["t_ds"] <= 60
                        plt.plot(
                            timeseries["t_ds"][t_plot],
                            timeseries["exc_rate"][t_plot],
                            lw=0.2,
                        )
                        plt.title(
                            f'{batch_seed}_{params_dict["params_settings"]["random_seed"]}'
                        )
                        plt.autoscale(tight=True)
                        plt.savefig(fig_name)
                        plt.close(fig)

                if not params_dict["params_analysis"]["early_stopped"]:
                    # save only if the full simulation ran, but always plot
                    # save spikes
                    save_spikes_h5(h5_raw_file, params_dict, spikes)

                    ### QUICK HACK TO NOT SAVE RATES FOR NOW
                    # should almost surely change this hack but...
                    if "exc_rate" in timeseries.keys():
                        timeseries.pop("exc_rate")
                    if "inh_rate" in timeseries.keys():
                        timeseries.pop("inh_rate")

                    # save time series
                    save_timeseries_h5(h5_raw_file, params_dict, timeseries)

    return h5_file_path


def fill_params_dict(
    params_dict_orig, theta_samples, theta_priors, return_n_dicts=False
):
    """Take a template params_dict and fill in generated samples.

    Args:
        params_dict_orig (dict): template parameter dictionary (in nested format).
        theta_samples (dict or pd.Dataframe): samples, can be organized as a dictionary or dataframe / series.
        theta_priors (dict): prior hyperparameters, really just needed for b2 units.
        return_n_dicts (bool or int, optional): return a list of n dictionaries if int, otherwise return a single
            dictionary where some fields are arrays. Defaults to False.

    Returns:
        dict or list: dictionary (or list of dictionaries) with sampled values filled into the template parameter dict.
    """
    # decide if samples is in dict, which assumes it has b2 units attached, otherwise in dataframe,
    # so need to convert to array and add unit because brian2 units work completely mysteriously
    samples_in_dict = type(theta_samples) is dict

    if return_n_dicts:
        # return list of dictionaries for multiple samples
        # deep copy so we don't modify original
        params_dict = [copy.deepcopy(params_dict_orig) for n in range(return_n_dicts)]
    else:
        # just make a single copy where fields with multiple values are stored as arrays
        params_dict = copy.deepcopy(params_dict_orig)

    for param, val in theta_samples.items():
        if "params" in param:
            # check if it's a network parameter
            param_name = param.split(".")
            if param in theta_priors.keys():
                # copy unit
                unit = theta_priors[param]["unit"]
            else:
                # param not in prior dictionary, copy directly but warn
                unit = 1
                print(param + " has no prior. Copied as bare value without unit.")

            if return_n_dicts:
                vals = val if samples_in_dict else np.array(val) * unit
                for i_n in range(return_n_dicts):
                    # loop over dictionaries to fill in
                    params_dict[i_n][param_name[0]][param_name[1]] = vals[i_n]
            else:
                # put directly into dictionary, useful for saving so non-repeating stuff doesn't get saved
                params_dict[param_name[0]][param_name[1]] = (
                    val if samples_in_dict else np.array(val) * unit
                )

    return params_dict


def update_params_dict(params_dict, update_dict):
    """Update a configuration dictionary with new values."""
    # update non-default values
    for k, v in update_dict.items():
        p, p_sub = k.split(".")
        params_dict[p][p_sub] = v
    return params_dict


def pickle_file(full_path, to_pickled):
    """Save a pickled file."""
    with open(full_path, "wb") as handle:
        pickle.dump(to_pickled, handle)


def load_pickled(filename):
    """Load a pickled file."""
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
    return loaded


def save_params_priors(path, params_dict, priors):
    """Save parameters and priors to pickle files."""
    with open(path + "prior.pickle", "wb") as handle:
        pickle.dump(priors, handle)
    with open(path + "params_dict_default.pickle", "wb") as handle:
        pickle.dump(params_dict, handle)


def process_simulated_samples(sim_collector, df_prior_samples_batch):
    """Process successfully-run simulated samples and append to dataframe."""
    # make a copy of the df
    df_prior_samples_ran = df_prior_samples_batch.copy()

    # sort results by random_seed just in case pool returned in mixed order
    sort_idx = np.argsort(
        np.array([s[0]["params_settings"]["random_seed"] for s in sim_collector])
    )
    sim_collector = [sim_collector[i] for i in sort_idx]

    # check all seeds are equal and lined up
    if np.all(
        df_prior_samples_ran["params_settings.random_seed"].values
        == np.array([s[0]["params_settings"]["random_seed"] for s in sim_collector])
    ):
        ran_metainfo = [
            [
                s[0]["params_settings"]["sim_time"] / second,  # get rid of b2 units
                s[0]["params_settings"]["real_run_time"],
                s[0]["params_analysis"]["early_stopped"],
            ]
            for s in sim_collector
        ]
        df_prior_samples_ran = pd.concat(
            (
                df_prior_samples_ran,
                pd.DataFrame(
                    ran_metainfo,
                    index=df_prior_samples_ran.index,
                    columns=[
                        "params_settings.sim_time",
                        "params_settings.real_run_time",
                        "params_analysis.early_stopped",
                    ],
                ),
            ),
            axis=1,
        )

    else:
        print("misaligned; failed to append")

    return df_prior_samples_ran


def convert_spike_array_to_dict(spikes, fs, pop_name="exc"):
    """Convert tuple of spike times to dict with cell id.

    Args:
        spikes (tuple): tuple of list of spike time.
        fs (float): sampling frequency.
        pop_name (str, optional): Name of population. Defaults to "exc".

    Returns:
        _type_: _description_
    """
    return {
        pop_name: {
            "%i" % i_s: np.array(s, ndmin=1) / fs
            for i_s, s in enumerate(spikes)
            if s is not None
        }
    }


def decode_df_float_axis(indices, out_type=float, splitter="_", idx_pos=1):
    """Decode dataframe indices as array of values. For freqs, etc."""
    return np.array([i.split(splitter)[idx_pos] for i in indices], dtype=out_type)


def subselect_features(df, feature_str):
    """
    Select features from dataframe based on string.
    """
    if type(feature_str) == str:
        feature_str = [feature_str]

    all_features = []
    for f_str in feature_str:
        all_features += list(df.columns[df.columns.str.contains(f_str)])

    return all_features


def separate_feature_columns(
    df,
    col_sets=[
        "params",
        "isi",
        "burst",
        "pca",
        "psd",
    ],
):
    """Separate columns of dataframe into different sets."""
    return [subselect_features(df, c) for c in col_sets]


def make_subfolders(parent_path, subfolders=["data", "figures"]):
    """Make directories for simulations, parameters, and figures."""
    paths = {}
    for subfolder in subfolders:
        full_path_sub = parent_path + "/" + subfolder + "/"
        # MAKE ABSOLUTE PATH HERE
        Path(full_path_sub).mkdir(parents=True, exist_ok=True)
        paths[subfolder] = full_path_sub
    return paths


def get_subfolders(parent_path, subpath_reqs=None):
    """Get subfolders in a parent directory, optionally those that satisfy the subpath requirements."""
    folders = [parent_path + f + "/" for f in listdir(parent_path) if "." not in f]
    if subpath_reqs:
        # include only those that satisfy the subpath requirement
        # usually looking for a folder
        folders = [f for f in folders if Path(f + "/" + subpath_reqs).exists()]

    return folders


def collect_csvs(run_folders, filename, subfolder="", merge=True):
    """Collect csv files across analysis of multiple runs."""
    collector = []
    for rf in run_folders:
        data_path = rf + subfolder
        path_dict = extract_data_files(data_path, [filename])
        if path_dict[filename]:
            print(data_path + path_dict[filename])
            collector.append(pd.read_csv(data_path + path_dict[filename], index_col=0))

    if merge:
        return pd.concat(collector, ignore_index=True, axis=0)
    else:
        return collector


def set_seed_by_time(params_dict):
    """Set batch seed based on current time."""
    from time import time

    batch_seed = int((time() % 1) * 1e7)
    params_dict["params_settings"]["batch_seed"] = batch_seed
    return batch_seed, params_dict


def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    from numpy.random import seed as np_seed
    from torch import manual_seed as torch_seed
    from brian2 import seed as b2_seed

    for seed_fn in [np_seed, torch_seed, b2_seed]:
        seed_fn(seed)


def merge_theta_and_x(df_prior_samples_ran, random_seeds, data_rows):
    """Align and merge theta and x dataframes based on random seeds."""
    if type(data_rows) != type(df_prior_samples_ran):
        # make it into dataframe
        data_rows = pd.concat(data_rows)

    matching_idx = []
    for rs in random_seeds:
        matched_prior = df_prior_samples_ran.index[
            df_prior_samples_ran["params_settings.random_seed"] == int(rs)
        ]
        if len(matched_prior) == 1:
            # found exactly 1 prior sample that matches summary row
            matching_idx.append(matched_prior)
        else:
            print(f"{rs} has {len(matched_prior)} matched priors, drop.")
            data_rows.drop(index=rs, inplace=True)

    df_merged_data = pd.concat(
        (df_prior_samples_ran, data_rows.set_index(np.array(matching_idx).squeeze())),
        axis=1,
    )
    return df_merged_data


def extract_data_files(data_folder, file_type_list, verbose=True):
    """Extract data files from a folder based on file types."""
    run_files = listdir(data_folder)
    data_file_dict = {}

    data_file_dict["root_path"] = data_folder
    for file_type in file_type_list:
        file_candidates = [f for f in run_files if file_type in f]
        if verbose:
            print(file_type, file_candidates)
        if file_candidates != []:
            data_file_dict[file_type.split(".")[0]] = file_candidates[0]
        else:
            data_file_dict[file_type.split(".")[0]] = None

    return data_file_dict


def filter_df(df, filter_col, filter_match, return_cols=None, return_array=False):
    """Filter dataframe based on column and match."""
    df_filtered = df[df[filter_col] == filter_match]
    if return_cols is not None:
        df_filtered = df_filtered[return_cols]
    return df_filtered.values if return_array else df_filtered


# Helper function for sorting through stuff
def grab_xo_and_posterior_preds(
    df_xos,
    query_xo,
    df_posterior_sims,
    query_pp,
    cols_summary,
    log_samples=True,
    stdz_func=None,
    include_mapnmean=False,
):
    """Grab xo and posterior predictives based on queries. Used in final analysis notebooks."""
    # Grab original xo and posterior predictives
    df_xo = dist_utils.find_matching_xo(df_xos, query_xo)  # xo
    df_per_xo = filter_df(df_posterior_sims, "x_o", query_pp)  # predictives

    dfs_collect = [
        filter_df(df_per_xo, "inference.type", i_t)
        for i_t in df_per_xo["inference.type"].unique()
    ]  # split by type
    assert "samples" in str(
        dfs_collect[-1].head(1)["inference.type"].values
    )  # check that last entry are samples
    df_samples = (
        pd.concat(dfs_collect[1:]) if include_mapnmean else dfs_collect[-1]
    )  # exclude gt_resim, and optionally map/mean samples

    # Preprocessing and whatnot
    xo = dist_utils.log_n_stdz(df_xo[cols_summary].values, log_samples, stdz_func)
    samples = dist_utils.log_n_stdz(
        df_samples[cols_summary].values, log_samples, stdz_func
    )
    return df_xo, dfs_collect, df_samples, xo, samples


def collect_all_xo_and_posterior_preds(
    df_xos,
    df_posterior_sims,
    xo_queries,
    cols_features,
    cols_params,
    log_samples=True,
    stdz_func=None,
    include_mapnmean=False,
    sort_samples=True,
    sort_weights=None,
):
    """Collect all xo and posterior predictives based on queries. Used in final analysis notebooks."""
    df_collect, samples_x_collect, samples_theta_collect = [], [], []

    for i_q, xo_query in enumerate(xo_queries):
        df_xo = dist_utils.find_matching_xo(df_xos, xo_query)
        df_samples = filter_df(df_posterior_sims, "x_o", "%s_%s" % xo_query)

        # log and standardize before sorting or not
        xo = dist_utils.log_n_stdz(df_xo[cols_features].values, log_samples, stdz_func)

        dfs, xs, thetas = {}, {}, {}
        for i_t, s_type in enumerate(df_samples["inference.type"].unique()):
            if ("samples" in s_type) and (include_mapnmean):
                df_type = pd.concat(
                    [
                        filter_df(df_samples, "inference.type", tt)
                        for tt in df_samples["inference.type"].unique()
                        if tt != "gt_resim"
                    ]
                )
            else:
                df_type = filter_df(df_samples, "inference.type", s_type)

            samples = dist_utils.log_n_stdz(
                df_type[cols_features].values, log_samples, stdz_func
            )
            if (sort_samples) and (samples.shape[0] > 1):
                samples_sorted, dists, idx_sorted = dist_utils.sort_closest_to_xo(
                    xo, samples, top_n=None, weights=sort_weights
                )
                samples = samples_sorted
            else:
                idx_sorted = np.arange(samples.shape[0])

            for key_type in ["resim", "map", "mean", "samples"]:
                if key_type in s_type:
                    dfs[key_type] = df_type.iloc[idx_sorted]
                    xs[key_type] = samples
                    thetas[key_type] = df_type[cols_params].iloc[idx_sorted].values

        dfs["xo"] = df_xo
        xs["xo"] = xo
        thetas["xo"] = None

        df_collect.append(dfs)
        samples_x_collect.append(xs)
        samples_theta_collect.append(thetas)

    return df_collect, samples_x_collect, samples_theta_collect


### Functions for loading stuff upfront for posterior predictive analysis
def load_for_posterior_predictives_analyses(
    xo_type,
    feature_set,
    algorithm="NPE",
    sample_datetime=None,
    idx_inf=-1,
    using_copied=True,
):
    """Load all necessary files for posterior predictive analyses. Used in final analysis notebooks."""
    # load posterior files
    head_dir = "/slurm_r2/" if using_copied else "/"
    data_dir_str = (
        f"../data/{head_dir}/inference_r2/{xo_type}/{feature_set}/{algorithm}/"
        + (f"{sample_datetime}/" if sample_datetime else "")
    )
    data_dirs = sorted(glob(data_dir_str + "*-*/*/data/"))

    print("All relevant inference directories:")
    [print(f"+++ {d}") for d in data_dirs]
    print(" ----- ")
    print(f"Loading...{data_dirs[idx_inf]}...")
    path_dict = extract_data_files(
        data_dirs[idx_inf],
        [
            "posterior.pickle",
            "params_dict_analysis_updated.pickle",
            "summary_data_merged.csv",
            "raw_data.hdf5",
        ],
    )

    # also load cfg but only when not on local
    if not using_copied:
        import yaml

        with open(path_dict["root_path"] + "../.hydra/overrides.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            print(cfg)
    else:
        cfg = {}

    # load original xo
    df_xos = load_df_xos(xo_type)

    # load posterior pred and densities
    df_posterior_sims, posterior, params_dict_default = load_df_posteriors(path_dict)
    _, theta_minmax = dist_utils.standardize_theta(
        posterior.prior.sample((1,)), posterior
    )
    _, cols_isi, cols_burst, cols_pca, cols_psd = separate_feature_columns(
        df_posterior_sims
    )
    cols_params = posterior.names
    return (
        df_xos,
        df_posterior_sims,
        posterior,
        theta_minmax,
        (cols_params, cols_isi, cols_burst, cols_pca, cols_psd),
        params_dict_default,
        path_dict,
    )


def load_df_xos(xo_type, xo_path=None):
    """Load target observation data based on xo type.
    NEED TO CHANGE DATA PATHS.
    """
    # TO DO: CHANGE DATA PATHS
    if xo_path is None:
        if "simulation" in xo_type:
            load_path = "../data/adex_MKI-round1-testset/analysis_summary/MK1_summary_merged.csv"
            #    '/analysis_summary/MK1_summary_merged.csv'
        elif xo_type == "organoid":
            load_path = "../data/adex_MKI/analysis_organoid/organoid_summary.csv"
            df_xos = pd.read_csv(load_path, index_col=0)
        elif xo_type in ["allen-hc", "allen-vis", "allenvc"]:
            load_path = "../data/adex_MKI/analysis_allenvc/allenvc_summary.csv"
    else:
        load_path = xo_path

    # load
    df_xos = pd.read_csv(load_path, index_col=0)
    if xo_type == "organoid":
        from .organoid_utils import convert_date_to_int

        df_xos.insert(0, ["day"], convert_date_to_int(df_xos), allow_duplicates=True)
    return df_xos


def load_df_posteriors(path_dict):
    """Load posterior samples, network, and configurations."""
    df_posterior_sims = pd.read_csv(
        path_dict["root_path"] + path_dict["summary_data_merged"], index_col=0
    )
    print(f"{df_posterior_sims['x_o'].value_counts().values[0]} samples per xo.")
    posterior = load_pickled(path_dict["root_path"] + path_dict["posterior"])
    params_dict_default = load_pickled(
        path_dict["root_path"] + path_dict["params_dict_analysis_updated"]
    )
    return df_posterior_sims, posterior, params_dict_default

def sort_neurons(membership, sorting_method="cluster_identity"):
    """
    Sort neurons based on the specified method.

    Parameters:
        membership (list/array of 2D arrays): Membership arrays for each simulation.
        sorting_method (str): "cluster_identity" or "n_clusters".

    Returns:
        sorted_indices (list of arrays): Sorted indices for each simulation.
    """
    sorted_indices = []
    #Sort by whether neurons are in one cluster or two clusters
    
    if sorting_method == "cluster_identity":
        # Sort by the first cluster identity
        sorted_idx = np.argsort(membership[:, 0])
        sorted_indices.append(sorted_idx)
    elif sorting_method == "n_clusters":
    #Neurons in one cluster have the same values in both columns
        single = np.where(membership[:,0] == membership[:,1])
        double = np.where(membership[:,0] != membership[:,1])
        sorted_indices.append(single)
        sorted_indices.append(double)
    else:
        raise ValueError("Invalid sorting_method. Use 'cluster_identity' or 'n_clusters'.")
    return sorted_indices