### Script and helpers for running analysis (i.e., summary stats) on simulation data.

import numpy as np
import pandas as pd
import h5py as h5
from time import time
import os
from multiprocessing import Pool
from functools import partial
from ..utils import data_utils, plot_utils, analysis_utils


def set_up_analysis(run_folder, analysis_settings_update, ref_params_dict_path=None):
    """Set up analysis by loading all necessary files and updating the params dict.

    Args:
        run_folder (str): path to the simulation folder.
        analysis_settings_update (dict): dictionary of analysis settings to update.
        ref_params_dict_path (str, optional): Path to reference configurations. Defaults to None.

    Returns:
        tuple: batch_seed, data_file_dict, prior, params_dict_default, df_prior_samples_ran
    """
    # extract relevant file names
    file_type_list = [
        "prior.pickle",
        "raw_data.hdf5",
        "prior_samples_ran.csv",
        "summary_data_merged.csv",
        "params_dict_default.pickle",
    ]
    data_folder = run_folder + "data/"
    data_file_dict = data_utils.extract_data_files(data_folder, file_type_list)
    batch_seed = data_file_dict["prior_samples_ran"].split("_")[0]

    # load everything
    prior = data_utils.load_pickled(data_folder + data_file_dict["prior"])
    params_dict_default = data_utils.load_pickled(
        data_folder + data_file_dict["params_dict_default"]
    )
    df_prior_samples_ran = pd.read_csv(
        data_folder + data_file_dict["prior_samples_ran"], index_col=0
    )

    if ref_params_dict_path:
        # load reference parameter set
        params_dict_updated = data_utils.load_pickled(ref_params_dict_path)

        # swap in the new analysis sub-dict and path dicts
        params_dict_default["params_analysis"] = params_dict_updated["params_analysis"]
        path_dict = data_utils.make_subfolders(run_folder, ["figures"])
        params_dict_default["path_dict"] = path_dict

    # update with new setting
    params_dict_default = data_utils.update_params_dict(
        params_dict_default, analysis_settings_update
    )

    return batch_seed, data_file_dict, prior, params_dict_default, df_prior_samples_ran


def make_summary_parallel(
    run_folder,
    analysis_settings_update,
    reference_params_path=None,
    num_cores=64,
    n_per_batch=200,
    overwrite_existing=False,
):
    """Script to run (parallelized) analysis on a simulation folder, streams while analyzing.

    Args:
        run_folder (str): path to the simulation folder, should be one level above '/data/' folder that contains '{batchseed}_raw.hd5f' file.
        analysis_settings_update (dict): dictionary of analysis settings to update.
        reference_params_path (str, optional): Path to reference configurations. Defaults to None.
        num_cores (int, optional): Number of cores to use. Defaults to 64.
        n_per_batch (int, optional): Number of simulations to process per batch. Defaults to 200.
        overwrite_existing (bool, optional): Overwrite existing summary file. Defaults to False.
    """

    if not os.path.isdir(run_folder + "/data/"):
        print("Invalid simulation folder, skip.")
        return

    (
        batch_seed,
        data_file_dict,
        prior,
        params_dict_default,
        df_prior_samples_ran,
    ) = set_up_analysis(run_folder, analysis_settings_update, reference_params_path)
    analysis_name = params_dict_default["params_analysis"]["summary_set_name"]

    # get all random seeds
    h5_file_path = data_file_dict["root_path"] + data_file_dict["raw_data"]
    with h5.File(h5_file_path, "r") as h5_file:
        random_seeds = list(h5_file[batch_seed].keys())
        random_seeds = list(
            np.array(random_seeds)[np.argsort([int(s) for s in random_seeds])]
        )

    print(len(random_seeds), "random seeds found in h5 file")
    print(len(df_prior_samples_ran), "random seeds found in prior samples file")

    # check if random seeds are unique
    n_samples = len(random_seeds)
    print(
        f"Random seeds of runs in h5 file are unique: {n_samples==len(np.unique(random_seeds))}"
    )

    # check if stream file already exists
    stream_file_path = (
        data_file_dict["root_path"] + f"{batch_seed}_{analysis_name}_summary_stream.csv"
    )
    if os.path.isfile(stream_file_path):
        print(f"{stream_file_path} exists, deleting...")
        os.remove(stream_file_path)

    # check if save file already exists
    save_file_path = (
        data_file_dict["root_path"]
        + f"{batch_seed}_{analysis_name}_summary_data_merged.csv"
    )
    if os.path.isfile(save_file_path):
        print(f"{save_file_path} exists.")
        if not overwrite_existing:
            df_summary_stream_previous = pd.read_csv(save_file_path, index_col=0)
            if np.all(
                [
                    int(rs)
                    in df_summary_stream_previous["params_settings.random_seed"].values
                    for rs in random_seeds
                ]
            ):
                print("Skipping analysis.")
                return
            else:
                print(f"File exists but incomplete, overwrite.")
        else:
            print(f"File exists but overwriting.")

    ### RUN ANALYSIS ###
    print("----  Running analysis... ----")
    n_batches = int(np.ceil(n_samples / n_per_batch))
    print(f"{n_samples} files / {n_batches} batches to process.")

    for i_b in range(n_batches):
        start_time = time()

        i_start, i_end = i_b * n_per_batch, (i_b + 1) * n_per_batch
        random_seeds_batch = random_seeds[i_start:i_end]

        # grab spikes and process
        with h5.File(h5_file_path, "r") as h5_file:
            spikes_collected = []
            # iterate through batches
            for random_seed in random_seeds_batch:
                matched_prior = df_prior_samples_ran[
                    df_prior_samples_ran["params_settings.random_seed"]
                    == int(random_seed)
                ]
                if len(matched_prior) == 1:
                    # only one match
                    # check if early stopped

                    if matched_prior["params_analysis.early_stopped"].values:
                        print(
                            f"{batch_seed}-{random_seed} early stopped run escaped, remove."
                        )
                        # remove the seed from this batch
                        random_seeds_batch.remove(random_seed)
                    else:
                        # add to processing queue
                        spikes_dict = data_utils.get_spikes_h5(
                            h5_file, f"{batch_seed}/{random_seed}/"
                        )
                        spikes_dict["t_end"] = matched_prior[
                            "params_settings.sim_time"
                        ].item()
                        spikes_collected.append(spikes_dict)
                else:
                    print(f"{random_seed} with {len(matched_prior)} matches, skip.")

        # process in parallel
        # find the appropriate analysis function
        if analysis_name == "prescreen":
            # just single unit spiketrain analysis
            if num_cores == 1:
                summary_dfs = [
                    analysis_utils.compute_spike_features_only(s, params_dict_default)
                    for s in spikes_collected
                ]
            else:
                func_analysis = partial(
                    analysis_utils.compute_spike_features_only,
                    params_dict=params_dict_default,
                )
                with Pool(num_cores) as pool:
                    # NOTE: POSSIBLE FAILURE HERE WHEN RETURN IS EMPTY
                    summary_dfs = pool.map(func_analysis, spikes_collected)

        elif analysis_name == "spikes_bursts":
            # full burst analysis
            func_analysis = partial(
                analysis_utils.compute_spike_burst_features,
                params_dict=params_dict_default,
            )
            with Pool(num_cores) as pool:
                # POSSIBLE FAILURE HERE WHEN RETURN IS EMPTY
                summary_dfs, pop_rates, burst_stats = list(
                    zip(*pool.map(func_analysis, spikes_collected))
                )

        elif analysis_name == "MK1":
            # full features with spikes, bursts, PSDs, PCA
            func_analysis = partial(
                analysis_utils.compute_summary_features, params_dict=params_dict_default
            )

            with Pool(num_cores) as pool:
                summary_stats = list(pool.map(func_analysis, spikes_collected))
            # squish them into one big df row
            summary_dfs = [
                pd.concat(
                    [
                        s["summary_spikes"],
                        s["summary_bursts"],
                        s["summary_pca"],
                        s["summary_psd"].loc["exc_rate"].to_frame(name=0).T,
                    ],
                    axis=1,
                )
                for s in summary_stats
            ]

        # collect dfs and stream out
        # here it's assumed that the seeds and output df are aligned, but is not guaranteed
        # NOTE: it fails silently in a way such that, if the lengths of random_seeds_batch
        # and summary_dfs are not the same, it only collects up to the shorter length
        df_summaries = pd.concat(
            dict(zip(random_seeds_batch, summary_dfs))
        ).reset_index(level=1, drop=True)
        df_summaries.to_csv(stream_file_path, mode="a", header=(i_b == 0))

        print(f"[batch {i_b+1} of {n_batches}]: analysis time", time() - start_time)

        # plot
        if analysis_settings_update["params_analysis.do_plot"]:
            if analysis_name == "spikes_bursts":
                func_plot = partial(
                    plot_utils.plot_wrapper, params_dict=params_dict_default
                )
                with Pool(num_cores) as pool:
                    pool.starmap(
                        func_plot, list(zip(pop_rates, burst_stats, random_seeds_batch))
                    )

            elif analysis_name == "MK1":
                func_plot = partial(
                    plot_utils.plot_wrapper_MK1, params_dict=params_dict_default
                )
                with Pool(num_cores) as pool:
                    pool.starmap(
                        func_plot, list(zip(summary_stats, random_seeds_batch))
                    )

            print(f"[batch {i_b+1}]: plot time", time() - start_time)

        print("\n-----")

    # save out params dict
    data_utils.pickle_file(
        data_file_dict["root_path"]
        + f"{batch_seed}_{analysis_name}_params_dict_analysis_updated.pickle",
        params_dict_default,
    )

    # reload and merge
    df_summary_stream = pd.read_csv(stream_file_path, index_col=0)
    df_summary = data_utils.merge_theta_and_x(
        df_prior_samples_ran, df_summary_stream.index.values, df_summary_stream
    )
    df_summary.to_csv(save_file_path)

    # remove streaming file
    os.remove(stream_file_path)


def summary_to_df(summary_collector, params_analysis):
    df_summaries = []
    for i_s, s in enumerate(summary_collector):    
        row = []
        for summary_type in ['spikes', 'bursts', 'pca', 'psd']:
            if params_analysis['do_' + summary_type]:
                if summary_type == 'psd':
                    row.append(s[f'summary_{summary_type}'].loc['exc_rate'].to_frame(name=0).T)
                else:
                    row.append(s[f'summary_{summary_type}'])

        df_summaries.append(pd.concat(row, axis=1))
    return pd.concat(df_summaries)
