### Organoid data analysis utilities

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..analysis import spikes_summary
from .data_utils import (
    convert_spike_array_to_dict,
    update_params_dict,
    make_subfolders,
    pickle_file,
)
from .analysis_utils import compute_summary_features
from .plot_utils import plot_rates_and_bursts
from ..sim.runners import construct_experiment_settings_adex


def convert_date_to_int(df_organoid):
    """
    Convert dates in the dataframe to integer days since the first date.

    Args:
        df_organoid (pd.DataFrame): DataFrame containing a 'date' column with dates in '%y%m%d' format.

    Returns:
        np.ndarray: Array of integers representing days since the first date.
    """
    day1 = datetime.strptime(df_organoid["date"][0], "%y%m%d")
    return np.array(
        [
            (datetime.strptime(d.split("_")[0], "%y%m%d") - day1).days
            for d in df_organoid["date"]
        ]
    )


def get_organoid_raw_data(org_spikes_path, query=None, bin_spike_params=None):
    """Quick access helper for loading organoid spiking data.

    Args:
        org_spikes_path (str): path string to organoid data file.
        query (dict, optional): {'date': YYMMDD, 'well': int}. Defaults to None.
        bin_spike_params (dict, optional): configurations for spike binning. Defaults to None.

    Returns:
        dict: dictionary with spikes and rate from queried date, or all spikes.
    """
    fs = 12500
    # load data
    data_all = np.load(org_spikes_path, allow_pickle=True)
    org_spikes_all, org_t_all, org_name_all = (
        data_all["spikes"],
        data_all["t"],
        data_all["recs"],
    )
    org_name_all_matched = [n.split(".")[0][7:] for n in org_name_all]

    data_out = {}
    if query is not None:
        # find queried datapoint
        idx_date, idx_well = org_name_all_matched.index(query["date"]), query["well"]
        spikes_query = org_spikes_all[idx_date][idx_well]
        data_out["spikes"] = spikes_query

        # population firing rate
        t_s = org_t_all[idx_date]
        spikes_well = (
            np.array([np.array(s, dtype=float) for s in spikes_query], dtype=object)
            / fs
        )
        data_out["t"], data_out["pop_rate"] = spikes_summary.bin_population_spikes(
            spikes_well, t_s[0], t_s[-1], **bin_spike_params
        )
        return data_out

    else:
        # no specific query, return all spikes
        data_out = {
            "spikes_all": org_spikes_all,
            "t_all": org_t_all,
            "name_all_matched": org_name_all_matched,
        }
        return data_out


def compute_summary_organoid_well(organoid_raw_data, day, well, params_dict_default):
    """
    Compute summary statistics for a specific well in the organoid data.

    Args:
        organoid_raw_data (dict): Raw organoid data containing spikes and time information.
        day (int): Day index to extract data from.
        well (int): Well index to extract data from.
        params_dict_default (dict): Default parameters for computing summary statistics.

    Returns:
        tuple: (dataframe, dict, dict) Summary statistics for the specified well.
    """
    # reformat into spike dictionary like simulations
    spikes = organoid_raw_data["spikes_all"][day][well]
    t_s = organoid_raw_data["t_all"][day]
    spikes_dict = convert_spike_array_to_dict(spikes, fs=12500)
    spikes_dict["exc_spikes"] = spikes_dict["exc"]
    spikes_dict["t_end"] = t_s[-1]
    summary_stats = compute_summary_features(spikes_dict, params_dict_default)

    df_features = pd.concat(
        [summary_stats["summary_spikes"], summary_stats["summary_bursts"]], axis=1
    )

    return df_features, summary_stats["pop_rates"], summary_stats["summary_burst_stats"]


def run_organoid_analysis(
    organoid_data_folder, output_folder, exp_settings_update, analysis_settings_update
):
    """
    Run analysis on organoid data.
    """
    params_dict_default = construct_experiment_settings_adex(exp_settings_update)
    params_dict_default = update_params_dict(
        params_dict_default, analysis_settings_update
    )

    # turn off psd and pca analyses
    params_dict_default["params_analysis"]["do_pca"] = False
    params_dict_default["params_analysis"]["do_psd"] = False

    [print(f"{k}: {v}") for k, v in params_dict_default["params_analysis"].items()]
    organoid_raw_data = get_organoid_raw_data(
        organoid_data_folder + "organoid_spikes.npz"
    )
    num_days = len(organoid_raw_data["name_all_matched"])
    df_aggregate_summary = pd.DataFrame([])
    do_plot = params_dict_default["params_analysis"]["do_plot"]

    for day in range(num_days):
        print(f"{day+1} / {num_days} days...")
        if do_plot:
            fig, axs = plt.subplots(
                8, 2, gridspec_kw={"width_ratios": [5, 1]}, figsize=(20, 24)
            )

        features = []
        for well in range(8):
            df_features, rates, burst_stats = compute_summary_organoid_well(
                organoid_raw_data, day, well, params_dict_default
            )
            features.append(df_features)

            if do_plot:
                plot_rates_and_bursts(
                    rates,
                    burst_stats,
                    vars_to_plot={"exc_rate": "k"},
                    burst_alpha=0.2,
                    fig_handles=(fig, axs[well, :]),
                    burst_time_offset=params_dict_default["params_analysis"][
                        "burst_win"
                    ][0],
                )
                axs[well, 0].set_xlim([0, 120])
                axs[well, 1].set_xlim(
                    params_dict_default["params_analysis"]["burst_win"]
                )
        df_info = pd.DataFrame(
            [
                [organoid_raw_data["name_all_matched"][day], well, rates["t_ds"][-1]]
                for well in range(8)
            ],
            columns=["date", "well", "t_rec"],
        )
        df_features = df_info.join(pd.concat(features).reset_index(drop=True))
        df_aggregate_summary = pd.concat(
            (df_aggregate_summary, df_features), ignore_index=True
        )
        path_dict = make_subfolders(output_folder, ["figures"])
        if do_plot:
            plt.tight_layout()
            plt.savefig(
                path_dict["figures"]
                + organoid_raw_data["name_all_matched"][day]
                + ".pdf"
            )
            plt.close()

    # save out params file
    pickle_file(
        output_folder + "/organoid_params_dict_default.pickle", params_dict_default
    )
    df_aggregate_summary.to_csv(output_folder + "/organoid_summary.csv")
    return
