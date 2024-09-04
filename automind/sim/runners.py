### This file contains the main simulation runner scripts and helpers.
import numpy as np
import brian2 as b2
import pandas as pd
from time import time
import sys

from . import b2_models, b2_interface
from ..analysis import spikes_summary
from ..utils import data_utils, dist_utils
from ..sim import b2_interface, default_configs


def adex_check_early_stop(net_collect, params_dict, verbose=False):
    """
    Check if simulation should be stopped based on firing rate.
    """
    params_analysis = params_dict["params_analysis"]
    params_settings = params_dict["params_settings"]
    t_refrac = np.array(
        params_dict["params_Epop"]["t_refrac"]
    )  # get refractory period for norming FR

    # compute FR stats
    query_spikes = {
        "exc_spikes": b2_interface._deunitize_spiketimes(
            net_collect["exc_spikes"].spike_trains()
        )
    }

    df_summary = spikes_summary.return_df_summary(
        query_spikes,
        params_analysis["analysis_window"],
        params_analysis["min_num_spikes"],
    )

    fr_norm = (
        t_refrac
        * df_summary["isi_numspks"].mean(skipna=True)
        / np.diff(params_analysis["early_stop_window"])
    )

    # compare
    if (fr_norm <= params_analysis["stop_fr_norm"][0]) or (
        fr_norm >= params_analysis["stop_fr_norm"][1]
    ):
        if verbose:
            print("early stop: normed FR = %.4f%%" % (fr_norm * 100))
        params_analysis["early_stopped"] = True
        params_settings["sim_time"] = params_analysis["t_early_stop"]
    else:
        if verbose:
            print("continuing...")

    return params_dict


def run_net_early_stop(net_collect, params_dict):
    """
    Run simulation while checking for early stoppage.
    """
    # ---- early stoppage ----
    start_time = time()
    if params_dict["params_analysis"]["t_early_stop"]:
        # run for short time and check for min/max firing
        net_collect.run(params_dict["params_analysis"]["t_early_stop"])
        params_dict = adex_check_early_stop(net_collect, params_dict, verbose=False)
        # did not stop, continue with sim the rest of the way
        if not params_dict["params_analysis"]["early_stopped"]:
            net_collect.run(
                params_dict["params_settings"]["sim_time"]
                - params_dict["params_analysis"]["t_early_stop"]
            )
    else:
        # no early stopping just go
        net_collect.run(params_dict["params_settings"]["sim_time"])

    # update params
    params_dict["params_settings"]["real_run_time"] = time() - start_time
    return params_dict, net_collect


def adex_simulator(params_dict):
    """
    Simulator wrapper for AdEx net, run simulations and collect data.
    """

    print(
        f"{params_dict['params_settings']['batch_seed']}-{params_dict['params_settings']['random_seed']}",
        end="|",
    )

    try:
        # set up and run model with early stopping
        network_type = params_dict["params_settings"]["network_type"]
        if (not network_type) or (network_type == "adex"):
            net_collect = b2_models.adaptive_exp_net(params_dict)
        elif network_type == "adex_clustered":
            net_collect = b2_models.adaptive_exp_net_clustered(params_dict)

        # run the model
        params_dict, net_collect = run_net_early_stop(net_collect, params_dict)

        # return pickleable outputs for pool
        spikes, timeseries = data_utils.collect_raw_data(net_collect, params_dict)

        return params_dict, spikes, timeseries

    except Exception as e:
        print("-----")
        print(e)
        print(
            f"{params_dict['params_settings']['batch_seed']}-{params_dict['params_settings']['random_seed']}: FAILED"
        )
        print("-----")
        return params_dict, {}, {}


########
## experiment setup stuff
########
def construct_experiment_settings_adex(update_dict=None):
    """Construct and fill experiment settings for AdEx model."""
    #### grab default configs and fill in
    params_Epop = default_configs.ADEX_NEURON_DEFAULTS_ZERLAUT.copy()
    params_Ipop = default_configs.ADEX_NEURON_DEFAULTS_ZERLAUT.copy()
    params_net = default_configs.ADEX_NET_DEFAULTS.copy()
    params_analysis = default_configs.ANALYSIS_DEFAULTS.copy()
    params_settings = default_configs.SIM_SETTINGS_DEFAULTS.copy()

    # network configurations & set up simulation time and clock
    params_settings["t_sigs"] = int(
        np.ceil(-np.log10(params_settings["dt"] / b2.second))
    )

    # collect
    params_dict = {
        "params_net": params_net,
        "params_Epop": params_Epop,
        "params_Ipop": params_Ipop,
        "params_analysis": params_analysis,
        "params_settings": params_settings,
    }

    # update non-default values
    if update_dict:
        params_dict = data_utils.update_params_dict(params_dict, update_dict)

    # set some final configurations
    b2_interface.set_adaptive_vcut(params_dict["params_Epop"])
    b2_interface.set_adaptive_vcut(params_dict["params_Ipop"])

    return params_dict


def set_up_for_presampled(
    cfg, hydra_path, exp_config, construct_experiment_settings_fn
):
    """Set up for experiments where parameter samples are pre-generated and saved in csv.
    This is the new workflow where prior or posterior samples are generated in advance and saved in csv files.
    """
    # make path dict
    path_dict = data_utils.make_subfolders(hydra_path)

    # Extract relevant file names
    file_type_list = ["prior.pickle", "_samples.csv"]
    data_file_dict = data_utils.extract_data_files(path_dict["data"], file_type_list)

    # Load samples
    df_prior_samples = pd.read_csv(
        path_dict["data"] + data_file_dict["_samples"], index_col=0
    )
    batch_seed = int(df_prior_samples.iloc[0]["params_settings.batch_seed"])

    # Check if there is already a ran file, if so, we continue from the end
    samples_ran_file_dict = data_utils.extract_data_files(
        path_dict["data"], ["samples_ran.csv"]
    )
    if samples_ran_file_dict["samples_ran"]:
        # Pick up from where we left off, load the stuff
        print(
            f"Found {samples_ran_file_dict['samples_ran']}, continuing from where we left off..."
        )

        # Extract params_dict_default
        params_file_dict = data_utils.extract_data_files(
            path_dict["data"], ["params_dict_default.pickle"]
        )

        # Load params_dict_default
        params_dict_default = data_utils.load_pickled(
            path_dict["data"] + params_file_dict["params_dict_default"]
        )

    else:
        # Set up params_dict from scratch
        exp_settings_update = exp_config.exp_settings_update
        if cfg["infrastructure"]["flag_test"]:
            exp_settings_update["params_settings.sim_time"] = 20 * b2.second

        if cfg["experiment"]["sim_time"]:
            print(
                f'Simulation duration updated: {cfg["experiment"]["sim_time"]} seconds.'
            )
            exp_settings_update["params_settings.sim_time"] = (
                cfg["experiment"]["sim_time"] * b2.second
            )

        # construct params and priors
        params_dict_default = construct_experiment_settings_fn(exp_settings_update)
        params_dict_default["params_settings"]["experiment"] = cfg["experiment"][
            "network_name"
        ]
        _, params_dict_default = data_utils.set_seed_by_time(params_dict_default)

        # Put path dict in params_dict
        params_dict_default["path_dict"] = path_dict

        # Set batch seed
        params_dict_default["params_settings"]["batch_seed"] = batch_seed

        # Save params_dict_default
        data_utils.pickle_file(
            params_dict_default["path_dict"]["data"]
            + f"{batch_seed}_params_dict_default.pickle",
            params_dict_default,
        )

    # Chop off the ones that have already been ran
    if samples_ran_file_dict["samples_ran"]:
        df_prior_samples_ran = pd.read_csv(
            path_dict["data"] + samples_ran_file_dict["samples_ran"], index_col=0
        )
        # Check that the samples that ran are consistent
        if df_prior_samples.iloc[: len(df_prior_samples_ran)][
            df_prior_samples.columns[:4]
        ].equals(df_prior_samples_ran[df_prior_samples.columns[:4]]):
            print(
                "Samples ran are consistent with the samples in the csv file. Continuing from where we left off..."
            )
            print(f"Samples ran: {len(df_prior_samples_ran)}")
            print(f"Samples total: {len(df_prior_samples)}")
            df_prior_samples = df_prior_samples.iloc[len(df_prior_samples_ran) :]
            print(
                f"Samples left to run: {len(df_prior_samples)}, from {df_prior_samples.index[0]} to {df_prior_samples.index[-1]}"
            )
        else:
            print(
                "Samples ran are inconsistent with the samples in the csv file. Exiting..."
            )
            sys.exit()

    prior = data_utils.load_pickled(path_dict["data"] + data_file_dict["prior"])
    n_samples = len(df_prior_samples)

    # Set seeds
    data_utils.set_all_seeds(batch_seed)

    # Plug samples into list of param dictionaries for simulation
    params_dict_list = data_utils.fill_params_dict(
        params_dict_default, df_prior_samples, prior.as_dict, n_samples
    )
    return prior, df_prior_samples, params_dict_default, params_dict_list


def set_up_from_hydra(cfg, hydra_path, exp_config, construct_experiment_settings_fn):
    """Set up for experiments where parameter samples are generated on the fly."""
    exp_settings_update = exp_config.exp_settings_update
    if cfg["infrastructure"]["flag_test"]:
        exp_settings_update["params_settings.sim_time"] = 20 * b2.second

    if cfg["experiment"]["sim_time"]:
        print(f'Simulation duration updated: {cfg["experiment"]["sim_time"]} seconds.')
        exp_settings_update["params_settings.sim_time"] = (
            cfg["experiment"]["sim_time"] * b2.second
        )

    # construct params and priors
    params_dict_default = construct_experiment_settings_fn(exp_settings_update)
    params_dict_default["params_settings"]["experiment"] = cfg["experiment"]["name"]

    # option to manually set seed
    if cfg["experiment"]["batch_seed"]:
        batch_seed = cfg["experiment"]["batch_seed"]
        params_dict_default["params_settings"]["batch_seed"] = batch_seed
    else:
        batch_seed, params_dict_default = data_utils.set_seed_by_time(
            params_dict_default
        )

    # make path dict
    path_dict = data_utils.make_subfolders(hydra_path)
    params_dict_default["path_dict"] = path_dict

    proposal_path = cfg["experiment"]["proposal_path"]
    # set seeds
    data_utils.set_all_seeds(batch_seed)

    ####---------------
    if "multiround" in cfg["experiment"]["name"]:
        if proposal_path == "none":
            print("Multi-round requires density estimator / proposal. Exiting...")
            sys.exit()

        # load proposal from previously pickled file
        print(proposal_path)
        prior = data_utils.load_pickled(proposal_path)

        # load xo queries and data
        n_samples = int(cfg["experiment"]["n_samples"])
        df_prior_samples = dist_utils.sample_proposal(prior, n_samples)
        batch, run = (cfg["experiment"]["xo_batch"], cfg["experiment"]["xo_run"])
        round_number = int(proposal_path.split("posterior_R")[1][0])
        df_prior_samples.insert(0, "x_o", f"{batch}_{run}")
        df_prior_samples.insert(2, "round", round_number)

    # specific setup depending on round1 or round2
    elif "round2" in cfg["experiment"]["name"]:
        # round 2, get density estimator and draw from posterior
        if proposal_path == "none":
            print("Round 2 requires density estimator / proposal. Exiting...")
            sys.exit()

        # load proposal from previously pickled file
        # technically, this is the unconditioned density estimator, but since
        # in round2 it's likely that we're running multiple observations in parallel
        # just save the density estimator as the "proposal" downstream
        print(proposal_path)
        prior = data_utils.load_pickled(proposal_path)

        # load xo queries and data
        n_samples_per = int(cfg["experiment"]["n_samples_per"])

        xo_queries_database = exp_config.xo_queries_database
        xo_queries = xo_queries_database[cfg["experiment"]["xo_type"]]
        n_samples = len(xo_queries) * n_samples_per
        df_xos = pd.read_csv(cfg["experiment"]["xo_path"], index_col=0)

        # fill all the burst nans when there is valid isi stats
        # df_xos = data_utils.fill_nans_in_xos(df_xos, "isi_", 0)

        # condition and sample from posterior
        df_prior_samples = dist_utils.sample_different_xos_from_posterior(
            prior, df_xos, xo_queries, n_samples_per
        )

    else:
        # assume its round1, restricted, or default
        # make or load prior
        if proposal_path == "none":
            # no proposal distribution object, make new one
            variable_params = exp_config.variable_params
            prior = dist_utils.CustomIndependentJoint(variable_params)
        else:
            # load proposal from previously pickled file
            print(proposal_path)
            prior = data_utils.load_pickled(proposal_path)

        # draw from prior
        # get sample size
        n_samples = cfg["infrastructure"]["n_samples"]

        # sample prior
        samples = prior.sample((n_samples,)).numpy().astype(float)
        df_prior_samples = pd.DataFrame(samples, columns=prior.names)
    #
    ####-------------------

    # sort out seed types and make into dataframe
    df_prior_samples.insert(
        loc=0, column="params_settings.batch_seed", value=batch_seed
    )

    random_seeds = np.random.choice(
        a=int(n_samples * 100), size=n_samples, replace=False
    )
    if "round2" in cfg["experiment"]["name"]:
        # sort seeds so it doesn't scramble the round 2 simulations
        random_seeds = np.sort(random_seeds)

    df_prior_samples.insert(
        loc=1,
        column="params_settings.random_seed",
        value=random_seeds,
    )
    df_prior_samples = df_prior_samples.sort_values(
        "params_settings.random_seed", ignore_index=True
    )

    # save prior, samples, and default configs
    df_prior_samples.to_csv(
        params_dict_default["path_dict"]["data"] + f"{batch_seed}_prior_samples.csv"
    )
    data_utils.save_params_priors(
        params_dict_default["path_dict"]["data"] + f"{batch_seed}_",
        params_dict_default,
        prior,
    )

    # plug samples into list of param dictionaries for simulation
    params_dict_list = data_utils.fill_params_dict(
        params_dict_default, df_prior_samples, prior.as_dict, n_samples
    )

    return prior, df_prior_samples, params_dict_default, params_dict_list
