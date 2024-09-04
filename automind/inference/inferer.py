# Defining the various inference algorithms, for both posterior and likelihood estimation, such that it can be called in the training script.
# Includes inference instantiation, posterior building, and sampling functions.

import pandas as pd
import torch
from sbi.inference import SNPE, SNLE, SNRE, MCMCPosterior
from sbi.utils import mcmc_transform
from automind.inference.algorithms.Regression import RegressionEnsemble
from automind.inference.algorithms.GBI import (
    GBInference,
    GBIPotential,
    mse_dist,
)
from automind.utils import dist_utils


def get_posterior_from_nn(
    neural_net, prior, algorithm, build_posterior_params=None, use_unity_prior=True
):
    """Build approximate posterior from neural net and prior using specified algorithm.

    Args:
        neural_net (nn.Module): Flow or ACE network.
        prior (Distribution): Prior distribution.
        algorithm (str): Inference algorithm to use.
        build_posterior_params (dict, optional): Additional parameters for posterior building. Defaults to None.
        use_unity_prior (bool, optional): Whether to use a dummy Box[0,1] prior. Defaults to True.

    Returns:
        Distribution: Approximate posterior distribution.
    """
    # Need to build this dummy prior if algorithms were trained with min-max standardized theta
    from sbi.utils import BoxUniform

    prior_unity = BoxUniform(
        torch.zeros(
            len(
                prior.names,
            )
        ),
        torch.ones(
            len(
                prior.names,
            )
        ),
    )

    prior_use = prior_unity if use_unity_prior else prior

    if algorithm == "NPE":
        posterior = SNPE().build_posterior(neural_net, prior_use)

    elif algorithm == "NLE":
        posterior = SNLE().build_posterior(
            neural_net,
            prior_use,
            mcmc_method=build_posterior_params["mcmc_method"],
            mcmc_parameters=build_posterior_params["mcmc_parameters"],
        )

    elif algorithm == "NRE":
        posterior = SNRE().build_posterior(
            neural_net,
            prior_use,
            mcmc_method=build_posterior_params["mcmc_method"],
            mcmc_parameters=build_posterior_params["mcmc_parameters"],
        )

    elif (algorithm == "REGR-R") or (algorithm == "REGR-F"):
        posterior = RegressionEnsemble(neural_net)

    elif algorithm == "ACE":
        # prior log-prob is -55 for original, 0 for dummy
        inference = GBInference(prior_use, mse_dist)

        # Get generalized log-likelihood function
        genlike_func = inference.build_amortized_GLL(neural_net)
        potential_fn = GBIPotential(
            prior_use, genlike_func, beta=build_posterior_params["ace_beta"]
        )

        # Make posterior
        theta_transform = mcmc_transform(prior_use)
        posterior = MCMCPosterior(
            potential_fn,
            theta_transform=theta_transform,
            proposal=prior_use,
            method=build_posterior_params["mcmc_method"],
            **build_posterior_params["mcmc_parameters"],
        )
        # Save beta in object
        posterior.beta = build_posterior_params["ace_beta"]

    # Inherit stuff from prior
    posterior = dist_utils.pass_info_from_prior(
        prior,
        posterior,
        [
            "names",
            "marginals",
            "b2_units",
            "as_dict",
            "x_bounds_and_transforms",
            "x_standardizing_func",
        ],
    )
    if not hasattr(posterior, "prior"):
        posterior.prior = prior_use

    posterior.prior_original = prior
    return posterior


def append_samples(df_to_append_to, new_samples, sample_type, param_names):
    """Append samples to existing dataframe.

    Args:
        df_to_append_to (DataFrame): Existing dataframe.
        new_samples (DataFrame): New samples to append.
        sample_type (str): Type of samples, e.g., 'NPE', 'NPE_oversample'.
        param_names (str): Parameter names.

    Returns:
        DataFrame: Updated dataframe.
    """
    df_cur = pd.DataFrame(columns=df_to_append_to.columns)
    df_cur[param_names] = new_samples
    df_cur["inference.type"] = sample_type
    return pd.concat((df_to_append_to, df_cur), axis=0)


def sample_from_posterior(
    posterior, prior, num_samples, x_o, cfg_algorithm, theta_o=None, batch_run=None
):
    """Sample from posterior distribution.

    Args:
        posterior (Distribution): Posterior distribution.
        prior (Distribution): Prior distribution.
        num_samples (int): Number of samples to draw.
        x_o (tensor): Observed data to condition on.
        cfg_algorithm (dict): Sampling algorithm configurations.
        theta_o (tensor, optional): True parameter set. Defaults to None.
        batch_run (list, optional): Batch/random seed ID. Defaults to None.

    Returns:
        DataFrame: Posterior samples.
    """
    df_posterior_samples = pd.DataFrame(columns=["inference.type"] + posterior.names)
    samples_dict = {}
    # Append GT if there is
    if theta_o is not None:
        df_posterior_samples = append_samples(
            df_posterior_samples,
            theta_o[None, :].numpy().astype(float),
            "gt_resim",
            posterior.names,
        )
    if "Regression" in str(posterior):
        # Custom sampling for RegressionEnsemble and pass through
        posterior_samples = posterior.sample(x=x_o)
        sample_type_samples = f'{cfg_algorithm["name"]}_samples'
        theta, _ = dist_utils.standardize_theta(
            posterior_samples, prior, destandardize=True
        )
        samples_dict[sample_type_samples] = theta
        # Just return all ones for log_probs
        log_prob_fn = lambda theta, x: (
            torch.zeros((1,)) if len(theta.shape) == 1 else torch.zeros(theta.shape[0])
        )
    else:
        # Get log_prob / potential function for MCMC or DirectPosterior
        log_prob_fn = (
            posterior.potential if "MCMC" in str(posterior) else posterior.log_prob
        )

        # Posterior samples, with option to oversample high-prob
        if cfg_algorithm["oversample_factor"] > 1:
            samples_ = posterior.sample(
                x=x_o, sample_shape=(num_samples * cfg_algorithm["oversample_factor"],)
            )
            posterior_samples = samples_[
                torch.sort(log_prob_fn(samples_, x_o), descending=True)[1]
            ][:num_samples]
            sample_type_samples = f'{cfg_algorithm["name"]}_samples_prune_{cfg_algorithm["oversample_factor"]}'
        else:
            posterior_samples = posterior.sample(x=x_o, sample_shape=(num_samples,))
            sample_type_samples = f'{cfg_algorithm["name"]}_samples'

        theta, _ = dist_utils.standardize_theta(
            posterior_samples, prior, destandardize=True
        )
        samples_dict[sample_type_samples] = theta

        # MAP sample
        if cfg_algorithm["do_sample_map"]:
            posterior.set_default_x(x_o)
            posterior_map = posterior.map()
            theta_map = (
                dist_utils.standardize_theta(posterior_map, prior, destandardize=True)[
                    0
                ][None, :]
                .numpy()
                .astype(float)
            )
            sample_type = f'{cfg_algorithm["name"]}_map'
            df_posterior_samples = append_samples(
                df_posterior_samples,
                theta_map,
                sample_type,
                posterior.names,
            )
            samples_dict[sample_type] = theta_map

    # posterior sample mean
    if cfg_algorithm["do_sample_pmean"]:
        posterior_mean = posterior_samples.mean(0)
        theta_pmean = (
            dist_utils.standardize_theta(posterior_mean, prior, destandardize=True)[0][
                None, :
            ]
            .numpy()
            .astype(float)
        )
        sample_type = f'{cfg_algorithm["name"]}_mean'
        df_posterior_samples = append_samples(
            df_posterior_samples,
            theta_pmean,
            sample_type,
            posterior.names,
        )
        samples_dict[sample_type] = theta_pmean

    # Append posterior samples last
    df_posterior_samples = append_samples(
        df_posterior_samples,
        theta.numpy().astype(float),
        sample_type_samples,
        posterior.names,
    )

    # Append log_prob
    theta_scaled = dist_utils.standardize_theta(
        torch.Tensor(df_posterior_samples[posterior.names].values.astype(float)), prior
    )[0]
    df_posterior_samples.insert(1, "infer.log_prob", log_prob_fn(theta_scaled, x=x_o))

    # Append xo info
    if batch_run is not None:
        df_posterior_samples.insert(0, "x_o", f"{batch_run[0]}_{batch_run[1]}")

    return df_posterior_samples, samples_dict
