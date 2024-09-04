### Utility functions for distributions and inference
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor, stack
from scipy.stats import gaussian_kde
from sbi.utils import MultipleIndependent
from . import data_utils
from .analysis_utils import manual_filter_logPSD, discard_by_avg_power
from ..inference import trainers


class CustomIndependentJoint(MultipleIndependent):
    """Custom class for base indepedent-marginal prior distributions."""

    def __init__(self, variable_params):
        dist_list = [vp[1](**vp[2]).expand((1,)) for vp in variable_params]
        super().__init__(dist_list)
        """
        Set the list of 1D distributions and names.
        """
        self.names = [vp[0] for vp in variable_params]
        self.marginals = [vp[1](**vp[2]) for vp in variable_params]
        self.b2_units = [vp[3] for vp in variable_params]
        self.make_prior_dict()

    def __repr__(self) -> str:
        out = ""
        for n, u, m in zip(self.names, self.b2_units, self.marginals):
            out += f"{n} ({u}) ~ {m} \n"
        return out

    def make_prior_dict(self):
        self.as_dict = {
            name: {"unit": self.b2_units[i_n], "marginal": self.marginals[i_n]}
            for i_n, name in enumerate(self.names)
        }
        return self.as_dict


class MinMaxStandardizeTransform(nn.Module):
    def __init__(
        self,
        min,
        max,
        new_min,
        new_max,
    ):
        """Transforms input tensor to new min/max range."""
        super().__init__()
        min, max, new_min, new_max = map(torch.as_tensor, (min, max, new_min, new_max))
        self.min = min
        self.max = max
        self.new_min = new_min
        self.new_max = new_max
        self.register_buffer("_min", min)
        self.register_buffer("_max", max)
        self.register_buffer("_new_min", new_min)
        self.register_buffer("_new_max", new_max)

    def forward(self, tensor):
        return standardize_minmax(
            tensor, self._min, self._max, self._new_min, self._new_max
        )


def pass_info_from_prior(
    dist1, dist2, passed_attrs=["names", "marginals", "b2_units", "as_dict"]
):
    """Pass attributes from one distribution to another."""
    for attr in passed_attrs:
        if hasattr(dist1, attr):
            setattr(dist2, attr, getattr(dist1, attr))
        else:
            print(f"{attr} not in giving distribution.")
    return dist2


def find_matching_xo(df_xos, xo_query):
    """Find xo in dataframe given query.

    Args:
        df_xos (pandas dataframe): Summary dataframe of all xos.
        xo_query (tuple): querying info of xo, (batch_seed, random_seed) for sims, (date, well) for organoid.

    Returns:
        dataframe: queried xo in the dataframe.
    """
    batch, run = xo_query
    if ("date" in df_xos.columns) & ("well" in df_xos.columns):
        # xos from organoid data, get xos by date and well
        # print(f"recording: {batch}, well: {run}")
        df_matched = df_xos[(df_xos["date"] == batch) & (df_xos["well"] == run)]

    elif ("mouse" in df_xos.columns) & ("area" in df_xos.columns):
        # xos from mouse neuropixels data, get by mouse and region
        # print(f"mouse: {batch}, area: {run}")
        df_matched = df_xos[(df_xos["mouse"] == batch) & (df_xos["area"] == run)]

    elif ("params_settings.batch_seed" in df_xos.columns) & (
        "params_settings.random_seed" in df_xos.columns
    ):
        # xos from simulation, get by batch and random seed
        # print(f"batch seed: {batch}, random seed: {run}")
        df_matched = df_xos[
            (df_xos["params_settings.batch_seed"] == batch)
            & (df_xos["params_settings.random_seed"] == run)
        ]
    return df_matched


def find_posterior_preds_of_xo(
    df_sims, xo_query, query_string_form="%s_%s", discard_early_stopped=True
):
    """Find simulations whose xo is queried.

    Args:
        df_sims (pandas dataframe): Dataframe containing simulation data.
        xo_query (tuple): querying info of xo.
        query_string_form (str, optional): String format of xo column in df_sims. Defaults to "%s_%s".
        discard_early_stopped (bool, optional): Whether to discard sims that didn't run. Defaults to True.

    Returns:
        dataframe: all simulation samples that have matching xo.
    """
    if discard_early_stopped:
        return df_sims[
            (df_sims["x_o"] == (query_string_form % xo_query))
            & (df_sims["params_analysis.early_stopped"] == False)
        ]
    else:
        return df_sims[(df_sims["x_o"] == (query_string_form % xo_query))]


def sample_proposal(proposal, n_samples, xo=None):
    """Sample from proposal distribution. If xo is provided, set it as default xo."""
    if proposal.default_x is not None:
        print("Proposal has default xo.")
        if xo is not None:
            print(
                f"Default xo is identical to current xo: {(proposal.default_x==xo).all()}"
            )
    else:
        print("Setting default xo")
        proposal.set_default_x(xo)

    samples = proposal.sample((n_samples,))
    df_samples = pd.DataFrame(samples.numpy().astype(float), columns=proposal.names)
    return df_samples


def sample_different_xos_from_posterior(
    density_estimator, df_xos, xo_queries, n_samples_per, return_xos=False
):
    """Draw samples conditioned on a list of observations (xos).

    Args:
        density_estimator (SBI posterior): posterior density estimator from SBI.
        df_xos (pandas dataframe): dataframe of all observations.
        xo_queries (list): list of tuples that uniquely identify xos to be matched in df_xos.
        n_samples_per (int): number of samples to be drawn per observation.
        return_xos (bool, optional): Whether to also return info on matched xos. Defaults to False.

    Returns:
        _type_: df_posterior_samples, list of [df_matched, batch, run, xo]
    """
    posterior_samples, xos = [], []
    # transform xos
    try:
        print("Applying custom feature transformations.")
        [print(k, v) for k, v in density_estimator.x_bounds_and_transforms.items()]
        if "freq_bounds" in density_estimator.x_bounds_and_transforms.keys():
            # Using PSD features
            df_xos[density_estimator.x_feat_set] = preproc_dataframe_psd(
                df_xos, density_estimator.x_bounds_and_transforms, drop_nans=False
            )[0]
        else:
            # Using spike/burst features
            df_xos[density_estimator.x_feat_set] = preproc_dataframe(
                df_xos, density_estimator.x_bounds_and_transforms, drop_nans=False
            )[0]
    except Exception as e:
        print(e)
        print(
            "Transformations failed (likely no transformation included in posterior)."
        )

    for batch, run in xo_queries:
        if ("date" in df_xos.columns) & ("well" in df_xos.columns):
            # xos from organoid data, get xos by date and well
            print(f"recording: {batch}, well: {run}")
            df_matched = df_xos[(df_xos["date"] == batch) & (df_xos["well"] == run)]
        elif ("mouse" in df_xos.columns) & ("area" in df_xos.columns):
            # xos from mouse neuropixels data, get by mouse and region
            print(f"mouse: {batch}, area: {run}")
            df_matched = df_xos[(df_xos["mouse"] == batch) & (df_xos["area"] == run)]
        elif ("params_settings.batch_seed" in df_xos.columns) & (
            "params_settings.random_seed" in df_xos.columns
        ):
            # xos from simulation, get by batch and random seed
            print(f"batch seed: {batch}, random seed: {run}")
            df_matched = df_xos[
                (df_xos["params_settings.batch_seed"] == batch)
                & (df_xos["params_settings.random_seed"] == run)
            ]

        # get xo in tensor
        xo = Tensor(df_matched[density_estimator.x_feat_set].astype(float).values)
        # sample posterior
        if density_estimator.default_x is not None:
            print("Density estimator has default x.")
            print(
                f"Default x is identical to current xo: {(density_estimator.default_x==xo).all()}"
            )

        samples = density_estimator.sample((n_samples_per,), x=xo)

        # store samples
        df_samples = pd.DataFrame(
            samples.numpy().astype(float), columns=density_estimator.names
        )
        df_samples.insert(0, "x_o", f"{batch}_{run}")
        posterior_samples.append(df_samples)
        xos.append((xo, df_matched, batch, run))

    df_posterior_samples = pd.concat(posterior_samples, ignore_index=True)
    if return_xos:
        return df_posterior_samples, xos
    else:
        return df_posterior_samples


def fill_gaps_with_nans(theta_full, x_partial, idx_good):
    """Fill in missing values in x_partial with nans. For ACE-GBI."""
    x_full = torch.zeros((theta_full.shape[0], x_partial.shape[1]))
    x_full[idx_good] = x_partial
    x_full[~idx_good] = torch.nan
    return Tensor(theta_full), x_full


def proc_one_column(df_col, bound_feat, log_feat):
    """Process one column of dataframe given bounds and log flag."""
    if bound_feat:
        df_col[(df_col < bound_feat[0]) | (df_col > bound_feat[1])] = np.nan
    if log_feat:
        df_col = np.log10(df_col)
    col_range = [df_col.min(), df_col.max()]
    return df_col, col_range


def preproc_dataframe(
    df_summary, x_transforms_dict, drop_nans=False, replace_infs=False
):
    """Preprocess dataframe for inference and pairplot.

    Discard entries outside of bound, and log features if indicated.
    Args:
        df_summary (pd dataframe): dataframe to preprocess.
        transforms_dict (dict): {name: (bounds, log_or_not)}

    Returns:
        df_copy, feat_bounds, feat_names_pretty :
    """
    columns = list(x_transforms_dict.keys())

    # make a copy of dataframe
    df_copy = df_summary[columns].copy()
    feat_bounds = []
    for i_c, col in enumerate(columns):
        # call the processing function
        df_col, col_range = proc_one_column(df_copy[col], *x_transforms_dict[col])
        df_copy[col] = df_col
        feat_bounds.append(col_range)

    if replace_infs:
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)

    if drop_nans:
        df_copy = df_copy[~df_copy.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # Get binary indices of good samples
    idx_good = np.zeros(len(df_summary), dtype=bool)
    # Get indices that have no nans or inf
    idx_good[df_copy[~df_copy.isin([np.nan, np.inf, -np.inf]).any(axis=1)].index] = True

    # Get pretty feature names
    feat_names_pretty = [
        (
            f"$log_{{10}}$ {' '.join(col.split('_'))}"
            if x_transforms_dict[col][1]
            else " ".join(col.split("_"))
        )
        for col in columns
    ]
    return df_copy, feat_bounds, idx_good, feat_names_pretty


def preproc_dataframe_psd(
    df_summary, f_transforms_dict, drop_nans=False, replace_infs=False
):
    """Preprocess PSD dataframe for inference."""

    # Get frequency axis and the indices of the frequencies we want to keep
    cols_psd = data_utils.subselect_features(df_summary, ["psd"])
    f_axis = data_utils.decode_df_float_axis(cols_psd)
    f_sel_idx = (f_axis >= f_transforms_dict["freq_bounds"][0]) & (
        f_axis <= f_transforms_dict["freq_bounds"][1]
    )
    idx_good = np.ones(len(df_summary), dtype=bool)

    # Discard bad samples if required and grab only the frequencies we want
    if f_transforms_dict["discard_stopped"]:
        if "params_analysis.early_stopped" in df_summary.columns:
            # Discard samples that was early stopped
            idx_keep = df_summary["params_analysis.early_stopped"] == False
            print(
                f"{idx_good.sum()-(idx_good & idx_keep).sum()} sims dropped due to early stopping."
            )
            idx_good = idx_good & idx_keep

    if f_transforms_dict["discard_conditions"]:
        if f_transforms_dict["discard_conditions"]["manual_filter_logPSD"]:
            log_psd = np.log10(df_summary[cols_psd].values)
            f_axis = data_utils.decode_df_float_axis(cols_psd)
            idx_keep = manual_filter_logPSD(log_psd, f_axis)
            print(
                f"{idx_good.sum()-(idx_good & idx_keep).sum()} sims dropped due to manual criteria."
            )
            idx_good = idx_good & idx_keep

        if f_transforms_dict["discard_conditions"]["power_thresh"]:
            idx_keep = discard_by_avg_power(
                df_summary,
                f_band=f_transforms_dict["discard_conditions"]["f_band"],
                power_thresh=f_transforms_dict["discard_conditions"]["power_thresh"],
                return_idx=True,
            )
            print(
                f"{idx_good.sum()-(idx_good & idx_keep).sum()} sims dropped due to power threshold discard."
            )
            idx_good = idx_good & idx_keep

    df_copy = df_summary[cols_psd][idx_good].iloc[:, f_sel_idx].copy()
    print(f"{len(df_summary) - len(df_copy)} samples discarded in total.")

    # Log transform if required
    if f_transforms_dict["log_power"]:
        df_copy = np.log10(df_copy)

    # Replace infs and nans
    if replace_infs:
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan)
    if drop_nans:
        df_copy = df_copy.dropna()

    return df_copy, f_axis[f_sel_idx], idx_good, f_sel_idx


def logPSD_scaler(log_psd, demean=True, scaler=0.1):
    """Scaling function for log-PSDs prior to training.

    Args:
        log_psd (tensor): log-PSD tensor.
        demean (bool, optional): Whether to demean the log PSD. Defaults to True.
        scaler (float, optional): Scaling factor. Defaults to 0.1.

    Returns:
        tensor: scaled log-PSD.
    """

    if demean:
        log_psd = log_psd - log_psd.mean(1, keepdim=True)
    return log_psd * scaler


def retrieve_trained_network(
    inf_dir,
    algorithm,
    feature_set,
    inference_datetime=None,
    job_id=None,
    load_network_prior=True,
):
    """Retrieve trained network and prior from inference directory."""
    from os import listdir, path

    full_path = f"{inf_dir}/{feature_set}/{algorithm}/"
    if inference_datetime is None:
        # Get latest inference run
        inference_datetime = np.sort(listdir(full_path))[-1]
        print(listdir(full_path))

    full_path += inference_datetime

    if job_id is None:
        # Get last job id
        job_id = np.sort(
            [f for f in listdir(full_path) if path.isdir(full_path + f"/{f}")]
        )[-1]

    full_path += f"/{job_id}/"
    # Check that the path exists

    if load_network_prior:
        if not path.isdir(full_path):
            raise ValueError(f"{full_path}: path does not exist.")
        else:
            print(f"Loading network and prior from {full_path}")
            neural_net = data_utils.load_pickled(full_path + "/neural_net.pickle")
            prior = data_utils.load_pickled(full_path + "/prior.pickle")
            return full_path, neural_net, prior
    else:
        if not path.isdir(full_path):
            print(f"Warning: {full_path} path does not exist but returned anyway.")
        return full_path


def df_to_tensor(df_theta, df_x):
    """Convert dataframes to tensors."""
    thetas = Tensor(df_theta.values)
    xs = Tensor(df_x.values)
    return thetas, xs


def log_n_stdz(samples, do_log=True, standardizing_func=None):
    """Log and standardize samples."""
    if do_log:
        samples = np.log10(samples)
    if standardizing_func:
        samples = standardizing_func(torch.Tensor(samples)).numpy()
    return samples


def train_algorithm(theta, x, prior, cfg):
    """Train algorithm based on config."""
    print(f"Training {cfg.name}: {cfg}...")
    if cfg.name == "NPE":
        inference, neural_net = trainers.train_NPE(theta, x, prior, cfg)
    elif cfg.name == "RestrictorNPE":
        inference, neural_net = trainers.train_RestrictorNPE(theta, x, prior, cfg)
    elif cfg.name == "NLE":
        inference, neural_net = trainers.train_NLE(theta, x, prior, cfg)
    elif cfg.name == "NRE":
        inference, neural_net = trainers.train_NRE(theta, x, prior, cfg)
    elif cfg.name == "ACE":
        inference, neural_net = trainers.train_ACE(theta, x, prior, cfg)
    elif (cfg.name == "REGR-F") or (cfg.name == "REGR-R"):
        # Forward or reverse is taken care of in the trainer
        inference, neural_net = trainers.train_Regression(theta, x, prior, cfg)
    else:
        raise NotImplementedError("Algorithm not recognised.")

    print("Training finished.")
    return inference, neural_net


def sort_closest_to_xo(xo, samples, distance_func="mse", top_n=None, weights=None):
    """Sort samples by distance to xo."""
    if weights is None:
        weights = np.ones(xo.shape[1])
    if distance_func in ["mse", "mae"]:
        dists = (samples - xo) ** 2 if distance_func == "mse" else np.abs(samples - xo)
        dist = (dists * weights).mean(1)
    else:
        raise NotImplementedError(f"Distance function {distance_func} not implemented.")

    # Sort by distance
    idx_dist_sorted = np.argsort(dist)
    if top_n is None:
        top_n = len(dist)
    return (
        samples[idx_dist_sorted][:top_n],
        dist[idx_dist_sorted][:top_n],
        idx_dist_sorted,
    )


def standardize_theta(theta, prior, low_high=Tensor([0.0, 1.0]), destandardize=False):
    """Standardize theta to new min-max range based on prior bounds, or convert back."""
    theta_minmax = get_minmax(prior)
    if destandardize:
        # Convert from standardized back to original range
        theta_standardized = standardize_minmax(
            theta, low_high[0], low_high[1], theta_minmax[:, 0], theta_minmax[:, 1]
        )
    else:
        # Standardize to [low, high] from prior bounds
        theta_standardized = standardize_minmax(
            theta, theta_minmax[:, 0], theta_minmax[:, 1], low_high[0], low_high[1]
        )

    return theta_standardized, theta_minmax


def get_minmax(prior):
    """Get minmax range of prior."""
    minmax = stack(
        [
            Tensor([prior.marginals[i].low, prior.marginals[i].high])
            for i in range(len(prior.names))
        ],
        0,
    )
    return minmax


def standardize_minmax(theta, min, max, new_min, new_max):
    """Standardize to between [low,high] based on prior bounds."""
    theta_transformed = ((theta - min) / (max - min)) * (new_max - new_min) + new_min
    return theta_transformed


def kde_estimate(data, bounds, points=1000):
    """Get kernel density estimate of samples."""
    kde = gaussian_kde(data)
    grid = np.linspace(bounds[0], bounds[1], points)
    density = kde(grid)
    return grid, density
