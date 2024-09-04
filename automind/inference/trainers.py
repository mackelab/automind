# Defining the various inference algorithms, for both posterior and likelihood estimation, such that it can be called in the training script.
# Includes inference instantiation, posterior building, and sampling functions.

import torch
from torch import nn
from sbi.inference import SNPE, SNLE, SNRE
from sbi.utils import get_nn_models, RestrictionEstimator, process_prior
from sbi.neural_nets.embedding_nets import FCEmbedding

from automind.inference.algorithms.Regression import RegressionInference
from automind.inference.algorithms.GBI import GBInference, get_distance_function
from automind.utils import data_utils


## NPE
def train_NPE(theta, x, prior, cfg):
    # some settings:
    # cfg.density_estimator: the type of density estimator to use, MAF, NSF, MoG, etc.
    # cfg.sigmoid_theta: apply sigmoid on theta to keep into prior range.
    # cfg.z_score_x: 'none', 'independent' or 'structured'.
    # cfg.z_score_theta: 'none', 'independent' or 'structured'.
    # cfg.use_embedding_net: whether to use embedding net or not, and its dimensions.
    embedding_net_x = _get_embedding_net(cfg, input_dim=x.shape[1])
    net = get_nn_models.posterior_nn(
        model=cfg.density_estimator,
        sigmoid_theta=cfg.sigmoid_theta,
        prior=prior,
        z_score_x=cfg.z_score_x,
        z_score_theta=cfg.z_score_theta,
        embedding_net=embedding_net_x,
        hidden_features=cfg.hidden_features,
        num_transforms=cfg.num_transforms,
        num_bins=cfg.num_bins,
    )
    inference = SNPE(prior=prior, density_estimator=net)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator


def train_Restrictor(theta, x, prior, cfg):
    # Train restriction estimator classifier first
    restriction_estimator = RestrictionEstimator(
        prior=prior,
        hidden_features=cfg.hidden_features_restrictor,
        num_blocks=cfg.num_blocks_restrictor,
        z_score=cfg.z_score_theta,
    )

    restriction_estimator.append_simulations(theta, x)
    classifier = restriction_estimator.train()
    proposal = restriction_estimator.restrict_prior()
    return proposal, classifier


def train_RestrictorNPE(theta, x, prior, cfg):
    # Train restriction estimator classifier first
    restriction_estimator = RestrictionEstimator(
        prior=prior,
        hidden_features=cfg.hidden_features_restrictor,
        num_blocks=cfg.num_blocks_restrictor,
        z_score=cfg.z_score_theta,
    )

    restriction_estimator.append_simulations(theta, x)
    classifier = restriction_estimator.train()
    # proposal = restriction_estimator.restrict_prior()
    proposal = process_prior(restriction_estimator.restrict_prior())[0]

    inference, density_estimator = train_NPE(theta, x, proposal, cfg)
    return inference, density_estimator


## NLE
def train_NLE(theta, x, prior, cfg):
    net = get_nn_models.likelihood_nn(
        model=cfg.density_estimator,
        z_score_x=cfg.z_score_x,
        z_score_theta=cfg.z_score_theta,
        num_transforms=cfg.num_transforms,
        num_bins=cfg.num_bins,
    )
    inference = SNLE(prior=prior, density_estimator=cfg.density_estimator)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator


## NRE
def train_NRE(theta, x, prior, cfg):
    embedding_net_x = _get_embedding_net(cfg, input_dim=x.shape[1])
    net = get_nn_models.classifier_nn(
        model=cfg.classifier,
        z_score_theta=cfg.z_score_theta,
        z_score_x=cfg.z_score_x,
        embedding_net_x=embedding_net_x,
    )
    inference = SNRE(prior=prior, classifier=net)
    density_estimator = inference.append_simulations(theta, x).train()
    return inference, density_estimator


# Regression
def train_Regression(theta, x, prior, cfg):
    inference = RegressionInference(
        theta,
        x,
        predict_theta=(cfg.name == "REGR-R"),
        num_layers=cfg.num_layers,
        num_hidden=cfg.num_hidden,
        net_type=cfg.net_type,
    )
    ensemble = []
    for i in range(cfg.n_ensemble):
        print("--------------------")
        print(f"Training {i+1} of {cfg.n_ensemble} networks...")
        inference.init_network(seed=i)
        neural_net = inference.train(
            training_batch_size=cfg.training_batch_size,
            max_n_epochs=cfg.max_n_epochs,
            stop_after_counter_reaches=cfg.stop_after_counter_reaches,
            validation_fraction=cfg.validation_fraction,
            print_every_n=cfg.print_every_n,
            plot_losses=cfg.plot_losses,
        )
        ensemble.append(neural_net)
    if cfg.n_ensemble == 1:
        ensemble = ensemble[0]
    return inference, ensemble


def _concatenate_xs(x1, x2):
    """Concatenate two tensors along the first dimension."""
    return torch.concat([x1, x2], dim=0)


# GBI ACE
def train_ACE(theta, x, prior, cfg):
    # Augment data with noise.
    if type(cfg.n_augmented_x) is int:
        n_augmented_x = cfg.n_augmented_x
    elif type(cfg.n_augmented_x) is float:
        n_augmented_x = int(cfg.n_augmented_x * x.shape[0])
    else:
        n_augmented_x = 0

    x_aug = x[torch.randint(x.shape[0], size=(n_augmented_x,))]
    x_aug = x_aug + torch.randn(x_aug.shape) * x.std(dim=0) * cfg.noise_level
    x_target = _concatenate_xs(x, x_aug)

    if cfg.train_with_obs:
        # Append observations.
        x_obs = data_utils.load_pickled(cfg.obs_data_path)
        # Put all together.
        x_target = _concatenate_xs(x_target, x_obs)

    distance_func = get_distance_function(cfg.dist_func)
    inference = GBInference(
        prior=prior,
        distance_func=distance_func,
        do_precompute_distances=cfg.do_precompute_distances,
        include_bad_sims=cfg.include_bad_sims,
        nan_dists_replacement=cfg.nan_dists_replacement,
    )
    inference = inference.append_simulations(theta, x, x_target, n_dists_precompute=5.0)
    inference.initialize_distance_estimator(
        num_layers=cfg.num_layers,
        num_hidden=cfg.num_hidden,
        net_type=cfg.net_type,
        positive_constraint_fn=cfg.positive_constraint_fn,
    )
    distance_net = inference.train(
        training_batch_size=cfg.training_batch_size,
        max_n_epochs=cfg.max_epochs,
        validation_fraction=cfg.validation_fraction,
        n_train_per_theta=cfg.n_train_per_theta,
        n_val_per_theta=cfg.n_val_per_theta,
        stop_after_counter_reaches=cfg.stop_after_counter_reaches,
        print_every_n=cfg.print_every_n,
        plot_losses=cfg.plot_losses,
    )
    return inference, distance_net


## embedding nets
def _get_embedding_net(cfg, input_dim):
    if cfg.use_embedding_net:
        embedding_net = FCEmbedding(
            input_dim=input_dim,
            output_dim=cfg.embedding_net_outdim,
            num_layers=cfg.embedding_net_layers,
            num_hiddens=cfg.embedding_net_num_hiddens,
        )
    else:
        embedding_net = nn.Identity()
    return embedding_net
