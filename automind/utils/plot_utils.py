### Utility functions for plotting, likely to be changed in the future.

import matplotlib.pyplot as plt

# plt.style.use("../../../assets/matplotlibrc_notebook")
import numpy as np
from ..utils import data_utils, dist_utils
from ..sim.default_configs import MKI_3col_plot_order


def plot_rates_tiny(
    rates_to_plot,
    figsize,
    fig_axs=None,
    color="k",
    alpha=0.8,
    lw=1,
    XL=(0, 60),
    ylim_past=None,
    fontsize=14,
):
    if fig_axs is None:
        fig, axs = plt.subplots(ncols=1, nrows=len(rates_to_plot), figsize=figsize)
    else:
        fig, axs = fig_axs

    for i_r, rates in enumerate(rates_to_plot):
        axs[i_r].plot(
            rates["t_ds"],
            rates["exc_rate"],
            alpha=alpha,
            color=color[i_r] if type(color) == list else color,
            lw=lw,
        )
        axs[i_r].set_xlim(XL)
        axs[i_r].set_xticks([])

        if ylim_past:
            ymax = rates["exc_rate"][ylim_past:].max()
            axs[i_r].set_ylim([0, ymax * 1.1])

        axs[i_r].set_yticks([int(np.ceil(axs[i_r].get_ylim()[1] / 1.1 / 5) * 5)])
        # axs[i_r].set_yticks([round(rates["exc_rate"].max() / 10) * 10])
        axs[i_r].spines["bottom"].set_visible(False)

    # axs[i_r].set_xticks(ticks=axs[i_r].get_xlim())
    # axs[i_r].set_xticklabels(labels=XL)
    axs[i_r].set_xlabel(f"{XL[1]-XL[0]} s", fontsize=fontsize)
    return fig, axs


def plot_rates_and_bursts(
    pop_rates,
    burst_stats,
    vars_to_plot=None,
    burst_alpha=0.5,
    figsize=(20, 3),
    w_ratio=(5, 1),
    fig_handles=None,
    burst_time_offset=-0.5,
):
    if fig_handles is None:
        fig, axs = plt.subplots(
            ncols=2, gridspec_kw={"width_ratios": w_ratio}, figsize=figsize
        )
    else:
        # just unpack input
        fig, axs = fig_handles

    fs = 1 / (pop_rates["t_ds"][1] - pop_rates["t_ds"][0])

    if vars_to_plot is None:
        # just plot exc rate
        ts_color = "C0"
        axs[0].plot(pop_rates["t_ds"], pop_rates["exc_rate"], alpha=0.9, color=ts_color)
    else:
        ts_color = list(vars_to_plot.values())[0]
        for k, c in vars_to_plot.items():
            axs[0].plot(pop_rates["t_ds"], pop_rates[k], alpha=0.7, color=c)

    if len(burst_stats["burst_times"]) > 0:
        # if there are bursts, mark burst peak and widths
        axs[0].plot(
            burst_stats["burst_times"],
            burst_stats["burst_heights"],
            "ow",
            mec="k",
            ms=6,
            alpha=0.9,
        )

    if len(burst_stats["burst_kernels"]) > 0:
        # plot kernels
        if "burst_kernels_refined" in burst_stats:
            ips = burst_stats["burst_width_ips"]
            for i_b, bk in enumerate(burst_stats["burst_kernels_refined"]):
                # t_burst = np.arange(int(ips[i][0]*fs), len(bk))
                t_burst = (
                    burst_stats["burst_width_ips"][i_b][0]
                    - burst_stats["burst_times"][i_b]
                ) + np.arange(len(bk)) / fs
                axs[1].plot(t_burst, bk, color=ts_color, alpha=burst_alpha, lw=1)

                # ips_idx = np.arange(int(ips[i][0]*fs), int(ips[i][1]*fs), dtype=int)

        else:
            axs[1].plot(
                pop_rates["t_ds"][: burst_stats["burst_kernels"][0].shape[0]]
                + burst_time_offset,
                np.array(burst_stats["burst_kernels"]).T,
                color=ts_color,
                alpha=burst_alpha,
                lw=1,
            )
        for i, ips in enumerate(burst_stats["burst_width_ips"]):
            b_height = burst_stats["burst_width_heights"][i]
            b_time = burst_stats["burst_times"][i]
            axs[0].plot([ips[0], ips[1]], [b_height] * 2, "b", lw=4)
            axs[1].plot(
                [ips[0] - b_time, ips[1] - b_time],
                [b_height + abs(np.random.randn()) * b_height * 2] * 2,
                "b",
                lw=1,
                alpha=0.75,
            )
        if "subpeak_times" in burst_stats:
            for subpeak_time in burst_stats["subpeak_times"]:
                subpeak_idx = (subpeak_time * fs).astype(int)
                axs[0].plot(
                    subpeak_time,
                    pop_rates["exc_rate"][subpeak_idx],
                    "ow",
                    mec="b",
                    ms=3,
                    alpha=0.8,
                )

    plt.tight_layout()
    return fig, axs


def plot_wrapper(pop_rates, burst_stats, random_seed, params_dict):
    fig, axs = plot_rates_and_bursts(
        pop_rates,
        burst_stats,
        vars_to_plot={"exc_rate": "k"},
        burst_alpha=0.2,
        figsize=(20, 2.5),
        w_ratio=(8, 1),
        burst_time_offset=params_dict["params_analysis"]["burst_win"][0],
    )
    axs[0].set_xlim([0, 120])
    # axs[0].set_ylim([0, None])
    axs[1].set_xlim(params_dict["params_analysis"]["burst_win"])
    axs[1].set_ylim([0, None])
    run_id = f"{params_dict['params_settings']['batch_seed']}_{random_seed}"
    axs[0].set_title(f"{run_id}")
    plt.savefig(params_dict["path_dict"]["figures"] + f"{run_id}_analyzed.pdf")
    plt.close()


def plot_wrapper_MK1(summary_stats, random_seed, params_dict):
    fig, axs = plt.subplots(
        ncols=4, gridspec_kw={"width_ratios": (7, 1, 1, 1)}, figsize=(20, 2.5)
    )
    plot_rates_and_bursts(
        summary_stats["pop_rates"],
        summary_stats["summary_burst_stats"],
        vars_to_plot={"exc_rate": "k"},
        burst_alpha=0.2,
        fig_handles=(fig, axs),
        burst_time_offset=params_dict["params_analysis"]["burst_win"][0],
    )

    axs[2].loglog(
        data_utils.decode_df_float_axis(summary_stats["summary_psd"].columns, float),
        summary_stats["summary_psd"].loc["exc_rate"],
        "C1",
        label="exc",
    )
    axs[2].loglog(
        data_utils.decode_df_float_axis(summary_stats["summary_psd"].columns, float),
        summary_stats["summary_psd"].loc["inh_rate"],
        "C4",
        label="inh",
    )
    axs[2].legend()

    # axs[3].loglog(np.arange(len(summary_stats['summary_pca'])-1)+1, summary_stats['summary_pca']['var_exp_ratio'].iloc[1:], 'o')
    axs[3].loglog(
        data_utils.decode_df_float_axis(summary_stats["summary_pca"].columns[1:], int),
        summary_stats["summary_pca"].iloc[0].values[1:],
        "o",
    )

    run_id = f"{params_dict['params_settings']['batch_seed']}_{random_seed}"
    if len(summary_stats["summary_burst_stats"]["burst_heights"]) > 0:
        YMAX = np.median(summary_stats["summary_burst_stats"]["burst_heights"]) * 1.1
    else:
        fs = 1 / (
            summary_stats["pop_rates"]["t_ds"][1]
            - summary_stats["pop_rates"]["t_ds"][0]
        )
        YMAX = summary_stats["pop_rates"]["exc_rate"][int(20 * fs) :].max() * 1.1
    axs[0].set_xlim([0, 120])
    axs[0].set_ylim([0, YMAX])
    axs[0].set_title(run_id)
    axs[1].set_xlim(params_dict["params_analysis"]["burst_win"])
    axs[1].set_ylim([0, YMAX])
    axs[1].set_title("bursts")

    axs[2].set_xlim([None, params_dict["params_analysis"]["f_lim"]])
    axs[2].set_xticks([1, 100])
    axs[2].set_xticklabels(["1", "100"])
    axs[2].set_title("PSD")

    axs[3].set_xticks([1, 10, 100])
    axs[3].set_xticklabels(["1", "10", "100"])
    axs[3].set_title("PCA eigval")

    plt.tight_layout()
    plt.savefig(params_dict["path_dict"]["figures"] + f"{run_id}_analyzed.pdf")
    plt.close()


def add_point_to_pairplot(df_point_plot, x_features, axs, **kwargs):
    for i_y, feat_x in enumerate(x_features):
        for i_x, feat_y in enumerate(x_features):
            if i_x > i_y:
                axs[i_y, i_x].plot(
                    df_point_plot[feat_y], df_point_plot[feat_x], **kwargs["upper"]
                )
            if i_x == i_y:
                axs[i_y, i_x].axvline(df_point_plot[feat_y], **kwargs["diag"])


def _plot_raster_pretty(
    spikes,
    XL,
    every_other=1,
    ax=None,
    fontsize=14,
    plot_combined=False,
    plot_inh=True,
    E_color="k",
    I_color="gray",
    mew=0.5,
    ms=1,
    **plot_kwargs,
):
    if ax == None:
        ax = plt.axes()

    if plot_combined:
        combined_spikes = list(spikes["exc_spikes"].values()) + list(
            spikes["inh_spikes"].values()
        )
        if plot_combined == "sorted":
            c_order = np.argsort([len(v) for v in combined_spikes])
        elif plot_combined == "random":
            c_order = np.random.permutation(len(combined_spikes))
        else:
            c_order = np.arange(len(combined_spikes))

        combined_spikes = [combined_spikes[i] for i in c_order]

        # import pdb; pdb.set_trace()
        [
            (
                ax.plot(
                    v[::every_other],
                    i_v * np.ones_like(v[::every_other]),
                    # c_order[i_v] * np.ones_like(v[::every_other]),
                    "|",
                    color=E_color,
                    alpha=1,
                    ms=ms,
                    mew=mew,
                )
                if len(v) > 0
                else None
            )
            for i_v, v in enumerate(combined_spikes)
        ]
    else:
        # plot exc (and inh) separately
        [
            (
                ax.plot(
                    v[::every_other],
                    i_v * np.ones_like(v[::every_other]),
                    "|",
                    color=E_color,
                    alpha=1,
                    ms=ms,
                    mew=mew,
                )
                if len(v) > 0
                else None
            )
            for i_v, v in enumerate(spikes["exc_spikes"].values())
        ]
        # plot inh
        if plot_inh:
            [
                (
                    ax.plot(
                        v[::every_other],
                        (i_v + len(spikes["exc_spikes"]))
                        * np.ones_like(v[::every_other]),
                        "|",
                        color=I_color,
                        alpha=1,
                        ms=ms,
                        mew=mew,
                    )
                    if len(v) > 0
                    else None
                )
                for i_v, v in enumerate(spikes["inh_spikes"].values())
            ]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xlim(XL)
    ax.set_ylabel("Raster", fontsize=fontsize)
    return ax

def _plot_rates_pretty(
    rates,
    XL,
    pops_to_plot=["exc_rate"],
    ylim_past=None,
    ax=None,
    fontsize=14,
    lw=0.5,
    color=None,
):
    if ax == None:
        ax = plt.axes()
    if color is None:
        C_dict = {"exc_rate": "C1", "inh_rate": "C4", "avgpop_rate": "C0"}
    else:
        C_dict = {"avgpop_rate": color}

    for pop in pops_to_plot:
        ax.plot(rates["t_ds"], rates[pop], lw=lw, alpha=1, color=C_dict[pop])
    # ax.set_yticks([])

    ax.set_ylabel("Rate", fontsize=fontsize, labelpad=None)
    ax.set_xticks([])
    ax.set_xlim(XL)
    if ylim_past:
        ymax = rates["exc_rate"][ylim_past:].max()
        ax.set_ylim([0, ymax * 1.1])
        # ax.set_yticks([round(ymax / 10) * 10])

    ax.set_yticks([0, int(np.ceil(ax.get_ylim()[1] / 5) * 5)])
    # ax.set_yticks([round(1+ax.get_ylim()[1] / 10) * 10])
    ax.set_xlabel(f"{XL[1]-XL[0]} s", fontsize=fontsize)
    ax.spines.bottom.set_visible(False)
    return ax


def _plot_eigspec_pretty(
    df_pca, n_pcs, ax=None, fontsize=14, color="C0", alpha=1, ms=1, lw=0
):
    if ax == None:
        ax = plt.axes()
    pcs = data_utils.decode_df_float_axis(df_pca.columns.values[1:])
    pca = df_pca.iloc[0, 1:].values
    ax.loglog(
        pcs[:n_pcs], pca[:n_pcs] * 100, "-o", color=color, lw=lw, alpha=alpha, ms=ms
    )
    ax.set_xticks([1, 10], ["1", "10"])
    # ax.set_yticks([1e-5, 1], [r"$10^{-5}$", "1"])
    ax.minorticks_off()
    ax.set_ylabel("PCA", fontsize=fontsize)
    return ax


def _plot_psd_pretty(
    df_psd,
    pops_to_plot=["exc_rate", "inh_rate"],
    ax=None,
    fontsize=14,
    color=None,
    alpha=1,
):
    if ax == None:
        ax = plt.axes()
    f_axis = data_utils.decode_df_float_axis(df_psd.columns.values)
    if color is None:
        C_dict = {"exc_rate": "C1", "inh_rate": "C4", "avgpop_rate": "C0"}
    else:
        C_dict = {"avgpop_rate": color}
    for pop in pops_to_plot:
        ax.loglog(
            f_axis, df_psd.loc[pop].values, color=C_dict[pop], lw=0.8, alpha=alpha
        )
    ax.set_xticks([1, 10, 100], ["1", "10", "100"])
    ax.set_yticks([])
    ax.set_xlim([0.5, 450])
    ax.set_ylabel("PSD", fontsize=fontsize)
    ax.minorticks_off()
    return ax


def strip_decimal(s):
    return int(s) if int(s) == s else s


def plot_params_pretty(
    thetas,
    prior,
    param_names,
    figsize=(2.5, 10),
    fig_axs=None,
    labelpad=40,
    fontsize=14,
    **plot_kwarg,
):

    n_params = thetas.shape[1]
    params_bound = [
        [prior.marginals[i].low, prior.marginals[i].high] for i in range(n_params)
    ]

    if fig_axs is None:
        fig, axs = plt.subplots(n_params, 1, figsize=figsize, constrained_layout=False)
    else:
        fig, axs = fig_axs

    for i in range(n_params):
        if plot_kwarg == {}:
            [axs[i].plot(th[i], 0, "o", ms=5, alpha=0.8) for th in thetas]
        else:
            [axs[i].plot(th[i], 0, **plot_kwarg) for th in thetas]

            # axs[i].errorbar(posterior_samples[:n_best,i].mean(), 0, xerr=stats.sem(posterior_samples[:n_best,i])*3, color=plt_colors[i_q], alpha=0.8)

        if fig_axs is None:
            axs[i].spines.left.set_visible(False)
            axs[i].spines.bottom.set_visible(False)
            axs[i].axhline(0, alpha=0.1, lw=1)

            axs[i].text(
                np.array(params_bound[i])[0],
                0,
                strip_decimal(np.array(params_bound[i])[0]),
                ha="right",
                va="center",
                fontsize=fontsize,
            )
            axs[i].text(
                np.array(params_bound[i])[1],
                0,
                strip_decimal(np.array(params_bound[i])[1]),
                ha="left",
                va="center",
                fontsize=fontsize,
            )

            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_ylabel(
                param_names[i],
                rotation=0,
                labelpad=labelpad,
                ha="right",
                va="center",
                fontsize=fontsize,
            )

            axs[i].set_xlim(np.array(params_bound[i]))
            axs[i].set_ylim([-0.1, 0.1])
        # plt.subplots_adjust(
        #     left=None, bottom=None, right=None, top=None, wspace=0.0, hspace=0
        # )
    return fig, axs


def plot_params_1D(
    thetas,
    param_bounds,
    param_names,
    fig_axs,
    color,
    draw_canvas=True,
    draw_kde=True,
    flip_density=False,
    draw_median=True,
    draw_samples=False,
    labelpad=40,
    fontsize=5,
    kde_points=100,
    **plot_kwarg,
):
    fig, axs = fig_axs

    if len(axs.shape) == 2:
        # split mode
        axes = axs.T.flatten()
        # hard coded order
        plt_order = MKI_3col_plot_order
    else:
        # single column mode
        axes = axs
        plt_order = np.arange(thetas.shape[1])
        assert len(axs) == thetas.shape[1]

    for i_, ax in enumerate(axes):
        i_a = plt_order[i_]
        if i_a == -1:
            ax.axis("off")
            continue

        # assert len(axs)==thetas.shape[1]
        # for i_a, ax in enumerate(axs):
        if draw_kde:
            kde = dist_utils.kde_estimate(thetas[:, i_a], param_bounds[i_a], kde_points)
            ax.fill_between(
                kde[0],
                -kde[1] if flip_density else kde[1],
                alpha=0.6,
                color=color,
                lw=0.5,
            )

        if draw_median:
            ax.plot(np.median(thetas[:, i_a]), 0, "|", color=color, mew=0.75, ms=2)

        if draw_samples:
            samples = thetas[:10, i_a]
            ax.plot(
                samples,
                samples * 0.0,
                ".",
                color=color,
                mew=0.75,
                ms=plot_kwarg["sample_ms"],
                alpha=plot_kwarg["sample_alpha"],
            )

        if draw_canvas:
            ax.axhline(0, lw=0.25, alpha=1, zorder=-1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(param_bounds[i_a])
            ax.spines.left.set_visible(False)
            ax.spines.bottom.set_visible(False)

            for i_s, side in enumerate(["right", "left"]):
                # Aligned to the opposite end of word, so left to right
                ax.text(
                    np.array(param_bounds[i_a])[i_s],
                    0,
                    strip_decimal(np.array(param_bounds[i_a])[i_s]),
                    ha=side,
                    va="center",
                    fontsize=fontsize,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(
                param_names[i_a],
                rotation=0,
                labelpad=labelpad,
                ha="right",
                va="center",
                fontsize=fontsize,
            )
            ax.set_xlim(np.array(param_bounds[i_a]))

        # YL=np.abs(ax.get_ylim()).max()
        # ax.set_ylim(YL*np.array([-1,1]))
    return fig, axs


def plot_corr_pv(pvals, ax, alpha_level=0.05, fmt="w*", ms=0.5):
    for i in range(pvals.shape[0]):
        for j in range(pvals.shape[0]):
            if pvals[i, j] < alpha_level:
                ax.plot(j, i, fmt, ms=ms, alpha=1)


#Just use the default plotting function with the sorted spikes 
'''
def plot_raster(
    spikes,
    membership,
    XL,
    plotting_method="cluster_identity",
    every_other=1,
    ax=None,
    fontsize=14,
    plot_inh=False,
    E_colors=None,
    I_color="gray",
    single_cluster_style="|",
    double_cluster_style="x",
    mew=0.5,
    ms=1,
    **plot_kwargs,
):
    """
    Plot raster plot with neurons sorted by cluster identity or number of clusters.

    Parameters:
        spikes (dict): Dictionary containing 'exc_spikes' and 'inh_spikes'.
        membership: Array/list of 2D membership arrays (from params_net['membership']).
        XL (list): X-axis limits.
        plotting_method (str): "cluster_identity" or "n_clusters".
        every_other (int): Plot every nth spike.
        ax (matplotlib axis): Axis to plot on.
        fontsize (int): Font size for labels.
        plot_inh (bool): Whether to plot inhibitory spikes.
        E_colors (list): Colors for excitatory clusters.
        I_color (str): Color for inhibitory spikes.
        single_cluster_style (str): Marker style for single-cluster neurons.
        double_cluster_style (str): Marker style for two-cluster neurons.
        mew (float): Marker edge width.
        ms (float): Marker size.
    """
    if ax is None:
        ax = plt.axes()

    exc_spikes = spikes["exc_spikes"]
    inh_spikes = spikes.get("inh_spikes", {})

    if plotting_method == "cluster_identity":
        # Sort by cluster identity
        sorted_indices = data_utils.sort_neurons(membership, sorting_method='cluster_identity')
        sorted_indices_list = sorted_indices[0].tolist()  # Convert to list of Python integers
        sorted_exc_spikes = {i: exc_spikes[idx] for i, idx in enumerate(sorted_indices_list)}
        #exc_spikes_to_plot = sorted_exc_spikes.values()
    elif plotting_method == "n_clusters":
        # Sort by number of clusters
        sorted_indices = data_utils.sort_neurons(membership, sorting_method='n_clusters')
        sorted_exc_spikes_single = {i: exc_spikes[idx] for i, idx in enumerate(sorted_indices[0][0])}
        sorted_exc_spikes_double = {i: exc_spikes[idx] for i, idx in enumerate(sorted_indices[1][0])}
        #exc_spikes_to_plot.append(sorted_exc_spikes_single)
        #exc_spikes_to_plot.append(sorted_exc_spikes_double)
    else:
        raise ValueError("Invalid plotting_method. Use 'cluster_identity' or 'n_clusters'.")

    # Plot excitatory spikes, single cluster in blue and double cluster in red respectively
    [
        (
            ax.plot(
                v[::every_other],
                i_v * np.ones_like(v[::every_other]),
                single_cluster_style,
                color='blue',
                alpha=1,
                ms=ms,
                mew=mew,
            )
            if len(v) > 0
            else None
        )
        for i_v, (t,v) in enumerate(sorted_exc_spikes_single.items())
    ]
    [
        (
            ax.plot(
                v[::every_other],
                (i_v+ len(sorted_indices[0][0])) * np.ones_like(v[::every_other]),
                single_cluster_style,
                color='red',
                alpha=1,
                ms=ms,
                mew=mew,
            )
            if len(v) > 0
            else None
        )
        for i_v, (t,v) in enumerate(sorted_exc_spikes_double.items())
    ]

    # Plot inhibitory spikes
    if plot_inh:
        [
            (
                ax.plot(
                    v[::every_other],
                    (i_v + len(sorted_indices[0][0]) + len(sorted_indices[1][0])) * np.ones_like(v[::every_other]),
                    "|",
                    color=I_color,
                    alpha=1,
                    ms=ms,
                    mew=mew,
                )
                if len(v) > 0
                else None
            )
            for i_v, v in enumerate(inh_spikes.values())
        ]

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_xlim(XL)
    ax.set_ylabel("Raster", fontsize=fontsize)
    return ax
'''
