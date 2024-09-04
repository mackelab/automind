### Brian2 interface functions, e.g., remove / add units, etc.
import brian2 as b2


def _deunitize_spiketimes(spiketrains_dict):
    """
    Takes a spiketrain dictionary and return a copy without b2 time units.
    Output spiketrains_dict are assumed to be in units of seconds.

    Args:
        spiketrains_dict (dict): Dictionary of spiketimes in the form {cell_index: array(spiketimes)}

    Returns:
        dict: Dictionary of spiketimes without units.

    """
    return {k: v / b2.second for k, v in spiketrains_dict.items()}


def _deunitize_rates(t, rate):
    """Deunitize timestamps and rate."""
    return t / b2.second, rate / b2.Hz


def set_adaptive_vcut(param_dict):
    param_dict["v_cut"] = param_dict["v_thresh"] + 5 * param_dict["delta_T"]


def parse_timeseries(net_collect, record_defs):
    """Parses recordings of continuous variables

    Args:
        net_collect (brian2 net): collector that has all the monitors.
        record_defs (dict): nested dict that defines recorded populations & variables.

    Returns:
        dict: time series
    """
    timeseries = {}
    for pop_name in record_defs.keys():
        if record_defs[pop_name]["rate"]:
            # deunitize rate
            t_rate = net_collect[pop_name + "_rate"].t / b2.second
            timeseries[pop_name + "_rate"] = (
                net_collect[pop_name + "_rate"].rate / b2.Hz
            )
            timeseries["t_rate"] = t_rate

        if record_defs[pop_name]["trace"]:
            var_names = record_defs[pop_name]["trace"][0]
            for vn in var_names:
                # taking the average of the recorded state variables
                timeseries[pop_name + "_" + vn] = getattr(
                    net_collect[pop_name + "_trace"], vn + "_"
                ).mean(0)
                t = getattr(net_collect[pop_name + "_trace"], "t") / b2.second
            timeseries["t"] = t

    return timeseries


def strip_b2_units(theta_samples, theta_priors):
    """Strips brian2 units from theta samples.

    Args:
        theta_samples (dict): Dictionary of parameter samples.
        theta_priors (dict): Prior, in the form of a dictionary.

    Returns:
        dict: Dictionary of parameter samples without units.
    """
    # this is pretty much only for converting to and saving as dataframe
    # WILL break if data not in array but I don't really care at this point
    theta_samples_unitless = {}
    for k, v in theta_samples.items():
        if k in theta_priors.keys():
            theta_samples_unitless[k] = (v / theta_priors[k]["unit"]).astype(v.dtype)
        else:
            theta_samples_unitless[k] = v

    return theta_samples_unitless


def clear_b2_cache(cache_path=None):
    """Clears brian2 cache.

    Args:
        cache_path (str, optional): location of cache.
    """
    if cache_path is None:
        try:
            b2.clear_cache("cython")
            print("cache cleared.")
        except:
            print("cache non-existent.")
    else:
        import shutil

        try:
            shutil.rmtree(cache_path)
            print(f"cache cleared: {cache_path}.")
        except:
            print("cache non-existent.")


def set_b2_cache(cache_dir, file_lock=False):
    """Set brian2 cache, optionally apply file lock."""
    b2.prefs.codegen.runtime.cython.cache_dir = cache_dir
    b2.prefs.codegen.runtime.cython.multiprocess_safe = file_lock
