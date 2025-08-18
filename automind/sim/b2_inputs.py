#Brian2 functions related to additional inputs 
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)

# Add input parameters as dictionary

def get_n_cluster_to_stim(param_dict, fraction):
    '''
    Generatae number of clusters to stimulate based on fraction 
    Args:
        param_dict: Existing parameter dicitonary   
        fraction: Fraction of clusters to stimulate (0-1)
    '''
    params_dict_net = param_dict["params_net"]
    has_clusters = (
    "n_clusters" in params_dict_net.keys() 
    and params_dict_net["n_clusters"] >= 2 
    and params_dict_net["R_pe2e"] != 1
    )
    if has_clusters:
        n_cluster_original = param_dict["params_net"]["n_clusters"]
        n_cluster_to_stim = int(n_cluster_original * fraction)
        if n_cluster_to_stim < 1:
            print("Fraction too low, one cluster will be stimulated.")
            n_cluster_to_stim = 1
        elif n_cluster_to_stim > n_cluster_original:
            raise ValueError(
                f"Fraction too high, cannot stimulate more clusters ({n_cluster_to_stim}) than available ({n_cluster_original})."
            )
    else:
        print("Network does not have clusters. Using single input mode instead.")

    return n_cluster_to_stim

def generate_params_input_subdict(param_dict, mode="default", single_input=None, 
                                   n_cluster_to_stim=None, cluster_input_list=None, 
                                   cluster_weight_list=None, selected_clusters=None):
    """
    Add input parameters to params_dict
    
    Args:
        param_dict: Existing parameter dictionary
        mode: "default", "single", or "cluster"
        single_input: Input array for single mode (if None, generates rectangle_input)
        n_cluster_to_stim: Number of clusters to stimulate (defaults to all)
        cluster_input_list: List of input arrays for each cluster (if None, generates cluster_specific_stim)
        cluster_weight_list: List of weights for each cluster (if None, generates cluster_specific_stim)
        selected_clusters: Specific cluster IDs to use (defaults to random selection)
    
    Returns:
        param_dict: Modified parameter dictionary with params_input added
    """
    
    if mode not in ["default", "single", "cluster"]:
        raise ValueError(f"Invalid mode: '{mode}'. Use 'default', 'single', or 'cluster'.")
    
    params_input = {"mode": mode}
    
    if mode == "single":
        # Generate default single input if not provided
        if single_input is None:
            single_input = rectangle_input(param_dict)
            print("Generated default rectangle input for single mode")
        params_input["single_input"] = single_input
        
    elif mode == "cluster":
        # Check if network has clusters
        params_net = param_dict["params_net"]
        if not (params_net.get("n_clusters", 0) >= 2 and params_net.get("R_pe2e") != 1): #No clusters
            print("Warning: Network has no clusters. Switching to single mode.")
            params_input["mode"] = "single"
            single_input = rectangle_input(param_dict)
            print("Generated default rectangle input for fallback single mode")
            params_input["single_input"] = single_input
        else:
            n_total_clusters = int(params_net["n_clusters"])
            
            # Default to all clusters if not specified
            if n_cluster_to_stim is None:
                n_cluster_to_stim = n_total_clusters
                print(f"Using all {n_total_clusters} clusters")
            
            # Generate default cluster inputs if not provided
            if cluster_input_list is None or cluster_weight_list is None:
                cluster_input_list, cluster_weight_list = cluster_specific_stim(
                    param_dict, n_cluster_to_stim
                )
                print(f"Generated default cluster-specific stimuli for {n_cluster_to_stim} clusters")
            
            # Default to random selection if not specified
            if selected_clusters is None:
                selected_clusters = np.random.choice(n_total_clusters, n_cluster_to_stim, replace=False)
                print(f"Randomly selected clusters: {selected_clusters}")
            
            # Validate inputs match number of clusters to stimulate
            if len(cluster_input_list) != n_cluster_to_stim:
                raise ValueError(f"cluster_input_list length ({len(cluster_input_list)}) != n_cluster_to_stim ({n_cluster_to_stim})")
            
            if len(cluster_weight_list) != n_cluster_to_stim:
                raise ValueError(f"cluster_weight_list length ({len(cluster_weight_list)}) != n_cluster_to_stim ({n_cluster_to_stim})")
            
            params_input.update({
                "n_clusters_to_stim": n_cluster_to_stim,
                "cluster_input_list": cluster_input_list,
                "cluster_weight_list": cluster_weight_list,
                "selected_clusters": selected_clusters,
            })
    
    param_dict["params_input"] = params_input
    return param_dict

def rectangle_input(all_param_dict, mu = 10**-9, sigma = 2.5*10**-11, buffer_period = 10, stim_time = 5):
    """   
    Inputs: 
    params_dict: dt and sim_time 

    Optional:
    mu: mean of stimulus
    sigma: standard deviation of stimulus
    buffer period (in seconds): How long before the stimulus is applied
    stim_time (in seconds): How long the stimulus is applied 

    Returns: 
    A rectangle stimulus lower bounded at 0
    """
    param_dict_settings = all_param_dict["params_settings"]

    #Set random seeds
    b2.seed(param_dict_settings["random_seed"])
    dt = param_dict_settings["dt"]

    #Setting stimulation and simulation length 
    stim_steps = int(stim_time * b2.second / dt) 
    one_second = int(b2.second / dt)
    buffer_period = buffer_period * one_second 
    total_steps = int(param_dict_settings['sim_time'] / dt)
    if stim_steps > total_steps: 
        raise ValueError("Stimulus steps exceed total simulation steps.")
    stim = np.zeros(total_steps)
    stim[buffer_period:buffer_period+stim_steps] = np.random.normal(mu, sigma, stim_steps)
    stim = np.maximum(stim,0) #Lower bound at 0
    return stim

#Use this to find input output range of network
def sequential_input(all_param_dict, buffer_period=10, vals = np.linspace(10**-10, 2.5*10**-9, 25), pulse_length = 1, interpulse_length = 1):
    '''
    Outputs a stimulus flickering between zero and provided input strengths.
    
    Inputs:
    all_param_dict: Parameter dictionary containing simulation settings

    Optional:
    buffer_period: Initial buffer period in seconds before stimulation starts (default: 10s)
    vals: Array of stimulation amplitudes to use (default: np.linspace(10**-10, 2.5*10**-9, 25))
    pulse_length: Length of each pulse in seconds (default: 1s)
    interpulse_length: Length of the interval between pulses in seconds (default: 1s)
    Stim time depends on all_param_dict. Around 60s of stim time is sufficient for all pulses to be used under default settings
    Returns:
    --------
    stim : numpy.ndarray
        Array containing the full stimulus time series
    '''

    param_dict_settings = all_param_dict["params_settings"]
    
    #Get dt and sim_time
    dt = param_dict_settings["dt"]
    
    # Convert time values to steps
    one_second = int(b2.second / dt)
    pulse_length_steps = int(pulse_length * one_second)
    interpulse_length_steps = int(interpulse_length * one_second)
    buffer_period_steps = int(buffer_period * one_second)
    total_steps = int(param_dict_settings['sim_time'] / dt)
    
    stim = np.zeros(total_steps)
    
    # Calculate how many full cycles we can fit after the buffer period
    available_steps = total_steps - buffer_period_steps
    cycle_length = pulse_length_steps + interpulse_length_steps # ON then OFF

    if len(vals) > available_steps // cycle_length:
        usable_vals = available_steps // cycle_length
        print(f"Warning: Not all stimulus values can fit in the simulation time. "
                f"Using {usable_vals} out of {len(vals)} values.")
    
    # Check if all values can fit within the simulation time
    for i in range(min(len(vals), available_steps // cycle_length)):
        start_idx = buffer_period_steps + i * cycle_length
        end_idx = start_idx + pulse_length_steps
        if end_idx <= total_steps:  # Safety check
            stim[start_idx:end_idx] = vals[i]
        else:
            print(f"Warning: Pulse {i+1} with stimulus amplitude {vals[i+1]} exceeds total simulation steps. Stopping early.")
            break
    return stim

def cluster_specific_stim(all_param_dict, n_clusters_to_stim, stim_sec = 5, buffer_sec = 5, means=np.linspace(10**-10,10**-9,25), std = 2.5*10**-11 ):
    """Generate cluster-specific stimuli and weights to be used in input_configs. 
    
    Parameters
    ----------
    all_param_dict: Network parameters
    n_clusters_to_stim: Number of clusters to stimulate
        
    Returns
    -------
    stim_list: List of stimulus arrays
    weight_list: List of weight arrays
    """

    #Setting param dicts
    param_dict_settings = all_param_dict["params_settings"]
    b2.seed(param_dict_settings["random_seed"])
    param_dict_net = all_param_dict["params_net"]
    N_exc = int(param_dict_net["N_pop"] * param_dict_net["exc_prop"])
    dt = param_dict_settings["dt"]

    #Stimulation length
    stim_steps = int(stim_sec * b2.second / dt) 
    one_second = int(b2.second / dt)
    buffer_period = buffer_sec * one_second #Allow network to stabilise first
    total_steps = int(param_dict_settings['sim_time'] / dt)
    if stim_steps > total_steps:
        raise ValueError("Stimulus steps exceed total simulation steps.")
    
    #List of stimuli and weights
    stim_list = []
    weight_list = []
    mean_random = np.random.permutation(means)
    for i in range(n_clusters_to_stim):
        stim = np.zeros(total_steps)
        stim[buffer_period:buffer_period+stim_steps] = np.random.normal(mean_random[i],std, stim_steps)
        stim = np.maximum(stim,0)
        weights = np.ones(N_exc) #Will be masked in get_input_configs(), just need array of ones 
        stim_list.append(stim)
        weight_list.append(weights)
    
    return stim_list, weight_list

def get_input_configs(cluster_list,stim_list,weight_list):
    '''
    Setting up stimuli and weights for each cluster. 
    '''

    if not (len(cluster_list) == len(stim_list) == len(weight_list)):
        raise ValueError("Must provide equal number of cluster list, stimuli, and weights")
    
    # Create configurations
    input_configs = []
    for clusters, stim, weights in zip(cluster_list, stim_list, weight_list):
        config = {
            'clusters': clusters,
            'stim': stim,
            'weights': weights
        }
        input_configs.append(config)
    
    return input_configs

def create_input_operation(E_pop, input_configs, membership=None):
    """
    Creates a (vectorised version) network operation to handle inputs with flexible clustering. Not usable outside of simulation.
    
    Parameters
    ----------
    E_pop : brian2.NeuronGroup
        The neuron population
    input_configs : list of dict
        Input configurations
    membership : list of lists, optional
        Cluster membership (if applicable)
    """

    N_exc = len(E_pop)
    if membership is not None and len(membership) != N_exc:
        raise ValueError(f"Membership array length {len(membership)} doesn't match number of neurons {N_exc}")
    
    n_inputs = len(input_configs)
    max_time_steps = max(len(config['stim']) for config in input_configs)
    
    #Pre-allocate matrices
    stimulus_matrix = np.zeros((n_inputs, max_time_steps))
    effective_weights = np.zeros((n_inputs, N_exc))
    
    for i, config in enumerate(input_configs):
        #Stimulus
        stim_len = len(config['stim'])
        stimulus_matrix[i, :stim_len] = config['stim']
        
        #Combined weights and mask computation
        weights = np.array(config['weights'][:N_exc])
        
        if membership is not None:
            target_clusters = np.array(config['clusters'])
            membership_array = np.array([membership[j] for j in range(N_exc)])
            
            #Check if any target cluster is in each neuron's membership
            mask = np.array([np.any(np.isin(membership_array[j], target_clusters)) 
                           for j in range(N_exc)])
        else:
            mask = np.zeros(N_exc)
            mask[config['clusters']] = 1
        
        effective_weights[i, :] = weights * mask
    
    @b2.network_operation(dt=b2.defaultclock.dt)
    def input_operation(t):
        idx = int(t / b2.defaultclock.dt)
        if idx < max_time_steps:
            total_input = stimulus_matrix[:, idx] @ effective_weights
            E_pop.I_ext_ = total_input * b2.amp
        else:
            E_pop.I_ext_ = np.zeros(N_exc) * b2.amp
    
    return input_operation
    
class NeuralDecoder:
    def __init__(self, method='logistic'):
        """
        Initializes the neural decoder for population firing rates.
        
        Args:
            method (str): Decoding method, either 'logistic' for logistic regression 
                         or 'svm' for support vector machine.
        """
        self.method = method
        self.model = None
        self.scaler = None
        self.cv_accuracy = None

    def train(self, firing_rates, input_strengths, n_splits=5):
        """
        Trains the decoder on multi-neuron firing rates.
        
        Args:
            firing_rates (np.ndarray): Shape (n_trials, n_neurons, n_timepoints) or 
                                      (n_trials, n_neurons), population firing rates.
            input_strengths (np.ndarray): Shape (n_trials,), the input strengths for each trial.
            n_splits (int): Number of folds for cross-validation.
        """

        
        # Check if firing_rates is 3D (including time dimension) or 2D (already averaged)
        if firing_rates.ndim == 3:
            # Already in format (n_trials, n_neurons, n_timepoints)
            # Average across time to get (n_trials, n_neurons)
            X = np.mean(firing_rates, axis=2)  # Shape (n_trials, n_neurons)
        elif firing_rates.ndim == 2:
            # Already in format (n_trials, n_neurons)
            X = firing_rates
        else:
            raise ValueError("firing_rates must be either 2D (n_trials, n_neurons) or 3D (n_trials, n_neurons, n_timepoints).")
        
        if len(input_strengths) != X.shape[0]:
            raise ValueError("input_strengths must have length n_trials.")

        # Create labels
        half_max = 0.5 * input_strengths.max()
        y = (input_strengths > half_max).astype(int)

        # Cross-validated training
        gkf = GroupKFold(n_splits=n_splits)
        accuracies = []

        for train_idx, test_idx in gkf.split(X, y, groups=np.arange(len(y))):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            if self.method == 'logistic':
                self.model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
            elif self.method == 'svm':
                self.model = SVC(kernel='rbf', C=1.0, probability=True)
            else:
                raise ValueError("Method must be either 'logistic' or 'svm'.")
                
            self.model.fit(X_train_scaled, y_train)

            # Evaluate on test set
            y_pred = self.model.predict(X_test_scaled)
            accuracies.append(accuracy_score(y_test, y_pred))

        self.cv_accuracy = np.mean(accuracies)
        print(f"Cross-validated accuracy using {self.method}: {self.cv_accuracy:.2f}")

    def predict(self, data, use_sliding_window=False, window_size=10):
        """
        Applies the trained model to new data.
        
        Args:
            data (np.ndarray): Shape (n_neurons, n_timepoints) for a single trial time series,
                              (n_trials, n_neurons) for pre-averaged data, or
                              (n_trials, n_neurons, n_timepoints) for multiple full trials.
            use_sliding_window (bool): If True, uses a sliding window for prediction.
            window_size (int): Size of the sliding window in timepoints.
        
        Returns:
            predictions: Binary predictions.
            confidence: Confidence scores (probability of class 1).
        """
        import numpy as np
        
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call `train` first.")

        if use_sliding_window:
            # Check if data is single trial time series
            if data.ndim == 2 and data.shape[0] < data.shape[1]:  # Likely (n_neurons, n_timepoints)
                # For a single trial with sliding window
                n_neurons, n_timepoints = data.shape
                
                if n_timepoints < window_size:
                    raise ValueError("Time series is shorter than the window size.")
                
                predictions = []
                confidence = []
                
                for i in range(0, n_timepoints - window_size + 1):
                    window = data[:, i:i + window_size]
                    # Average across time points in the window for each neuron
                    features = np.mean(window, axis=1)  # Shape (n_neurons,)
                    features_scaled = self.scaler.transform([features])
                    pred = self.model.predict(features_scaled)[0]
                    
                    if self.method == 'logistic':
                        conf = self.model.predict_proba(features_scaled)[0][1]
                    else:  # SVM
                        conf = self.model.predict_proba(features_scaled)[0][1]
                    
                    predictions.append(pred)
                    confidence.append(conf)
                
                return np.array(predictions), np.array(confidence)
            else:
                raise ValueError("For sliding window, data should be 2D with shape (n_neurons, n_timepoints).")
        else:
            # Handle different input shapes for non-sliding window predictions
            if data.ndim == 3:  # Shape (n_trials, n_neurons, n_timepoints)
                # Average across time to get (n_trials, n_neurons)
                X = np.mean(data, axis=2)
            elif data.ndim == 2:
                if data.shape[0] < data.shape[1]:  # Likely (n_neurons, n_timepoints) for a single trial
                    # Average across time, reshape to (1, n_neurons)
                    X = np.mean(data, axis=1).reshape(1, -1)
                else:  # Shape (n_trials, n_neurons)
                    X = data
            else:
                raise ValueError("Unsupported data shape. Expected (n_neurons, n_timepoints), (n_trials, n_neurons), or (n_trials, n_neurons, n_timepoints).")
            
            # Predict
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            if self.method == 'logistic':
                confidence = self.model.predict_proba(X_scaled)[:, 1]
            else:  # SVM
                confidence = self.model.predict_proba(X_scaled)[:, 1]
            
            return predictions, confidence
        
def plot_multiple_decoders(decoder_class, data_sets, input_strengths, test_data, nrows=10, ncols=5, 
                          method='logistic', window_size=10, figsize=(20, 20)):

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(f"Neural Decoder Predictions for {len(data_sets)} Sets", fontsize=16)
    

    axes = axes.ravel()
    
    for i in range(len(data_sets)):
        # Initialize and train the decoder on single input 
        decoder = decoder_class(method=method)
        decoder.train(data_sets[i], input_strengths)
        
        # Test on continuous pulse
        predictions, confidence = decoder.predict(test_data[i], use_sliding_window=True, window_size=window_size)
        
        ax = axes[i]
        ax.plot(confidence, label='Confidence', color='blue', alpha=0.7)
        ax.plot(predictions, label='Prediction', color='red', linestyle='--', alpha=0.7)
        if i == 0:
            ax.legend()
            ax.set_xlabel('Time (sliding window position)')
            ax.set_ylabel('Prediction')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            
        ax.set_title(f"Set {i+1}")
    for i in range(len(data_sets), nrows * ncols):
        axes[i].set_visible(False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

'''
#Not used anymore but you might find useful
def DM_cluster(all_param_dict, mu = 5*10**-9, sigma=2.5*10**-11):
    """
    DM task mimicking Wang 2002 https://www.cell.com/neuron/fulltext/S0896-6273(02)01092-9
    
    Returns:
        - stimulus_A: Stimulus for population A 
        - stimulus_B: Stimulus for population B 
    """

    #Dicts
    param_dict_settings = all_param_dict["params_settings"]
    param_dict_net = all_param_dict["params_net"]

    #Set random seeds
    b2.seed(param_dict_settings["random_seed"])
    b2.defaultclock.dt = param_dict_settings["dt"]

    #Setting stimulation and simulation length 
    stim_steps = int(5 * b2.second / b2.defaultclock.dt) #Potentially change stim length to user added input 
    one_second = int(b2.second / b2.defaultclock.dt)
    buffer_period = 20 * one_second #Allow network to stabilise first
    total_steps = int(param_dict_settings['sim_time']/b2.defaultclock.dt)
    if stim_steps > total_steps:
        raise ValueError("Stimulus steps exceed total simulation steps.")
    
    #Check number of clusters in the network 
    if (
        ("n_clusters" not in param_dict_net.keys())
        or (param_dict_net["n_clusters"] < 2)
    ):
        stim_list = np.random.normal(mu,sigma,stim_steps) #One stimulus for everything 
    else:
        stim_list = []
        means = np.linspace(10**-10,2.5*10**-9,25)  #Meaningful input probably lies within 10^-10 to 10^-9
        mean_to_be_used = np.random.choice(means,size=int(param_dict_net["n_clusters"]),replace=False)
        for i in range(int(param_dict_net["n_clusters"])):
            stim = np.zeros(total_steps)
            stim[buffer_period:buffer_period+stim_steps] = np.random.normal(mean_to_be_used[i],0.01 * mean_to_be_used[i], stim_steps)
            stim = np.maximum(stim,0)
            stim_list.append(stim)

    return stim_list

def input_scaling(all_param_dict,scaling):
    param_dict_settings = all_param_dict["params_settings"]
    param_dict_net = all_param_dict["params_net"]
    #Set random seeds
    b2.seed(param_dict_settings["random_seed"])
    b2.defaultclock.dt = param_dict_settings["dt"]

    #N_neurons
    n_neurons, exc_prop = param_dict_net["N_pop"], param_dict_net["exc_prop"]
    N_exc = int(n_neurons * exc_prop) 

    #Type of scaling 
    if scaling == 'all':
        input_scaling = np.ones(n_neurons).astype(int)
    elif scaling == 'rand':
        input_scaling = np.random.rand(n_neurons)
    elif scaling == 'subset':
        input_scaling = np.random.randint(0,2,n_neurons)
    elif scaling == 'e_only':
        input_scaling = np.zeros(n_neurons).astype(int)
        input_scaling[:N_exc] = 1
    elif scaling == 'i_only':
        input_scaling = np.zeros(n_neurons).astype(int)
        input_scaling[N_exc:] = 1
    else:
        raise ValueError(f"Invalid input for 'scaling': {scaling}. Choose from 'all', 'rand', 'subset', 'e_only' or 'i_only'.")
    return input_scaling
'''