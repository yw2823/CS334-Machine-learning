import numpy as np

class MarkovModel:
    """
    A Hidden Markov Model (HMM) implementation for sequence analysis.
    """
    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, 
                 prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """
        Initializes the HMM with given parameters.

        Args:
            observation_states (np.ndarray): Observed states.
            hidden_states (np.ndarray): Hidden states.
            prior_p (np.ndarray): Prior probabilities of hidden states.
            transition_p (np.ndarray): Transition probabilities between hidden states.
            emission_p (np.ndarray): Emission probabilities from hidden to observed states.
        """
        self.observation_states = observation_states
        self.observation_states_dict = {state: idx for idx, state in enumerate(observation_states)}
        
        self.hidden_states = hidden_states
        self.hidden_states_dict = {idx: state for idx, state in enumerate(hidden_states)}
        
        self.prior_p = np.log(prior_p)  # Log probabilities for numerical stability
        self.transition_p = np.log(transition_p)
        self.emission_p = np.log(emission_p)

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        Computes the log-likelihood of an observed sequence using the Forward algorithm.

        Args:
            input_observation_states (np.ndarray): Sequence of observed states.
        
        Returns:
            float: Log-likelihood of the observed sequence.
        """
        n_hidden = len(self.hidden_states)
        n_obs = len(input_observation_states)
        forward_p = np.full((n_hidden, n_obs), -np.inf)  # Log probabilities initialization
        obs_indices = [self.observation_states_dict[obs] for obs in input_observation_states]
        
        # Initialization step
        forward_p[:, 0] = self.prior_p + self.emission_p[:, obs_indices[0]]
        
        # Recursion step
        for t in range(1, n_obs):
            for j in range(n_hidden):
                forward_p[j, t] = np.logaddexp.reduce(forward_p[:, t - 1] + self.transition_p[:, j]) + self.emission_p[j, obs_indices[t]]
        
        # Termination step
        return np.logaddexp.reduce(forward_p[:, -1])

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        Finds the most likely sequence of hidden states using the Viterbi algorithm.

        Args:
            decode_observation_states (np.ndarray): Sequence of observed states.
        
        Returns:
            list: Most likely sequence of hidden states.
        """
        n_hidden = len(self.hidden_states)
        n_obs = len(decode_observation_states)
        viterbi_table = np.full((n_hidden, n_obs), -np.inf)
        backtrace = np.zeros((n_hidden, n_obs), dtype=int)
        obs_indices = [self.observation_states_dict[obs] for obs in decode_observation_states]
        
        # Initialization step
        viterbi_table[:, 0] = self.prior_p + self.emission_p[:, obs_indices[0]]
        
        # Recursion step
        for t in range(1, n_obs):
            for j in range(n_hidden):
                trans_probs = viterbi_table[:, t - 1] + self.transition_p[:, j]
                max_prob_idx = np.argmax(trans_probs)
                viterbi_table[j, t] = trans_probs[max_prob_idx] + self.emission_p[j, obs_indices[t]]
                backtrace[j, t] = max_prob_idx
        
        
        best_path_idx = np.argmax(viterbi_table[:, -1])
        best_path = [best_path_idx]
        for t in range(n_obs - 1, 0, -1):
            best_path_idx = backtrace[best_path_idx, t]
            best_path.insert(0, best_path_idx)
        
        return [self.hidden_states[idx] for idx in best_path]
