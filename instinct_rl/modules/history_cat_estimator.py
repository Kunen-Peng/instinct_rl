import torch
import torch.nn as nn

class HistoryCatEstimator(nn.Module):
    """
    Estimator wrapper that concatenates the latest history step to the base estimator's output.
    
    This is useful when the downstream consumer (e.g. policy) needs direct access to the 
    most recent observation in addition to the estimated features.
    """
    def __init__(
        self,
        base_estimator: nn.Module,
        temporal_steps: int,
        num_one_step_obs: int,
        history_format: str = "oldest_first"
    ):
        """
        Args:
            base_estimator: The base estimator module.
            temporal_steps: Number of temporal steps in the history.
            num_one_step_obs: Dimension of a single observation step.
            history_format: "oldest_first" or "newest_first".
                           "oldest_first": [t-H, ..., t-1, t] (default)
                           "newest_first": [t, t-1, ..., t-H]
        """
        super().__init__()
        self.base_estimator = base_estimator
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.history_format = history_format
        
        if history_format not in ["oldest_first", "newest_first"]:
            raise ValueError(f"history_format must be 'oldest_first' or 'newest_first', got {history_format}")

    def forward(self, obs_history):
        """
        Args:
            obs_history: Flattened observation history [batch_size, temporal_steps * num_one_step_obs]
            
        Returns:
            Concatenated tensor [base_output, latest_obs]
        """
        # Compute base estimator output
        base_feat = self.base_estimator(obs_history)
        
        # If base estimator returns a tuple (e.g. vel, latent), we might need to handle that.
        # Assuming for now it returns a single tensor or we modify the last element if it's a tuple.
        # But per the request "cat to estimator's input" (interpreted as output of this block), 
        # let's assume we maintain the tuple structure if present, or just cat if tensor.
        
        # However, to be safe and generic, let's look at how we get the latest obs.
        
        batch_size = obs_history.shape[0]
        obs_reshaped = obs_history.reshape(batch_size, self.temporal_steps, self.num_one_step_obs)
        
        if self.history_format == "oldest_first":
            # Latest is the last element
            latest_obs = obs_reshaped[:, -1, :]
        else: # newest_first
            # Latest is the first element
            latest_obs = obs_reshaped[:, 0, :]
            
        if isinstance(base_feat, tuple):
            # If tuple, we assume the intention is to append to the LAST element of the tuple, 
            # or maybe the user wants a flat tensor? 
            # "cat existing step info ... to estimator's input" -> likely extending the feature vector.
            # If it returns (vel, z), maybe return (vel, cat(z, latest_obs))?
            # Let's handle the simplest case: base_feat is a tensor.
            # If it's a tuple, we might break things. 
            # Let's check HIMEstimator. It returns (vel, z).
            # If we wrap HIMEstimator, we probably want (vel, cat(z, latest_obs)).
            
            *others, last_feat = base_feat
            new_last_feat = torch.cat([last_feat, latest_obs], dim=-1)
            return (*others, new_last_feat)
        else:
            return torch.cat([base_feat, latest_obs], dim=-1)

    @property
    def output_size(self):
        # Allow checking output size if the base estimator has it
        if hasattr(self.base_estimator, "output_size"):
             return self.base_estimator.output_size + self.num_one_step_obs
        return None
