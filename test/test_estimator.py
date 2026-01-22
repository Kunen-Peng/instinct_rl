
import torch
import torch.nn as nn
from instinct_rl.modules.history_cat_estimator import HistoryCatEstimator

class DummyEstimator(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.output_size = output_dim # for property check
    
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.ones(batch_size, self.output_dim)

def test_history_cat_estimator():
    batch_size = 2
    temporal_steps = 3
    num_one_step_obs = 4
    estimator_output_dim = 5
    
    # Create dummy estimator
    base = DummyEstimator(estimator_output_dim)
    
    # Create wrapper
    wrapper = HistoryCatEstimator(
        base_estimator=base,
        temporal_steps=temporal_steps,
        num_one_step_obs=num_one_step_obs,
        history_format="oldest_first"
    )
    
    print(f"Wrapper output size: {wrapper.output_size}")
    
    # Create dummy history Input: [B, T * D]
    # Let's make it distinct to verify correctness
    # Batch 0: 
    #   t=0: [0,0,0,0]
    #   t=1: [1,1,1,1]
    #   t=2: [2,2,2,2] -> Latest
    # Batch 1:
    #   t=0: [10,10,10,10]
    #   t=1: [11,11,11,11]
    #   t=2: [12,12,12,12] -> Latest
    
    hist = torch.zeros(batch_size, temporal_steps, num_one_step_obs)
    for t in range(temporal_steps):
        hist[0, t, :] = t
        hist[1, t, :] = t + 10
        
    hist_flat = hist.reshape(batch_size, -1)
    
    output = wrapper(hist_flat)
    
    print(f"Input shape: {hist_flat.shape}")
    print(f"Output shape: {output.shape}")
    
    expected_dim = estimator_output_dim + num_one_step_obs
    assert output.shape == (batch_size, expected_dim), f"Expected shape {(batch_size, expected_dim)}, got {output.shape}"
    
    # Verify content
    # First part should be ones (from dummy estimator)
    assert torch.all(output[:, :estimator_output_dim] == 1.0), "Base estimator output mismatch"
    
    # Second part should be latest history
    latest_part = output[:, estimator_output_dim:]
    
    print("Latest part (Batch 0):", latest_part[0])
    print("Expected (Batch 0):", hist[0, -1])
    
    assert torch.allclose(latest_part[0], hist[0, -1]), "Latest history mismatch for batch 0"
    assert torch.allclose(latest_part[1], hist[1, -1]), "Latest history mismatch for batch 1"
    
    print("Test Passed for Tensor output!")
    
    # Test tuple output
    class TupleEstimator(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 1), torch.zeros(x.shape[0], 2)
            
    tuple_base = TupleEstimator()
    tuple_wrapper = HistoryCatEstimator(tuple_base, temporal_steps, num_one_step_obs)
    
    out1, out2 = tuple_wrapper(hist_flat)
    print(f"Tuple output shapes: {out1.shape}, {out2.shape}")
    
    assert torch.allclose(out2[:, 2:], hist[:, -1]), "Tuple latest history mismatch"
    print("Test Passed for Tuple output!")

if __name__ == "__main__":
    test_history_cat_estimator()
