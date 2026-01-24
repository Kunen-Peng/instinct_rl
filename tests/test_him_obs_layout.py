"""Tests for HIM observation layout conversion.

Tests the term_major_to_time_major conversion function to ensure
correct handling of mjlab's term-major observation format.
"""

import pytest
import torch

from instinct_rl.modules.him_actor_critic import term_major_to_time_major


class TestTermMajorToTimeMajor:
    """Test term_major_to_time_major conversion function."""
    
    def test_basic_conversion(self):
        """Test basic 2-term, 3-step history conversion."""
        batch_size = 2
        history_length = 3
        term_dims = [2, 3]  # Term A has 2 dims, Term B has 3 dims
        
        # Term-major layout: [A_t0, A_t1, A_t2, B_t0, B_t1, B_t2]
        # A values: 1, 2 (dim) × 3 (time) = [1.0, 1.1, 2.0, 2.1, 3.0, 3.1]
        # B values: 3 (dim) × 3 (time) = [10, 11, 12, 20, 21, 22, 30, 31, 32]
        term_a = torch.tensor([
            [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],  # batch 0: A_t0=[1.0, 1.1], A_t1=[2.0, 2.1], A_t2=[3.0, 3.1]
            [4.0, 4.1, 5.0, 5.1, 6.0, 6.1],  # batch 1
        ])
        term_b = torch.tensor([
            [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0],  # B_t0, B_t1, B_t2
            [40.0, 41.0, 42.0, 50.0, 51.0, 52.0, 60.0, 61.0, 62.0],
        ])
        
        # Combine into term-major format
        obs_term_major = torch.cat([term_a, term_b], dim=-1)
        assert obs_term_major.shape == (batch_size, 15)  # (2+3)*3 = 15
        
        # Convert to time-major
        result = term_major_to_time_major(obs_term_major, term_dims, history_length)
        
        # Expected time-major: [obs_t0, obs_t1, obs_t2]
        # obs_t0 = [A_t0, B_t0] = [1.0, 1.1, 10.0, 11.0, 12.0]
        # obs_t1 = [A_t1, B_t1] = [2.0, 2.1, 20.0, 21.0, 22.0]
        # obs_t2 = [A_t2, B_t2] = [3.0, 3.1, 30.0, 31.0, 32.0]
        expected_batch0 = torch.tensor([
            1.0, 1.1, 10.0, 11.0, 12.0,  # t0
            2.0, 2.1, 20.0, 21.0, 22.0,  # t1
            3.0, 3.1, 30.0, 31.0, 32.0,  # t2
        ])
        
        assert result.shape == (batch_size, 15)
        assert torch.allclose(result[0], expected_batch0)
    
    def test_single_term(self):
        """Test with single observation term (should be identity)."""
        batch_size = 4
        history_length = 5
        term_dims = [10]
        
        obs = torch.randn(batch_size, 50)  # 10 * 5 = 50
        result = term_major_to_time_major(obs, term_dims, history_length)
        
        # With single term, term-major and time-major are identical
        assert torch.allclose(result, obs)
    
    def test_go2_observation_structure(self):
        """Test with Go2 HIM observation structure.
        
        Go2 HIM policy observations:
        - command: 3
        - base_ang_vel: 3
        - projected_gravity: 3
        - joint_pos: 12
        - joint_vel: 12
        - actions: 12
        Total: 45 dims per step, 6 steps = 270 dims
        """
        batch_size = 16
        history_length = 6
        term_dims = [3, 3, 3, 12, 12, 12]
        num_one_step_obs = sum(term_dims)  # 45
        
        # Create term-major observations
        obs_term_major = torch.randn(batch_size, num_one_step_obs * history_length)
        
        # Convert to time-major
        result = term_major_to_time_major(obs_term_major, term_dims, history_length)
        
        # Shape should be preserved
        assert result.shape == obs_term_major.shape
        
        # Verify the last timestep is correct
        # In time-major, last step is the final num_one_step_obs elements
        last_step_time_major = result[:, -num_one_step_obs:]
        
        # In term-major, last step is the last element of each term's history
        # term0[-1], term1[-1], ..., termN[-1]
        last_step_term_major_parts = []
        offset = 0
        for dim in term_dims:
            term_history = obs_term_major[:, offset:offset + dim * history_length]
            term_reshaped = term_history.reshape(batch_size, history_length, dim)
            last_step_term_major_parts.append(term_reshaped[:, -1, :])
            offset += dim * history_length
        last_step_expected = torch.cat(last_step_term_major_parts, dim=-1)
        
        assert torch.allclose(last_step_time_major, last_step_expected)
    
    def test_reversibility(self):
        """Test that time_major -> term_major -> time_major is identity.
        
        This validates the conversion is correct by doing a round-trip.
        """
        batch_size = 8
        history_length = 4
        term_dims = [5, 3, 7]
        num_one_step_obs = sum(term_dims)
        
        # Start with time-major observations
        obs_time_major = torch.randn(batch_size, num_one_step_obs * history_length)
        
        # Convert time_major to term_major (inverse operation)
        # Reshape to [batch, history, obs] then rearrange
        reshaped = obs_time_major.reshape(batch_size, history_length, num_one_step_obs)
        
        # Split by term
        term_parts = torch.split(reshaped, term_dims, dim=-1)
        
        # For each term, flatten time dimension
        term_major_parts = [part.reshape(batch_size, -1) for part in term_parts]
        
        # Concatenate to get term-major
        obs_term_major = torch.cat(term_major_parts, dim=-1)
        
        # Now convert back to time-major
        result = term_major_to_time_major(obs_term_major, term_dims, history_length)
        
        # Should match original
        assert torch.allclose(result, obs_time_major)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
