
import torch
import torch.nn as nn
from instinct_rl.modules.cenet import CENet

def test_legacy_cenet():
    print("Testing Legacy CENet (No RNN)...")
    cenet = CENet(num_encoder_obs=64, num_decoder_obs=48)
    assert cenet.rnn is None
    
    obs = torch.randn(10, 64)
    # Forward pass (encode)
    out = cenet.encode(obs)
    assert out.shape == (10, 19) # 3 + 16
    print("Legacy Encode OK")

def test_recurrent_cenet():
    print("Testing Recurrent CENet (GRU)...")
    cenet = CENet(num_encoder_obs=64, num_decoder_obs=48, rnn_type="gru", rnn_hidden_size=128, rnn_num_layers=1)
    assert cenet.rnn is not None
    assert cenet.encoder.model[0].in_features == 128
    
    batch_size = 5
    obs = torch.randn(batch_size, 64)
    hidden_states = torch.zeros(1, batch_size, 128)
    
    # Forward pass (encode) with hidden states
    # Note: encode logic modified to call self.rnn if exists
    # And if hidden_states passed, it uses them.
    
    # 1. Test without passing hidden states (Inference Mode logic inside Memory)
    out_inf = cenet.encode(obs)
    assert out_inf.shape == (batch_size, 19)
    print("Recurrent Encode (Inference Mode / No Hidden) OK")
    
    # 2. Test with passing hidden states (Batch Mode / Explicit)
    # WARNING: My implementation of encode uses self.rnn(obs, hidden_states=hs)
    # Memory module returns (out, state) only if NOT batch mode?
    # Let's check Memory again. 
    # forward(input, hidden_states):
    #   if hidden_states is not None: return rnn(input, hidden_states) -> returns (out, hidden)
    # WAIT! Standard GRU returns (out, hidden). 
    # My previous concern was about Memory wrapper.
    # If Memory wrapper just calls self.rnn, it returns (out, hidden).
    # so `rnn_out` in `encode` would be a TUPLE if I'm not careful.
    
    # Let's verify what `Memory.forward` returns.
    # "out, _ = self.rnn(input, hidden_states)" -> It explicitly discards the hidden state returning `_`! 
    # SO IT RETURNS `out` only.
    # This confirms my implementation in `cenet.encode` is safe (it expects `rnn_out` to be tensor).
    
    out_batch = cenet.encode(obs, hidden_states=hidden_states)
    assert isinstance(out_batch, torch.Tensor)
    assert out_batch.shape == (batch_size, 19)
    print("Recurrent Encode (Batch Mode / With Hidden) OK")
    
    # 3. Test explicit inference recurrent
    v_mean, next_hidden = cenet.encoder_inference_recurrent(obs, hidden_states)
    assert v_mean.shape == (batch_size, 19)
    assert next_hidden.shape == (1, batch_size, 128)
    print("Recurrent Inference Explicit OK")

if __name__ == "__main__":
    test_legacy_cenet()
    test_recurrent_cenet()
    print("All tests passed!")
