"""Tests for multi-timescale memory."""

import torch
import pytest
from nnfs.memory.multiscale import FastSlowMemory, MultiScaleMemory, MemoryCell


class TestFastSlowMemory:
    def test_forward_shape(self):
        mem = FastSlowMemory(10, 32, 16)
        x = torch.randn(4, 10)
        out = mem(x)
        assert out.shape == (4, 32)  # Output is fast_dim
    
    def test_sequence_forward(self):
        mem = FastSlowMemory(10, 32, 16)
        x = torch.randn(4, 5, 10)  # batch, seq, input
        out = mem(x)
        assert out.shape == (4, 5, 32)
    
    def test_state_update(self):
        mem = FastSlowMemory(10, 32, 16, fast_update_rate=0.5, slow_update_rate=0.1)
        x = torch.randn(2, 10)
        
        # First step
        out1 = mem(x)
        fast1, slow1 = mem.get_states()
        
        # Second step with same input
        out2 = mem(x)
        fast2, slow2 = mem.get_states()
        
        # States should change
        assert not torch.allclose(fast1, fast2)
        assert not torch.allclose(slow1, slow2)
    
    def test_reset_state(self):
        mem = FastSlowMemory(10, 32, 16)
        x = torch.randn(2, 10)
        _ = mem(x)
        
        # States should be non-zero
        fast, slow = mem.get_states()
        assert fast.abs().sum() > 0
        assert slow.abs().sum() > 0
        
        # Reset
        mem.reset_state(batch_size=2)
        fast, slow = mem.get_states()
        assert fast.abs().sum() == 0
        assert slow.abs().sum() == 0
    
    def test_different_batch_size(self):
        mem = FastSlowMemory(10, 32, 16)
        x1 = torch.randn(2, 10)
        _ = mem(x1)
        
        x2 = torch.randn(5, 10)
        out = mem(x2)  # Should handle batch size change
        assert out.shape == (5, 32)


class TestMultiScaleMemory:
    def test_forward_shape(self):
        mem = MultiScaleMemory(10, 32, num_scales=4)
        x = torch.randn(4, 10)
        out = mem(x)
        assert out.shape == (4, 32)
    
    def test_sequence_forward(self):
        mem = MultiScaleMemory(10, 32, num_scales=4)
        x = torch.randn(4, 5, 10)
        out = mem(x)
        assert out.shape == (4, 5, 32)
    
    def test_return_all_states(self):
        mem = MultiScaleMemory(10, 32, num_scales=4)
        x = torch.randn(4, 10)
        out, states = mem(x, return_all_states=True)
        assert out.shape == (4, 32)
        assert states.shape == (4, 4, 32)  # num_scales, batch, hidden
    
    def test_scale_rates(self):
        mem = MultiScaleMemory(10, 32, num_scales=4, base_update_rate=0.3, rate_decay=0.5)
        # Rates should be: 0.3, 0.15, 0.075, 0.0375
        expected = [0.3, 0.15, 0.075, 0.0375]
        assert mem.update_rates == pytest.approx(expected, rel=1e-6)
    
    def test_cross_scale_attention(self):
        mem = MultiScaleMemory(10, 32, num_scales=4, cross_scale_attention=True)
        x = torch.randn(4, 10)
        out = mem(x)
        assert out.shape == (4, 32)
    
    def test_no_cross_scale_attention(self):
        mem = MultiScaleMemory(10, 32, num_scales=4, cross_scale_attention=False)
        x = torch.randn(4, 10)
        out = mem(x)
        assert out.shape == (4, 32)


class TestMemoryCell:
    def test_forward(self):
        cell = MemoryCell(10, 16, update_rate=0.1)
        x = torch.randn(4, 10)
        h = torch.randn(4, 16)
        h_new = cell(x, h)
        assert h_new.shape == (4, 16)
    
    def test_gate_range(self):
        cell = MemoryCell(10, 16, update_rate=0.1)
        x = torch.randn(100, 10)
        h = torch.randn(100, 16)
        h_new = cell(x, h)
        
        # Change should be bounded by update_rate
        change = (h_new - h).abs().mean()
        # With update_rate=0.1, max change per dimension is ~0.1 * max(candidate)
        assert change < 1.0  # Reasonable bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])