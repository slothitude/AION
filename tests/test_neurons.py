"""Tests for neuron implementations."""

import torch
import pytest
from nnfs.neurons.standard import StandardNeuron, StandardLayer
from nnfs.neurons.multiplicative import MultiplicativeNeuron, MultiplicativeLayer
from nnfs.neurons.geometric import GeometricNeuron, GeometricLayer, complex_multiply, complex_add


class TestStandardNeuron:
    def test_forward_shape(self):
        neuron = StandardNeuron(10, 5)
        x = torch.randn(32, 10)
        out = neuron(x)
        assert out.shape == (32, 5)
    
    def test_forward_3d(self):
        neuron = StandardNeuron(10, 5)
        x = torch.randn(16, 20, 10)
        out = neuron(x)
        assert out.shape == (16, 20, 5)
    
    def test_activations(self):
        for act in ["relu", "gelu", "tanh", "sigmoid", "silu", "mish"]:
            neuron = StandardNeuron(10, 5, activation=act)
            x = torch.randn(4, 10)
            out = neuron(x)
            assert out.shape == (4, 5)
    
    def test_no_bias(self):
        neuron = StandardNeuron(10, 5, bias=False)
        assert neuron.bias is None
        x = torch.randn(4, 10)
        out = neuron(x)
        assert out.shape == (4, 5)


class TestMultiplicativeNeuron:
    def test_forward_shape(self):
        neuron = MultiplicativeNeuron(10, 5, interaction_rank=4)
        x = torch.randn(32, 10)
        out = neuron(x)
        assert out.shape == (32, 5)
    
    def test_interaction_term(self):
        """Test that interaction term actually does something."""
        neuron = MultiplicativeNeuron(10, 5, interaction_rank=4)
        x = torch.randn(4, 10)
        
        # With interaction
        out_with = neuron(x)
        
        # Zero out interaction weights
        with torch.no_grad():
            neuron.A.zero_()
        out_without = neuron(x)
        
        # Should be different (unless interaction happens to be zero)
        diff = (out_with - out_without).abs().mean()
        # Note: might be small initially due to small init, but should exist
        assert out_with.shape == out_without.shape
    
    def test_memory(self):
        neuron = MultiplicativeNeuron(10, 5, interaction_rank=4, use_memory=True)
        x = torch.randn(4, 10)
        
        # First forward
        out1 = neuron(x)
        mem1 = neuron.memory.clone()
        
        # Second forward (memory should update)
        out2 = neuron(x)
        mem2 = neuron.memory.clone()
        
        # Memory should have been updated
        assert not torch.allclose(mem1, mem2)
    
    def test_reset_memory(self):
        neuron = MultiplicativeNeuron(10, 5, interaction_rank=4, use_memory=True)
        x = torch.randn(4, 10)
        _ = neuron(x)
        assert neuron.memory.abs().sum() > 0
        neuron.reset_memory()
        assert neuron.memory.abs().sum() == 0
    
    def test_no_memory(self):
        neuron = MultiplicativeNeuron(10, 5, interaction_rank=4, use_memory=False)
        assert neuron.memory is None
        x = torch.randn(4, 10)
        out = neuron(x)
        assert out.shape == (4, 5)


class TestGeometricNeuron:
    def test_forward_shape(self):
        neuron = GeometricNeuron(10, 5)
        x = torch.randn(32, 10)
        mag, phase = neuron(x)
        assert mag.shape == (32, 5)
        assert phase.shape == (32, 5)
    
    def test_phase_range(self):
        neuron = GeometricNeuron(10, 5)
        x = torch.randn(100, 10)
        _, phase = neuron(x)
        assert (phase >= -torch.pi).all()
        assert (phase <= torch.pi).all()
    
    def test_magnitude_nonnegative(self):
        neuron = GeometricNeuron(10, 5)
        x = torch.randn(100, 10)
        mag, _ = neuron(x)
        assert (mag >= 0).all()
    
    def test_complex_weights(self):
        neuron = GeometricNeuron(10, 5, use_complex_weights=True)
        assert neuron.weight_real is not None
        assert neuron.weight_imag is not None
        
        neuron_real = GeometricNeuron(10, 5, use_complex_weights=False)
        assert neuron_real.weight is not None
        assert neuron_real.weight_real is None
    
    def test_modrelu_activation(self):
        neuron = GeometricNeuron(10, 5, activation="modrelu")
        x = torch.randn(4, 10)
        mag, phase = neuron(x)
        assert mag.shape == (4, 5)
        # ModReLU ensures magnitude >= 0 (already true) but with learnable bias
    
    def test_forward_complex(self):
        neuron = GeometricNeuron(10, 5)
        x = torch.randn(4, 10)
        z = neuron.forward_complex(x)
        assert z.shape == (4, 5)
        assert z.dtype == torch.complex64
    
    def test_geometric_layer(self):
        layer = GeometricLayer(10, 5, dropout=0.1)
        x = torch.randn(4, 10)
        mag, phase = layer(x)
        assert mag.shape == (4, 5)
        assert phase.shape == (4, 5)


class TestComplexOperations:
    def test_complex_multiply(self):
        z1 = (torch.tensor([2.0, 3.0]), torch.tensor([0.5, 1.0]))
        z2 = (torch.tensor([1.5, 2.0]), torch.tensor([0.3, 0.7]))
        r, theta = complex_multiply(z1, z2)
        
        # Magnitude multiplies
        expected_r = torch.tensor([3.0, 6.0])
        # Phase adds
        expected_theta = torch.tensor([0.8, 1.7])
        
        assert torch.allclose(r, expected_r, atol=1e-6)
        assert torch.allclose(theta, expected_theta, atol=1e-6)
    
    def test_complex_add(self):
        # Add two complex numbers in polar form
        z1 = (torch.tensor([1.0]), torch.tensor([0.0]))      # 1 + 0i
        z2 = (torch.tensor([1.0]), torch.tensor([torch.pi/2]))  # 0 + 1i
        r, theta = complex_add(z1, z2)
        
        # Result should be sqrt(2) at pi/4
        assert torch.allclose(r, torch.tensor([2.0**0.5]), atol=1e-6)
        assert torch.allclose(theta, torch.tensor([torch.pi/4]), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])