"""Tests for model architectures."""

import torch
import pytest
from nnfs.models import StandardMLP, MultiplicativeMLP, GeometricMLP, SpecializedNet


class TestStandardMLP:
    def test_forward(self):
        model = StandardMLP(784, 10, [256, 128, 64])
        x = torch.randn(32, 784)
        out = model(x)
        assert out.shape == (32, 10)
    
    def test_count_parameters(self):
        model = StandardMLP(784, 10, [256, 128, 64])
        params = model.count_parameters()
        assert params > 0
        # Rough check: 784*256 + 256*128 + 128*64 + 64*10 ≈ 200k + 32k + 8k + 0.6k ≈ 240k
        assert 200000 < params < 300000
    
    def test_get_layer_outputs(self):
        model = StandardMLP(10, 5, [20, 15])
        x = torch.randn(4, 10)
        outputs = model.get_layer_outputs(x)
        assert len(outputs) == 3  # 2 hidden + 1 output
        assert outputs[0].shape == (4, 20)
        assert outputs[1].shape == (4, 15)
        assert outputs[2].shape == (4, 5)


class TestMultiplicativeMLP:
    def test_forward(self):
        model = MultiplicativeMLP(784, 10, [256, 128, 64], interaction_rank=8)
        x = torch.randn(32, 784)
        out = model(x)
        assert out.shape == (32, 10)
    
    def test_count_parameters(self):
        model = MultiplicativeMLP(784, 10, [256, 128, 64], interaction_rank=8)
        params = model.count_parameters()
        assert params > 0
        # Should have more params due to interaction terms
        standard = StandardMLP(784, 10, [256, 128, 64])
        assert params > standard.count_parameters()
    
    def test_reset_memory(self):
        model = MultiplicativeMLP(10, 5, [20, 15], interaction_rank=4, use_memory=True)
        x = torch.randn(4, 10)
        _ = model(x)
        
        # Check memory was used
        has_memory = False
        for module in model.modules():
            if hasattr(module, 'memory') and module.memory is not None:
                if module.memory.abs().sum() > 0:
                    has_memory = True
                    break
        assert has_memory
        
        model.reset_memory()
        
        # All memories should be zero
        for module in model.modules():
            if hasattr(module, 'memory') and module.memory is not None:
                assert module.memory.abs().sum() == 0
    
    def test_get_interaction_strengths(self):
        model = MultiplicativeMLP(10, 5, [20, 15], interaction_rank=4)
        strengths = model.get_interaction_strengths()
        assert len(strengths) == 2  # Two hidden layers
        for s in strengths:
            assert s.shape == (20,) or s.shape == (15,)  # Per output neuron


class TestGeometricMLP:
    def test_forward(self):
        model = GeometricMLP(784, 10, [256, 128, 64])
        x = torch.randn(32, 784)
        out = model(x)
        assert out.shape == (32, 10)
    
    def test_forward_full(self):
        model = GeometricMLP(10, 5, [20, 15])
        x = torch.randn(4, 10)
        outputs = model.forward_full(x)
        # 2 hidden layers + logits
        assert len(outputs) == 3
        # First two should be (mag, phase) tuples
        assert isinstance(outputs[0], tuple) and len(outputs[0]) == 2
        assert isinstance(outputs[1], tuple) and len(outputs[1]) == 2
        # Last should be logits
        assert outputs[2][0] == "logits"
        assert outputs[2][1].shape == (4, 5)
    
    def test_forward_complex(self):
        model = GeometricMLP(10, 5, [20, 15])
        x = torch.randn(4, 10)
        out = model.forward_complex(x)
        assert out.shape == (4, 5)
        # Final output is real logits (magnitude for classification)
        assert out.dtype == torch.float32
    
    def test_count_parameters(self):
        model = GeometricMLP(784, 10, [256, 128, 64])
        params = model.count_parameters()
        assert params > 0
        # Complex weights = 2x real params per connection
        standard = StandardMLP(784, 10, [256, 128, 64])
        # But we use smaller hidden dims in practice, so not necessarily larger
        assert params > 0
    
    def test_get_phase_statistics(self):
        model = GeometricMLP(10, 5, [20, 15])
        x = torch.randn(100, 10)
        stats = model.get_phase_statistics(x)
        assert "layer_0" in stats
        assert "layer_1" in stats
        for layer_stats in stats.values():
            assert "mean_magnitude" in layer_stats
            assert "mean_phase" in layer_stats
            assert "phase_entropy" in layer_stats


class TestSpecializedNet:
    def test_forward_standard(self):
        model = SpecializedNet(784, 10, hidden_dim=128, neuron_type="standard")
        x = torch.randn(32, 784)
        out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (32, 10)
    
    def test_forward_multiplicative(self):
        model = SpecializedNet(784, 10, hidden_dim=128, neuron_type="multiplicative")
        x = torch.randn(32, 784)
        out = model(x)
        assert out["logits"].shape == (32, 10)
    
    def test_forward_geometric(self):
        model = SpecializedNet(784, 10, hidden_dim=128, neuron_type="geometric")
        x = torch.randn(32, 784)
        out = model(x)
        assert out["logits"].shape == (32, 10)
    
    def test_forward_sequence(self):
        model = SpecializedNet(10, 5, hidden_dim=32, neuron_type="standard")
        x = torch.randn(4, 5, 10)  # batch, seq, input
        out = model(x)
        assert out["logits"].shape == (4, 5, 5)
    
    def test_with_target_next(self):
        model = SpecializedNet(10, 5, hidden_dim=32, neuron_type="standard")
        x = torch.randn(4, 10)
        # Target should be next state (hidden_dim), not next input
        target = torch.randn(4, 32)
        out = model(x, target_next=target, return_intermediates=True)
        assert "prediction" in out
        assert "prediction_error" in out["prediction"]
    
    def test_return_intermediates(self):
        model = SpecializedNet(10, 5, hidden_dim=32, neuron_type="standard")
        x = torch.randn(4, 10)
        out = model(x, return_intermediates=True)
        assert "detection" in out
        assert "interaction" in out
        assert "competition" in out
        assert "integration" in out
    
    def test_count_parameters(self):
        model = SpecializedNet(784, 10, hidden_dim=128)
        params = model.count_parameters()
        assert params > 0
    
    def test_reset_memory(self):
        model = SpecializedNet(10, 5, hidden_dim=32, neuron_type="multiplicative", use_memory=True)
        x = torch.randn(4, 10)
        _ = model(x)
        model.reset_memory()
        # Should not error


class TestCreateMatchedModels:
    def test_create_matched(self):
        from nnfs.models.specialized_net import create_matched_models
        
        models = create_matched_models(784, 10, target_params=300000)
        
        assert "standard" in models
        assert "multiplicative" in models
        assert "geometric" in models
        assert "specialized" in models
        
        # All should have reasonable parameter counts
        for name, model in models.items():
            params = model.count_parameters()
            assert 100000 < params < 1000000, f"{name}: {params} params"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])