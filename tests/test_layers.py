"""Tests for specialized layers."""

import torch
import pytest
from nnfs.layers.detection import DetectionLayer
from nnfs.layers.interaction import InteractionLayer, BilinearInteraction, GeometricInteraction
from nnfs.layers.competition import CompetitionLayer, GeometricCompetitionLayer
from nnfs.layers.integration import IntegrationLayer
from nnfs.layers.prediction import PredictionLayer


class TestDetectionLayer:
    def test_standard_forward(self):
        layer = DetectionLayer(10, 20, neuron_type="standard")
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)
    
    def test_multiplicative_forward(self):
        layer = DetectionLayer(10, 20, neuron_type="multiplicative", interaction_rank=4)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)
    
    def test_geometric_forward(self):
        layer = DetectionLayer(10, 20, neuron_type="geometric")
        x = torch.randn(4, 10)
        mag, phase = layer(x)
        assert mag.shape == (4, 20)
        assert phase.shape == (4, 20)


class TestInteractionLayer:
    def test_standard_forward(self):
        layer = InteractionLayer(20, 30, neuron_type="standard")
        x = torch.randn(4, 20)
        out = layer(x)
        assert out.shape == (4, 30)
    
    def test_multiplicative_forward(self):
        layer = InteractionLayer(20, 30, neuron_type="multiplicative", interaction_rank=8)
        x = torch.randn(4, 20)
        out = layer(x)
        assert out.shape == (4, 30)
    
    def test_geometric_forward(self):
        layer = InteractionLayer(20, 30, neuron_type="geometric")
        x = torch.randn(4, 20)
        mag, phase = layer(x)
        assert mag.shape == (4, 30)
        assert phase.shape == (4, 30)
    
    def test_bilinear_interaction(self):
        layer = BilinearInteraction(10, 10, 5, rank=4)
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)
        out = layer(x, y)
        assert out.shape == (4, 5)
    
    def test_geometric_interaction(self):
        layer = GeometricInteraction(10, num_relations=4)
        mag = torch.rand(4, 10)
        phase = torch.rand(4, 10) * 2 * torch.pi - torch.pi
        out_mag, out_phase = layer(mag, phase)
        assert out_mag.shape == (4, 10)
        assert out_phase.shape == (4, 10)


class TestCompetitionLayer:
    def test_softmax_competition(self):
        layer = CompetitionLayer(16, group_size=4, competition_mode="softmax")
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 16)
        # Each group should sum to 1 (softmax)
        grouped = out.view(4, 4, 4)
        sums = grouped.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    
    def test_wta_competition(self):
        layer = CompetitionLayer(16, group_size=4, competition_mode="wta")
        # Use distinct values to avoid ties
        x = torch.arange(16, dtype=torch.float32).unsqueeze(0).repeat(4, 1)
        out = layer(x)
        assert out.shape == (4, 16)
        # Each group should have exactly one non-zero (the max)
        grouped = out.view(4, 4, 4)
        nonzeros = (grouped > 0).sum(dim=-1)
        assert (nonzeros == 1).all()
        # The non-zero should be the max value in each group
        for b in range(4):
            for g in range(4):
                group_data = x[b, g*4:(g+1)*4]
                max_val = group_data.max()
                out_group = out[b, g*4:(g+1)*4]
                assert (out_group[out_group > 0] == max_val).all()
    
    def test_sparsemax_competition(self):
        layer = CompetitionLayer(16, group_size=4, competition_mode="sparsemax")
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 16)
        # Each group should sum to 1 (sparsemax property)
        grouped = out.view(4, 4, 4)
        sums = grouped.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        # Should be sparse (some zeros)
        assert (grouped == 0).any()
    
    def test_adaptive_temperature(self):
        layer = CompetitionLayer(16, group_size=4, adaptive_temperature=True)
        x = torch.randn(4, 16)
        out = layer(x)
        assert out.shape == (4, 16)
        temps = layer.get_temperatures()
        assert temps.shape == (4,)  # 4 groups
        assert (temps > 0).all()
    
    def test_geometric_competition(self):
        layer = GeometricCompetitionLayer(16, group_size=4)
        mag = torch.rand(4, 16)
        phase = torch.rand(4, 16) * 2 * torch.pi - torch.pi
        out_mag, out_phase = layer(mag, phase)
        assert out_mag.shape == (4, 16)
        assert out_phase.shape == (4, 16)


class TestIntegrationLayer:
    def test_standard_forward(self):
        layer = IntegrationLayer(20, 30, neuron_type="standard")
        x = torch.randn(4, 20)
        out = layer(x)
        assert out.shape == (4, 30)
    
    def test_with_context(self):
        layer = IntegrationLayer(20, 30, neuron_type="standard", context_dim=10)
        x = torch.randn(4, 20)
        context = torch.randn(4, 10)
        out = layer(x, context)
        assert out.shape == (4, 30)
    
    def test_geometric_forward(self):
        layer = IntegrationLayer(20, 30, neuron_type="geometric")
        x = torch.randn(4, 20)
        mag, phase = layer(x)
        assert mag.shape == (4, 30)
        assert phase.shape == (4, 30)


class TestPredictionLayer:
    def test_standard_forward(self):
        layer = PredictionLayer(20, neuron_type="standard")
        state = torch.randn(4, 20)
        out = layer(state)
        assert "predicted_state" in out
        assert out["predicted_state"].shape == (4, 20)
    
    def test_with_input(self):
        layer = PredictionLayer(20, input_dim=10, neuron_type="standard")
        state = torch.randn(4, 20)
        inp = torch.randn(4, 10)
        out = layer(state, inp)
        assert out["predicted_state"].shape == (4, 20)
    
    def test_with_target(self):
        layer = PredictionLayer(20, neuron_type="standard")
        state = torch.randn(4, 20)
        target = torch.randn(4, 20)
        out = layer(state, target_next_state=target)
        assert "prediction_error" in out
        assert out["prediction_error"].shape == (4, 20)
        assert "error_features" in out
    
    def test_uncertainty(self):
        layer = PredictionLayer(20, neuron_type="standard", use_uncertainty=True)
        state = torch.randn(4, 20)
        out = layer(state)
        assert "uncertainty" in out
        assert out["uncertainty"].shape == (4, 20)
        assert (out["uncertainty"] >= 0).all()
    
    def test_predict_sequence(self):
        layer = PredictionLayer(20, input_dim=10, neuron_type="standard", predict_steps=3)
        state = torch.randn(4, 20)
        inputs = torch.randn(4, 3, 10)
        preds = layer.predict_sequence(state, inputs, steps=3)
        assert preds.shape == (4, 3, 20)
    
    def test_geometric_forward(self):
        layer = PredictionLayer(20, neuron_type="geometric")
        state = torch.randn(4, 20)
        out = layer(state)
        assert out["predicted_state"].shape == (4, 20)
        assert out["predicted_state"].dtype == torch.complex64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])