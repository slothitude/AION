"""Specialized network with distinct layer types (Detection, Interaction, Competition, Integration, Prediction)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from nnfs.layers.detection import DetectionLayer
from nnfs.layers.interaction import InteractionLayer
from nnfs.layers.competition import CompetitionLayer
from nnfs.layers.integration import IntegrationLayer
from nnfs.layers.prediction import PredictionLayer
from nnfs.memory.multiscale import MultiScaleMemory
from nnfs.neurons.standard import StandardLayer
from nnfs.models.standard_mlp import StandardMLP
from nnfs.models.multiplicative_mlp import MultiplicativeMLP
from nnfs.models.geometric_mlp import GeometricMLP
from nnfs.config import ModelConfig


class SpecializedNet(nn.Module):
    """
    Specialized Neural Architecture with distinct layer types.
    
    Layer 1 - Detection: Identify individual features and signals
    Layer 2 - Interaction: Combine signals using multiplication/geometry
    Layer 3 - Competition: Neurons compete; useful patterns suppress noise
    Layer 4 - Integration: Combine patterns into higher-level representations
    Layer 5 - Prediction: Predict next state and compare against reality
    
    Plus multi-timescale memory system.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        neuron_type: str = "multiplicative",  # "standard", "multiplicative", "geometric"
        activation: str = "relu",
        dropout: float = 0.1,
        # Memory
        use_memory: bool = True,
        fast_dim: int = 128,
        slow_dim: int = 64,
        # Competition
        group_size: int = 8,
        adaptive_temp: bool = True,
        # Prediction
        predict_steps: int = 1,
        use_uncertainty: bool = True,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        
        if config is not None:
            hidden_dim = config.hidden_dims[0] if config.hidden_dims else 256
            activation = config.activation
            dropout = config.dropout
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.neuron_type = neuron_type
        
        # Layer 1: Detection
        self.detection = DetectionLayer(
            input_dim, hidden_dim, neuron_type, activation, dropout,
            interaction_rank=8, use_memory=False,
            use_complex_weights=True
        )
        
        # Layer 2: Interaction
        self.interaction = InteractionLayer(
            hidden_dim, hidden_dim, neuron_type, activation, dropout,
            interaction_rank=16, use_memory=use_memory,
            use_complex_weights=True, interaction_mode="multiplicative"
        )
        
        # Layer 3: Competition
        if neuron_type == "geometric":
            from nnfs.layers.competition import GeometricCompetitionLayer
            self.competition = GeometricCompetitionLayer(
                hidden_dim, group_size=group_size,
                adaptive_temperature=adaptive_temp
            )
        else:
            self.competition = CompetitionLayer(
                hidden_dim, group_size=group_size,
                adaptive_temperature=adaptive_temp
            )
        
        # Layer 4: Integration
        self.integration = IntegrationLayer(
            hidden_dim, hidden_dim, neuron_type, activation, dropout,
            interaction_rank=16, use_memory=use_memory,
            use_complex_weights=True, use_cross_attention=True,
            context_dim=hidden_dim
        )
        
        # Layer 5: Prediction
        self.prediction = PredictionLayer(
            hidden_dim, input_dim, neuron_type, activation, dropout,
            interaction_rank=16, use_memory=use_memory,
            use_complex_weights=True, predict_steps=predict_steps,
            use_uncertainty=use_uncertainty
        )
        
        # Multi-timescale memory
        if use_memory:
            self.memory = MultiScaleMemory(
                hidden_dim, hidden_dim, num_scales=4,
                base_update_rate=0.3, rate_decay=0.3,
                activation=activation, dropout=dropout
            )
        else:
            self.memory = None
        
        # Output projection
        self.output_proj = StandardLayer(hidden_dim, output_dim, "identity", 0.0)
        
        # For geometric neurons, we need special handling
        self.is_geometric = neuron_type == "geometric"
        
        self._param_count = None
    
    def forward(
        self, 
        x: torch.Tensor, 
        target_next: Optional[torch.Tensor] = None,
        return_intermediates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through specialized layers.
        
        Args:
            x: Input (batch, input_dim) or (batch, seq, input_dim)
            target_next: Optional next input for prediction error
            return_intermediates: Return all layer outputs
            
        Returns:
            Dict with logits and optionally intermediates
        """
        intermediates = {}
        is_seq = x.dim() == 3
        
        if is_seq:
            batch, seq_len, _ = x.shape
            # Process sequence step by step for memory
            outputs = []
            for t in range(seq_len):
                out = self._forward_step(x[:, t], target_next[:, t] if target_next is not None else None)
                outputs.append(out["logits"])
            logits = torch.stack(outputs, dim=1)
        else:
            out = self._forward_step(x, target_next)
            logits = out["logits"]
        
        result = {"logits": logits}
        
        if return_intermediates:
            # For single step, return intermediates
            if not is_seq:
                result.update(out)
        
        return result
    
    def _forward_step(
        self, 
        x: torch.Tensor, 
        target_next: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Single forward step."""
        intermediates = {}
        original_input = x  # Preserve original input for prediction
        
        # Layer 1: Detection
        if self.is_geometric:
            det_mag, det_phase = self.detection(x)
            intermediates["detection"] = (det_mag, det_phase)
            x_mag, x_phase = det_mag, det_phase
        else:
            x = self.detection(x)
            intermediates["detection"] = x
        
        # Layer 2: Interaction
        if self.is_geometric:
            # For geometric, pass magnitude as tensor to interaction layer
            # The GeometricLayer inside will handle the phase internally
            int_mag, int_phase = self.interaction(x_mag)
            intermediates["interaction"] = (int_mag, int_phase)
            x_mag, x_phase = int_mag, int_phase
        else:
            x = self.interaction(x)
            intermediates["interaction"] = x
        
        # Layer 3: Competition
        if self.is_geometric:
            comp_mag, comp_phase = self.competition(x_mag, x_phase)
            intermediates["competition"] = (comp_mag, comp_phase)
            x_mag, x_phase = comp_mag, comp_phase
        else:
            x = self.competition(x)
            intermediates["competition"] = x
        
        # Multi-timescale memory
        if self.memory is not None:
            if self.is_geometric:
                mem_out = self.memory(x_mag)
                # Memory output modulates magnitude
                x_mag = x_mag + 0.1 * mem_out
                intermediates["memory"] = mem_out
            else:
                mem_out = self.memory(x)
                x = x + 0.1 * mem_out
                intermediates["memory"] = mem_out
        
        # Layer 4: Integration
        if self.is_geometric:
            int_mag, int_phase = self.integration((x_mag, x_phase))
            intermediates["integration"] = (int_mag, int_phase)
            x_mag, x_phase = int_mag, int_phase
            final_features = x_mag
        else:
            x = self.integration(x)
            intermediates["integration"] = x
            final_features = x
        
        # Layer 5: Prediction (if target provided)
        if target_next is not None:
            if self.is_geometric:
                # For geometric, we'd need complex target - skip for now
                pred_out = {"predicted_state": final_features}
            else:
                # prediction expects (state, input_signal, target_next_state)
                # final_features is state (hidden_dim), original_input is input (input_dim)
                pred_out = self.prediction(final_features, original_input, target_next)
            intermediates["prediction"] = pred_out
        
        # Output projection
        logits = self.output_proj(final_features)
        
        result = {"logits": logits}
        result.update(intermediates)
        return result
    
    def predict_next(self, x: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """Predict next state(s) from current input."""
        self.eval()
        with torch.no_grad():
            out = self._forward_step(x)
            features = out.get("integration", out["logits"])
            
            if self.is_geometric:
                # Simplified for geometric
                predictions = []
                current = features
                for _ in range(steps):
                    pred = self.prediction.predictor(torch.cat([current, x], dim=-1))
                    predictions.append(pred[0])  # magnitude
                    current = pred[0]
                return torch.stack(predictions, dim=1)
            else:
                return self.prediction.predict_sequence(features, x, steps)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        if self._param_count is None:
            self._param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self._param_count
    
    def reset_memory(self):
        """Reset all memory states."""
        if self.memory is not None:
            self.memory.reset_state()
        # Reset multiplicative neuron memories (exclude self to avoid recursion)
        for module in self.modules():
            if module is not self and hasattr(module, 'reset_memory'):
                module.reset_memory()
    
    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
                f"output_dim={self.output_dim}, type={self.neuron_type}")


def create_matched_models(
    input_dim: int,
    output_dim: int,
    target_params: int = 500000,
    hidden_dim: int = 256,
) -> Dict[str, nn.Module]:
    """
    Create three models with approximately matched parameter counts.
    
    This is crucial for fair comparison.
    """
    models = {}
    
    # Standard MLP - adjust hidden dims to match params
    standard = StandardMLP(input_dim, output_dim, [hidden_dim, hidden_dim//2, hidden_dim//4])
    standard_params = standard.count_parameters()
    
    # Scale factor for other models
    scale = (target_params / standard_params) ** 0.5
    adj_hidden = int(hidden_dim * scale)
    adj_hidden = max(64, min(adj_hidden, 512))  # Clamp
    # Ensure divisible by group_size (8) for competition layer
    adj_hidden = (adj_hidden // 8) * 8
    if adj_hidden < 64:
        adj_hidden = 64
    
    models["standard"] = StandardMLP(input_dim, output_dim, [adj_hidden, adj_hidden//2, adj_hidden//4])
    
    # Multiplicative MLP - fewer hidden units due to interaction params
    mult_hidden = int(adj_hidden * 0.8)
    models["multiplicative"] = MultiplicativeMLP(
        input_dim, output_dim, [mult_hidden, mult_hidden//2, mult_hidden//4],
        interaction_rank=8
    )
    
    # Geometric MLP - complex weights = 2x params, so smaller
    geom_hidden = int(adj_hidden * 0.7)
    models["geometric"] = GeometricMLP(
        input_dim, output_dim, [geom_hidden, geom_hidden//2, geom_hidden//4]
    )
    
    # Specialized net - use smaller hidden dim due to many layers
    spec_hidden = max(64, adj_hidden // 2)
    spec_hidden = (spec_hidden // 8) * 8  # Divisible by group_size
    models["specialized"] = SpecializedNet(
        input_dim, output_dim, hidden_dim=spec_hidden, neuron_type="multiplicative",
        use_memory=False  # Disable memory to reduce params
    )
    
    # Report actual params
    for name, model in models.items():
        print(f"{name}: {model.count_parameters():,} parameters")
    
    return models