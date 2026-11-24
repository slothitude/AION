"""Layer 5: Prediction - Predicts next state and compares against reality."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from nnfs.neurons.standard import StandardNeuron
from nnfs.neurons.multiplicative import MultiplicativeNeuron
from nnfs.neurons.geometric import GeometricNeuron, GeometricLayer


class PredictionLayer(nn.Module):
    """
    Prediction Layer - Layer 5
    
    Predicts the next state and computes prediction error.
    This layer implements PREDICTIVE CODING:
    - Forward prediction: next state from current
    - Error computation: prediction vs reality
    - Error-driven learning signals
    - Uncertainty estimation
    """
    
    def __init__(
        self,
        state_dim: int,
        input_dim: Optional[int] = None,
        neuron_type: str = "standard",
        activation: str = "relu",
        dropout: float = 0.0,
        # Multiplicative-specific
        interaction_rank: int = 16,
        use_memory: bool = True,
        memory_decay: float = 0.95,
        # Geometric-specific
        use_complex_weights: bool = True,
        # Prediction-specific
        predict_steps: int = 1,
        use_uncertainty: bool = True,
        error_scale: float = 1.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim or state_dim
        self.neuron_type = neuron_type
        self.predict_steps = predict_steps
        self.use_uncertainty = use_uncertainty
        self.error_scale = error_scale
        
        # State transition predictor: (state + input) -> next_state
        # Can work with or without input_signal
        self.predictor_with_input = True
        predictor_input_dim = state_dim + self.input_dim
        
        if neuron_type == "standard":
            self.predictor = StandardNeuron(predictor_input_dim, state_dim, activation)
            self.predictor_no_input = StandardNeuron(state_dim, state_dim, activation)
            if use_uncertainty:
                self.uncertainty_head = StandardNeuron(predictor_input_dim, state_dim, "softplus")
                self.uncertainty_head_no_input = StandardNeuron(state_dim, state_dim, "softplus")
                
        elif neuron_type == "multiplicative":
            self.predictor = MultiplicativeNeuron(
                predictor_input_dim, state_dim, interaction_rank, activation,
                use_memory=use_memory, memory_decay=memory_decay
            )
            self.predictor_no_input = MultiplicativeNeuron(
                state_dim, state_dim, interaction_rank, activation,
                use_memory=use_memory, memory_decay=memory_decay
            )
            if use_uncertainty:
                self.uncertainty_head = MultiplicativeNeuron(
                    predictor_input_dim, state_dim, interaction_rank, "softplus",
                    use_memory=use_memory, memory_decay=memory_decay
                )
                self.uncertainty_head_no_input = MultiplicativeNeuron(
                    state_dim, state_dim, interaction_rank, "softplus",
                    use_memory=use_memory, memory_decay=memory_decay
                )
                
        elif neuron_type == "geometric":
            self.predictor = GeometricLayer(
                predictor_input_dim, state_dim, activation, use_complex_weights, dropout=0.0
            )
            self.predictor_no_input = GeometricLayer(
                state_dim, state_dim, activation, use_complex_weights, dropout=0.0
            )
            if use_uncertainty:
                self.uncertainty_head = GeometricLayer(
                    predictor_input_dim, state_dim, "softplus", use_complex_weights, dropout=0.0
                )
                self.uncertainty_head_no_input = GeometricLayer(
                    state_dim, state_dim, "softplus", use_complex_weights, dropout=0.0
                )
        else:
            raise ValueError(f"Unknown neuron_type: {neuron_type}")
        
        # Error processor: computes prediction error features
        self.error_processor = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self, 
        state: torch.Tensor, 
        input_signal: Optional[torch.Tensor] = None,
        target_next_state: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state and compute error.
        
        Args:
            state: Current state (..., state_dim)
            input_signal: Optional input (..., input_dim)
            target_next_state: Ground truth next state for error computation
            
        Returns:
            Dict with:
                - predicted_state: (..., state_dim)
                - uncertainty: (..., state_dim) if use_uncertainty
                - prediction_error: (..., state_dim) if target provided
                - error_features: (..., state_dim) processed error
        """
        # Prepare predictor input
        if input_signal is not None:
            predictor_input = torch.cat([state, input_signal], dim=-1)
            use_input = True
        else:
            predictor_input = state
            use_input = False
        
        # Predict next state
        if self.neuron_type == "geometric":
            if use_input:
                pred_mag, pred_phase = self.predictor(predictor_input)
                if self.use_uncertainty:
                    unc_mag, unc_phase = self.uncertainty_head(predictor_input)
            else:
                pred_mag, pred_phase = self.predictor_no_input(predictor_input)
                if self.use_uncertainty:
                    unc_mag, unc_phase = self.uncertainty_head_no_input(predictor_input)
            predicted_state = pred_mag * torch.exp(1j * pred_phase)
            
            if self.use_uncertainty:
                uncertainty = unc_mag  # Use magnitude as uncertainty
            else:
                uncertainty = None
        else:
            if use_input:
                predicted_state = self.predictor(predictor_input)
                if self.use_uncertainty:
                    uncertainty = F.softplus(self.uncertainty_head(predictor_input))
            else:
                predicted_state = self.predictor_no_input(predictor_input)
                if self.use_uncertainty:
                    uncertainty = F.softplus(self.uncertainty_head_no_input(predictor_input))
            if not self.use_uncertainty:
                uncertainty = None
        
        predicted_state = self.dropout(predicted_state)
        
        output = {
            "predicted_state": predicted_state,
            "uncertainty": uncertainty,
        }
        
        # Compute prediction error if target provided
        if target_next_state is not None:
            if self.neuron_type == "geometric":
                # Complex error
                target_complex = target_next_state
                pred_complex = predicted_state
                error_complex = target_complex - pred_complex
                prediction_error = torch.abs(error_complex)  # Magnitude of error
                error_phase = torch.angle(error_complex)
            else:
                prediction_error = target_next_state - predicted_state
                error_phase = None
            
            # Scale error
            prediction_error = prediction_error * self.error_scale
            
            # Process error into features
            if self.neuron_type == "geometric":
                error_input = torch.cat([prediction_error, error_phase], dim=-1)
            else:
                error_input = torch.cat([prediction_error, predicted_state], dim=-1)
            
            error_features = self.error_processor(error_input)
            
            output["prediction_error"] = prediction_error
            output["error_features"] = error_features
            
            if self.neuron_type == "geometric":
                output["error_phase"] = error_phase
        
        return output
    
    def predict_sequence(
        self, 
        initial_state: torch.Tensor, 
        inputs: torch.Tensor,
        steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Autoregressive prediction over multiple steps.
        
        Args:
            initial_state: (..., state_dim)
            inputs: (..., steps, input_dim) or (..., input_dim) for single step
            steps: Number of steps (default: predict_steps)
            
        Returns:
            Predicted states: (..., steps, state_dim)
        """
        steps = steps or self.predict_steps
        current_state = initial_state
        predictions = []
        
        for t in range(steps):
            if inputs.dim() > current_state.dim():
                input_t = inputs[..., t, :]
            else:
                input_t = inputs
            
            pred = self.forward(current_state, input_t)
            current_state = pred["predicted_state"]
            predictions.append(current_state)
        
        return torch.stack(predictions, dim=-2)
    
    def extra_repr(self) -> str:
        return (f"state_dim={self.state_dim}, predict_steps={self.predict_steps}, "
                f"uncertainty={self.use_uncertainty}, type={self.neuron_type}")


class PredictiveCodingLoss(nn.Module):
    """
    Predictive coding loss combining prediction error and complexity.
    
    L = ||x - x̂||² / σ² + log σ² + λ * complexity
    """
    
    def __init__(self, lambda_complexity: float = 0.01):
        super().__init__()
        self.lambda_complexity = lambda_complexity
    
    def forward(
        self, 
        predicted: torch.Tensor, 
        target: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        complexity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute predictive coding loss.
        
        Args:
            predicted: Predicted state
            target: Target state
            uncertainty: Predicted uncertainty (variance)
            complexity: Complexity penalty (e.g., weight norms)
        """
        if uncertainty is not None:
            # Gaussian negative log-likelihood
            error = target - predicted
            nll = 0.5 * (error**2 / (uncertainty + 1e-8) + torch.log(uncertainty + 1e-8))
            loss = nll.mean()
        else:
            # Simple MSE
            loss = F.mse_loss(predicted, target)
        
        if complexity is not None:
            loss = loss + self.lambda_complexity * complexity
        
        return loss