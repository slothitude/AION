"""
Relational Network - Network built from plastic relational connections.

Instead of fixed layers, this is a graph of neurons connected by
plastic relationships that learn continuously from experience.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from nnfs.connections.plastic import (
    PlasticRelationalConnection,
    create_plastic_connection,
    FastSlowPlasticity,
    EligibilityTrace,
)


class RelationalNetwork(nn.Module):
    """
    Network of neurons connected by plastic relational connections.
    
    Architecture:
    - Neurons: computational nodes with internal state
    - Connections: plastic relational links between neurons
    - Context: global state modulating relationships
    - Predictions: each neuron predicts its next state
    - Learning: continuous online updates from prediction errors
    """
    
    def __init__(
        self,
        num_neurons: int,
        neuron_dim: int = 16,
        sparsity: float = 0.1,
        relation_type: str = "bilinear",
        learning_rate: float = 0.01,
        trace_decay: float = 0.95,
        consolidation_rate: float = 0.001,
        consolidation_threshold: float = 0.1,
        context_dim: int = 8,
        blend_ratio: float = 0.7,
        hidden_dim: int = 32,
        consolidation_interval: int = 1000,
        state_leak_rate: float = 0.1,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.neuron_dim = neuron_dim
        self.sparsity = sparsity
        self.relation_type = relation_type
        self.context_dim = context_dim
        self.blend_ratio = blend_ratio
        self.consolidation_interval = consolidation_interval
        self.state_leak_rate = state_leak_rate
        
        # Neuron states (batch, num_neurons, neuron_dim)
        self.register_buffer("neuron_states", torch.zeros(1, num_neurons, neuron_dim))
        
        # Create sparse connection matrix
        # connections[i, j] = connection from j -> i
        total_possible = num_neurons * num_neurons
        num_active = int(total_possible * sparsity)
        
        # Random sparse connections (no self-connections)
        indices = torch.randperm(total_possible)[:num_active]
        rows = indices // num_neurons
        cols = indices % num_neurons
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]
        
        self.register_buffer("conn_rows", rows)
        self.register_buffer("conn_cols", cols)
        self.num_connections = len(rows)

        # In-degree per neuron: each incoming connection is supervised with its
        # SHARE of the post neuron's target (parallel connections otherwise all
        # chase the full target and their summed transmission overshoots n x).
        in_degree = torch.zeros(num_neurons)
        for i in rows:
            in_degree[i] += 1
        self.register_buffer("in_degree", in_degree)
        
        # Create connections
        self.connections = nn.ModuleDict()
        for idx, (i, j) in enumerate(zip(rows, cols)):
            conn = PlasticRelationalConnection(
                pre_idx=j,
                post_idx=i,
                pre_dim=neuron_dim,
                post_dim=neuron_dim,
                context_dim=context_dim,
                relation_type=relation_type,
                hidden_dim=neuron_dim,
                learning_rate=learning_rate,
                trace_decay=trace_decay,
                consolidation_rate=consolidation_rate,
                consolidation_threshold=consolidation_threshold,
                blend_ratio=blend_ratio,
                consolidation_interval=consolidation_interval,
            )
            self.connections[f"conn_{idx}"] = conn
        
        # Context encoder
        if context_dim > 0:
            self.context_encoder = nn.Linear(num_neurons * neuron_dim, context_dim)
        else:
            self.context_encoder = None
        
        # Global prediction head
        self.prediction_head = nn.Linear(
            num_neurons * neuron_dim, 
            num_neurons * neuron_dim
        )
        
        # Readout for downstream tasks
        self.readout = nn.Linear(num_neurons * neuron_dim, neuron_dim)
        
        # Statistics
        self.register_buffer("step_count", torch.tensor(0))
        
    def reset_states(self, batch_size: int = 1):
        """Reset neuron states and all eligibility traces."""
        device = self.neuron_states.device
        if self.neuron_states.shape[0] != batch_size:
            self.neuron_states = torch.zeros(batch_size, self.num_neurons, self.neuron_dim, device=device)
        else:
            self.neuron_states.zero_()
        for conn in self.connections.values():
            conn.reset_traces()
    
    def rebuild_connection_indices(self):
        """Rebuild connection index buffers after structural changes."""
        # Collect all pre/post indices from current connections
        rows = []
        cols = []
        new_connections = nn.ModuleDict()
        
        for idx, (key, conn) in enumerate(self.connections.items()):
            pre_idx = int(conn.pre_idx.item()) if hasattr(conn.pre_idx, 'item') else int(conn.pre_idx)
            post_idx = int(conn.post_idx.item()) if hasattr(conn.post_idx, 'item') else int(conn.post_idx)
            rows.append(post_idx)
            cols.append(pre_idx)
            new_connections[f"conn_{idx}"] = conn
        
        self.conn_rows = torch.tensor(rows, dtype=torch.long, device=self.conn_rows.device)
        self.conn_cols = torch.tensor(cols, dtype=torch.long, device=self.conn_cols.device)
        self.num_connections = len(rows)
        in_degree = torch.zeros(self.num_neurons, device=self.conn_rows.device)
        for i in rows:
            in_degree[i] += 1
        self.in_degree = in_degree
        self.connections = new_connections
    
    def step(
        self, 
        input_signal: torch.Tensor,
        target_next: Optional[torch.Tensor] = None,
        learn: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Single step of network dynamics.
        
        Args:
            input_signal: External input (batch, num_neurons, neuron_dim) or (batch, neuron_dim)
            target_next: Target next state for prediction error (batch, num_neurons, neuron_dim)
            learn: Whether to update connections
            
        Returns:
            Dict with new states, predictions, errors, and stats
        """
        batch_size = input_signal.shape[0]
        
        # Ensure neuron_states has correct batch size
        if self.neuron_states.shape[0] != batch_size:
            # Create new buffer with correct batch size
            device = self.neuron_states.device
            self.neuron_states = torch.zeros(batch_size, self.num_neurons, self.neuron_dim, device=device)
        
        # Ensure input has right shape
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(1).expand(-1, self.num_neurons, -1)
        
        # Compute context from global state
        context = None
        if self.context_encoder is not None:
            flat_state = self.neuron_states.view(batch_size, -1)
            context = self.context_encoder(flat_state)
        
        # Compute all connection transmissions
        transmissions = torch.zeros_like(self.neuron_states)
        total_stats = {"active_connections": 0, "trace_mean": 0.0, "weight_change": 0.0, "error_norm": 0.0}

        for idx, (i, j) in enumerate(zip(self.conn_rows, self.conn_cols)):
            conn = self.connections[f"conn_{idx}"]

            pre = self.neuron_states[:, j]      # (batch, neuron_dim)
            post = self.neuron_states[:, i]     # (batch, neuron_dim)

            # Ground-truth target for this connection's post neuron, split
            # across the neuron's incoming connections (share supervision).
            conn_target = None
            if target_next is not None:
                conn_target = target_next[:, i] / torch.clamp(self.in_degree[i], min=1.0)

            transmitted, stats = conn(
                pre, post, context,
                target=conn_target,
                learn=learn
            )

            transmitted = transmitted.squeeze(-1) if transmitted.dim() == 3 else transmitted
            transmissions[:, i] += transmitted

            # Aggregate stats
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    total_stats[k] = total_stats.get(k, 0) + v
        
        # Update neuron states: leaky integration
        self.neuron_states = self.neuron_states + self.state_leak_rate * (
            transmissions + input_signal - self.neuron_states
        )
        
        # Global prediction
        flat_state = self.neuron_states.view(batch_size, -1)
        predicted_next = self.prediction_head(flat_state).view(batch_size, self.num_neurons, self.neuron_dim)
        
        # Readout
        output = self.readout(flat_state)
        
        result = {
            "states": self.neuron_states,
            "transmissions": transmissions,
            "predicted_next": predicted_next,
            "output": output,
            "context": context,
            "stats": total_stats,
        }

        # Metric contract: PRIMARY = connection_prediction_mse (the actual
        # plastic network output - summed transmissions - vs ground truth);
        # state_prediction_mse = integrated next-state prediction;
        # readout_mse = untrained prediction_head (legacy "mse").
        connection_prediction_mse = None
        state_prediction_mse = None
        readout_mse = None
        if target_next is not None:
            pred_error = target_next - predicted_next
            result["prediction_error"] = pred_error
            readout_mse = pred_error.pow(2).mean().item()
            state_prediction_mse = (
                (self.neuron_states - target_next).pow(2).mean().item()
            )
            connection_prediction_mse = (
                (transmissions - target_next).pow(2).mean().item()
            )

        result["connection_prediction_mse"] = connection_prediction_mse
        result["state_prediction_mse"] = state_prediction_mse
        result["readout_mse"] = readout_mse
        # Legacy alias - do NOT use to judge learning (head is untrained)
        result["mse"] = readout_mse

        self.step_count.add_(1)
        return result
    
    def consolidate_all(self) -> Dict[str, float]:
        """Consolidate all connections."""
        total_stats = {"consolidated": 0}
        for conn in self.connections.values():
            stats = conn.consolidate()
            for k, v in stats.items():
                total_stats[k] = total_stats.get(k, 0) + v
        return total_stats
    
    def forward(self, x: torch.Tensor, steps: int = 1, target_seq: Optional[torch.Tensor] = None):
        """
        Run network for multiple steps.
        
        Args:
            x: Input sequence (batch, steps, num_neurons, neuron_dim) or single step
            steps: Number of steps
            target_seq: Target sequence for learning (batch, steps, num_neurons, neuron_dim)
            
        Returns:
            List of step outputs
        """
        if x.dim() == 3:  # (batch, steps, features)
            x = x.unsqueeze(2)  # Add neuron dim
        
        outputs = []
        for t in range(steps):
            if x.dim() == 4:
                inp = x[:, t]
            else:
                inp = x
            
            target = target_seq[:, t] if target_seq is not None else None
            
            out = self.step(inp, target_next=target, learn=self.training)
            outputs.append(out)
        
        return outputs


def create_relational_network(
    num_neurons: int = 32,
    neuron_dim: int = 16,
    sparsity: float = 0.1,
    relation_type: str = "bilinear",
    **kwargs
) -> RelationalNetwork:
    """Factory function for relational networks."""
    return RelationalNetwork(
        num_neurons=num_neurons,
        neuron_dim=neuron_dim,
        sparsity=sparsity,
        relation_type=relation_type,
        **kwargs
    )