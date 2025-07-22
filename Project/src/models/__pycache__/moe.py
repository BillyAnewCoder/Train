"""Mixture-of-Experts implementation with Multi-Head Latent attention."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional, Tuple, List, Dict, Any

class Expert(nn.Module):
    """Single expert network (2-layer MLP with SwiGLU activation)."""
    
    def __init__(self, hidden_size: int, expert_hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        
        # SwiGLU requires two linear projections
        self.gate_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=False)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize expert weights with proper scaling."""
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert network."""
        # SwiGLU activation: gate * silu(up)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        activated = F.silu(up)  # SiLU activation
        gated = gate * activated
        output = self.down_proj(gated)
        return self.dropout(output)

class GatingNetwork(nn.Module):
    """Gating network for expert selection with load balancing."""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # Initialize gate weights
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through gating network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            temperature: Temperature for softmax (lower = more concentrated)
            
        Returns:
            top_k_gates: Gate values for top-k experts
            top_k_indices: Indices of top-k experts
            load_balance_loss: Load balancing auxiliary loss
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute gate logits
        gate_logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        
        # Apply temperature scaling
        gate_logits = gate_logits / temperature
        
        # Compute softmax probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize top-k gates
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute load balancing loss
        # Encourage uniform expert usage across the batch
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.top_k):
            expert_counts.scatter_add_(0, top_k_indices[:, :, i].flatten(), 
                                     torch.ones_like(top_k_indices[:, :, i].flatten(), dtype=torch.float))
        
        # Coefficient of variation as load balance loss
        expert_usage = expert_counts / expert_counts.sum()
        cv_squared = (expert_usage.var() / (expert_usage.mean() + 1e-8)) ** 2
        load_balance_loss = cv_squared
        
        return top_k_gates, top_k_indices, load_balance_loss

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with efficient token routing."""
    
    def __init__(self, 
                 hidden_size: int,
                 num_experts: int,
                 top_k: int = 2,
                 expert_hidden_size: int = None,
                 dropout: float = 0.1,
                 load_balance_coeff: float = 0.01):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_size = expert_hidden_size or hidden_size * 4
        self.load_balance_coeff = load_balance_coeff
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, self.expert_hidden_size, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = GatingNetwork(hidden_size, num_experts, top_k)
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer with efficient routing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            temperature: Gate temperature
            
        Returns:
            output: Mixed expert outputs
            load_balance_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute gating
        top_k_gates, top_k_indices, load_balance_loss = self.gate(x, temperature)
        
        # Flatten for expert processing
        x_flat = x.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process through selected experts
        for i in range(self.top_k):
            # Get expert indices and gates for this position
            expert_idx = top_k_indices[:, :, i].flatten()  # (batch_size * seq_len,)
            gates = top_k_gates[:, :, i].flatten()  # (batch_size * seq_len,)
            
            # Process each expert
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                expert_mask = (expert_idx == expert_id)
                if not expert_mask.any():
                    continue
                
                # Get tokens for this expert
                expert_tokens = x_flat[expert_mask]
                expert_gates = gates[expert_mask]
                
                # Process through expert
                expert_output = self.experts[expert_id](expert_tokens)
                
                # Weight by gate values and add to output
                weighted_output = expert_output * expert_gates.unsqueeze(-1)
                output[expert_mask] += weighted_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, load_balance_loss

class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention for memory efficiency."""
    
    def __init__(self,
                 hidden_size: int,
                 latent_tokens: int = 128,
                 latent_hidden_size: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.latent_tokens = latent_tokens
        self.latent_hidden_size = latent_hidden_size
        self.num_heads = num_heads
        self.head_dim = latent_hidden_size // num_heads
        self.dropout = dropout
        
        assert latent_hidden_size % num_heads == 0, "latent_hidden_size must be divisible by num_heads"
        
        # Projection layers
        self.to_latent = nn.Linear(hidden_size, latent_tokens * latent_hidden_size)
        self.from_latent = nn.Linear(latent_tokens * latent_hidden_size, hidden_size)
        
        # Attention projections
        self.q_proj = nn.Linear(latent_hidden_size, latent_hidden_size, bias=False)
        self.k_proj = nn.Linear(latent_hidden_size, latent_hidden_size, bias=False)
        self.v_proj = nn.Linear(latent_hidden_size, latent_hidden_size, bias=False)
        self.o_proj = nn.Linear(latent_hidden_size, latent_hidden_size)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        nn.init.normal_(self.to_latent.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.from_latent.weight, mean=0.0, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLA layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            output: Processed tensor of same shape as input
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Project to latent space
        latent_flat = self.to_latent(x)  # (batch_size, seq_len, latent_tokens * latent_hidden_size)
        latent = latent_flat.view(batch_size, seq_len, self.latent_tokens, self.latent_hidden_size)
        
        # Reshape for attention computation
        latent_reshaped = latent.view(batch_size * seq_len, self.latent_tokens, self.latent_hidden_size)
        
        # Multi-head attention on latent tokens
        q = self.q_proj(latent_reshaped)
        k = self.k_proj(latent_reshaped)
        v = self.v_proj(latent_reshaped)
        
        # Reshape for multi-head computation
        q = q.view(batch_size * seq_len, self.latent_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size * seq_len, self.latent_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size * seq_len, self.latent_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size * seq_len, self.latent_tokens, self.latent_hidden_size
        )
        attn_output = self.o_proj(attn_output)
        
        # Reshape back to original sequence structure
        attn_output = attn_output.view(batch_size, seq_len, self.latent_tokens, self.latent_hidden_size)
        
        # Project back to original hidden size
        output_flat = attn_output.view(batch_size, seq_len, self.latent_tokens * self.latent_hidden_size)
        output = self.from_latent(output_flat)
        
        return output

class MoETransformerBlock(nn.Module):
    """Transformer block with MoE and optional MLA."""
    
    def __init__(self,
                 hidden_size: int,
                 num_experts: int,
                 top_k: int = 2,
                 expert_hidden_size: int = None,
                 use_mla: bool = False,
                 mla_config: Dict[str, Any] = None,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.use_mla = use_mla
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
        # Multi-Head Latent Attention (optional)
        if use_mla and mla_config:
            self.mla = MultiHeadLatentAttention(
                hidden_size=hidden_size,
                dropout=dropout,
                **mla_config
            )
        else:
            self.mla = None
            
        # Mixture of Experts
        self.moe = MixtureOfExperts(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_size=expert_hidden_size or hidden_size * 4,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through transformer block."""
        
        # Input layernorm
        normed_input = self.input_layernorm(x)
        
        # Multi-Head Latent Attention (if enabled)
        if self.mla:
            attn_output = self.mla(normed_input)
            x = x + attn_output  # Residual connection
            
            # Post-attention layernorm
            normed_input = self.post_attention_layernorm(x)
        
        # Mixture of Experts
        moe_output, load_balance_loss = self.moe(normed_input, temperature)
        
        # Residual connection
        output = x + moe_output
        
        return output, load_balance_loss

# Export key classes
__all__ = [
    'Expert',
    'GatingNetwork', 
    'MixtureOfExperts',
    'MultiHeadLatentAttention',
    'MoETransformerBlock'
]
