"""Complete MoE Transformer model with MLA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from .moe import MoETransformerBlock

class MoETransformer(nn.Module):
    """Complete Mixture-of-Experts Transformer with Multi-Head Latent Attention."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.vocab_size = config['model']['vocab_size']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.max_position_embeddings = config['model']['max_position_embeddings']
        self.layer_norm_eps = config['model'].get('layer_norm_eps', 1e-5)
        self.dropout = config['model'].get('dropout', 0.1)
        
        # MoE configuration
        self.num_experts = config['moe']['num_experts']
        self.top_k = config['moe']['top_k']
        self.expert_hidden_size = config['moe']['expert_hidden_size']
        self.load_balance_coeff = config['moe']['load_balance_loss_coeff']
        
        # MLA configuration
        self.mla_interval = config['mla']['mla_interval']
        self.mla_config = {
            'latent_tokens': config['mla']['latent_tokens'],
            'latent_hidden_size': config['mla']['latent_hidden_size']
        }
        
        # Multi-token prediction
        self.multi_token_prediction = config['training']['multi_token_prediction']
        
        # Initialize layers
        self._init_layers()
        
    def _init_layers(self):
        """Initialize model layers."""
        
        # Embedding layers
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.embed_positions = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Transformer layers with MoE and MLA
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers):
            use_mla = (layer_idx % self.mla_interval == 0) and (layer_idx > 0)
            
            layer = MoETransformerBlock(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                expert_hidden_size=self.expert_hidden_size,
                use_mla=use_mla,
                mla_config=self.mla_config if use_mla else None,
                dropout=self.dropout,
                layer_norm_eps=self.layer_norm_eps
            )
            self.layers.append(layer)
        
        # Final layer norm
        self.norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Output layers
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Multi-token prediction heads
        if self.multi_token_prediction > 1:
            self.prediction_heads = nn.ModuleList([
                nn.Linear(self.hidden_size, self.vocab_size, bias=False)
                for _ in range(self.multi_token_prediction)
            ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with proper scaling."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                gate_temperature: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask (optional)
            position_ids: Position IDs (optional)
            gate_temperature: Temperature for MoE gating
            
        Returns:
            logits: Output logits for next token prediction
            multi_token_logits: Logits for multi-token prediction (if enabled)
            load_balance_loss: Total load balancing loss across all layers
        """
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout_layer(hidden_states)
        
        # Forward through transformer layers
        total_load_balance_loss = 0.0
        
        for layer in self.layers:
            hidden_states, load_balance_loss = layer(hidden_states, gate_temperature)
            total_load_balance_loss += load_balance_loss
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Primary language modeling head
        logits = self.lm_head(hidden_states)
        
        # Multi-token prediction heads
        multi_token_logits = []
        if self.multi_token_prediction > 1:
            for head in self.prediction_heads:
                multi_logits = head(hidden_states)
                multi_token_logits.append(multi_logits)
        
        return logits, multi_token_logits, total_load_balance_loss
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 do_sample: bool = True,
                 pad_token_id: int = 0,
                 eos_token_id: int = 2) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            
        Returns:
            generated_ids: Generated token sequences
        """
        
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(generated)
                logits = outputs[0]  # Get main logits
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)
                
                # Check for EOS token
                if (next_tokens == eos_token_id).all():
                    break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Calculate the total number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters from count
            
        Returns:
            n_params: Total number of parameters
        """
        
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            n_params -= self.embed_positions.weight.numel()
            
        return n_params
    
    def save_pretrained(self, save_directory: str):
        """Save model configuration and weights."""
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load model from saved checkpoint."""
        
        # Load configuration
        config_path = os.path.join(model_path, 'config.json')
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model
        model = cls(config)
        
        # Load weights if available
        model_weights_path = os.path.join(model_path, 'pytorch_model.bin')
        if os.path.exists(model_weights_path):
            state_dict = torch.load(model_weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Model weights loaded from {model_weights_path}")
        else:
            print(f"No model weights found at {model_weights_path}, using random initialization")
        
        print(f"Model loaded from {model_path}")
        return model

# Export the main model class
__all__ = ['MoETransformer']
