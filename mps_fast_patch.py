"""
Fast MPS optimization for Chatterbox TTS that pre-computes rotary embeddings.
"""

import torch
import torch.nn as nn
import logging
import math
from functools import lru_cache

logger = logging.getLogger(__name__)


class FastMPSRotaryEmbedding(nn.Module):
    """
    Ultra-fast rotary embedding for MPS that pre-computes everything.
    """
    
    def __init__(self, dim=None, max_position_embeddings=2048, base=10000, device=None, config=None):
        """
        Initializes the FastMPSRotaryEmbedding module with precomputed rotary embedding parameters.
        
        Supports initialization via explicit parameters or a configuration object. Precomputes inverse frequencies and caches cosine and sine embeddings for all positions up to the maximum sequence length on CPU for efficient rotary positional encoding on MPS devices.
        """
        super().__init__()
        
        # Handle both old-style (dim, max_position_embeddings, base) and new-style (config) initialization
        if config is not None:
            self.dim = config.rope_dim if hasattr(config, 'rope_dim') else config.hidden_size // config.num_attention_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.base = config.rope_theta if hasattr(config, 'rope_theta') else 10000.0
            self.rope_type = getattr(config, 'rope_type', 'default')
        else:
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.rope_type = "default"
            
        self.attention_scaling = 1.0  # Compatibility
        self._call_count = 0  # Track number of calls
        
        # Pre-compute everything on CPU once
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Pre-compute cos/sin for all positions
        self._precompute_all_positions()
    
    def _precompute_all_positions(self):
        """
        Precompute and cache cosine and sine rotary embeddings for all positions up to the maximum sequence length.
        
        This method generates and stores the cosine and sine values for rotary positional encoding as non-persistent buffers, enabling efficient retrieval during model inference.
        """
        # Create position indices
        position_ids = torch.arange(self.max_position_embeddings, dtype=torch.float32).unsqueeze(0)
        
        # Compute frequencies
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(1, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Compute on CPU to avoid MPS issues
        with torch.no_grad():
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Pre-compute cos and sin
            self.register_buffer("cos_cached", emb.cos(), persistent=False)
            self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Returns precomputed cosine and sine rotary embeddings for the given input tensor and position IDs.
        
        If `position_ids` are provided, gathers the corresponding cosine and sine embeddings for each batch element; otherwise, uses sequential positions up to the input sequence length. The returned embeddings are moved to match the device and dtype of the input tensor.
        
        Parameters:
            x (torch.Tensor): Input tensor whose shape determines batch size and sequence length.
            position_ids (torch.Tensor or None): Optional tensor specifying position indices for each batch element.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Cosine and sine embedding tensors of shape [batch_size, seq_len, dim].
        """
        self._call_count += 1
        if self._call_count % 100 == 0:
            logger.debug(f"FastMPSRotaryEmbedding called {self._call_count} times")
        
        # Handle different input shapes
        # x can be [batch_size, seq_len, ...] or [batch_size, num_heads, seq_len, ...]
        if len(x.shape) == 4:
            batch_size = x.shape[0]
            seq_len = x.shape[2]
        else:
            batch_size = position_ids.shape[0] if position_ids is not None else x.shape[0]
            seq_len = position_ids.shape[-1] if position_ids is not None else x.shape[1]
        
        # Get the actual positions we need
        if position_ids is not None:
            # position_ids shape: [batch_size, seq_len]
            # We need to gather the cos/sin values for these specific positions
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            
            # Gather cos/sin for the specific positions
            # cos_cached shape: [1, max_seq_len, dim]
            # We need to index into the sequence dimension
            cos_list = []
            sin_list = []
            
            for b in range(batch_size):
                pos = position_ids[b]  # [seq_len]
                cos_b = self.cos_cached[0, pos]  # [seq_len, dim]
                sin_b = self.sin_cached[0, pos]  # [seq_len, dim]
                cos_list.append(cos_b.unsqueeze(0))
                sin_list.append(sin_b.unsqueeze(0))
            
            cos = torch.cat(cos_list, dim=0)  # [batch_size, seq_len, dim]
            sin = torch.cat(sin_list, dim=0)  # [batch_size, seq_len, dim]
        else:
            # No position_ids provided, use sequential positions
            cos = self.cos_cached[:, :seq_len, :]  # [1, seq_len, dim]
            sin = self.sin_cached[:, :seq_len, :]  # [1, seq_len, dim]
            
            if batch_size > 1:
                cos = cos.expand(batch_size, -1, -1)
                sin = sin.expand(batch_size, -1, -1)
        
        # Move to target device and dtype
        cos = cos.to(device=x.device, dtype=x.dtype, non_blocking=True)
        sin = sin.to(device=x.device, dtype=x.dtype, non_blocking=True)
        
        return cos, sin


def monkey_patch_llama_for_mps():
    """
    Replaces the Hugging Face transformers LlamaRotaryEmbedding class with FastMPSRotaryEmbedding for improved MPS performance.
    
    Returns:
        bool: True if the patch was applied successfully or was already present, False if patching failed.
    """
    try:
        from transformers.models.llama import modeling_llama
        
        # Check if already patched
        if hasattr(modeling_llama.LlamaRotaryEmbedding, '_mps_patched'):
            logger.info("✅ LlamaRotaryEmbedding already patched for MPS")
            return True
        
        # Save original
        _original_rotary = modeling_llama.LlamaRotaryEmbedding
        
        # Replace with our fast version
        modeling_llama.LlamaRotaryEmbedding = FastMPSRotaryEmbedding
        
        # Mark as patched
        FastMPSRotaryEmbedding._mps_patched = True
        
        logger.info("✅ Monkey-patched LlamaRotaryEmbedding for fast MPS performance")
        return True
    except Exception as e:
        logger.error(f"Failed to monkey-patch: {e}")
        return False


def optimize_chatterbox_for_fast_mps(chatterbox_model):
    """
    Optimizes a Chatterbox TTS model for fast inference on Apple MPS devices by replacing rotary embeddings and configuring backend settings.
    
    This function monkey-patches the Hugging Face transformers library to use a fast MPS-optimized rotary embedding implementation, replaces all rotary embedding modules in the model's transformer with the optimized version, disables PyTorch autocasting, enables fast matrix multiplication for MPS if available, and ensures key model components are set to evaluation mode.
    
    Returns:
        The optimized Chatterbox model with fast MPS rotary embeddings and backend settings applied.
    """
    logger.info("⚡ Applying fast MPS optimizations...")
    
    # First, monkey-patch the transformers library
    monkey_patch_llama_for_mps()
    
    # Replace existing rotary embeddings in the T3 transformer
    replaced = 0
    if hasattr(chatterbox_model, 't3') and hasattr(chatterbox_model.t3, 'tfmr'):
        # Work on the transformer model
        tfmr = chatterbox_model.t3.tfmr
        
        for name, module in tfmr.named_modules():
            if module.__class__.__name__ == "LlamaRotaryEmbedding":
                # Get parent and attribute
                parts = name.split('.')
                parent = tfmr
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                # Get parameters
                dim = module.inv_freq.shape[0] * 2 if hasattr(module, 'inv_freq') else 128
                max_pos = getattr(module, 'max_position_embeddings', 2048)
                base = getattr(module, 'base', 10000)
                
                # Create and set new module - ensure it's on the same device as the original
                device = next(module.parameters()).device if list(module.parameters()) else module.inv_freq.device
                new_module = FastMPSRotaryEmbedding(dim, max_pos, base, device)
                
                # Move to the same device as the original module
                new_module = new_module.to(device)
                
                setattr(parent, parts[-1], new_module)
                replaced += 1
        
        if replaced > 0:
            logger.info(f"✅ Replaced {replaced} rotary embeddings with fast MPS version")
    
    # Disable autocast for MPS
    torch.set_autocast_enabled(False)
    
    # Enable fast matmul for MPS
    if hasattr(torch.backends.mps, 'enable_fast_math'):
        torch.backends.mps.enable_fast_math(True)
    
    # Verify model is still in eval mode
    if hasattr(chatterbox_model.t3, 'eval'):
        chatterbox_model.t3.eval()
    if hasattr(chatterbox_model.s3gen, 'eval'):
        chatterbox_model.s3gen.eval()
    if hasattr(chatterbox_model.ve, 'eval'):
        chatterbox_model.ve.eval()
    
    return chatterbox_model

# Apply the monkey patch immediately when the module is imported
# This ensures any new model instantiations use our optimized version
if torch.backends.mps.is_available():
    monkey_patch_llama_for_mps() 