import torch
import math

class AirLLMCacheManager:
    """
    Paged and Tiered KV Cache Manager for AirLLM.
    Designed for layer-wise inference where VRAM is extremely limited.
    """
    def __init__(self, num_layers, num_heads, head_dim, device="cuda:0", dtype=torch.float16, use_8bit_cache=False):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_8bit_cache = use_8bit_cache
        
        # Layer-wise cache stored in CPU RAM
        self.k_cache = [None] * num_layers
        self.v_cache = [None] * num_layers
        
        # For 8-bit cache metadata
        self.k_scales = [None] * num_layers
        self.v_scales = [None] * num_layers

    def _quantize_8bit(self, tensor):
        # Simple per-channel quantization
        scale = tensor.abs().max(dim=-1, keepdim=True).values / 127.0
        scale = scale.clamp(min=1e-5)
        quantized = (tensor / scale).round().to(torch.int8)
        return quantized, scale

    def _dequantize_8bit(self, quantized, scale):
        return quantized.to(self.dtype) * scale

    def update(self, layer_idx, k_new, v_new):
        """Append new KV tensors to the cache for a specific layer."""
        # Ensure input is on CPU for concatenation to avoid VRAM peaks
        k_cpu = k_new.to("cpu")
        v_cpu = v_new.to("cpu")
        
        if self.use_8bit_cache:
            k_cpu, k_scale = self._quantize_8bit(k_cpu)
            v_cpu, v_scale = self._quantize_8bit(v_cpu)

        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k_cpu
            self.v_cache[layer_idx] = v_cpu
            if self.use_8bit_cache:
                self.k_scales[layer_idx] = k_scale
                self.v_scales[layer_idx] = v_scale
        else:
            # Concatenate along the sequence length dimension (assumed to be 2 for [batch, heads, seq, dim])
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k_cpu], dim=2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v_cpu], dim=2)
            if self.use_8bit_cache:
                self.k_scales[layer_idx] = torch.cat([self.k_scales[layer_idx], k_scale], dim=2)
                self.v_scales[layer_idx] = torch.cat([self.v_scales[layer_idx], v_scale], dim=2)

    def get_layer_kv(self, layer_idx):
        """Get the KV cache for a specific layer, moved to the target device."""
        if self.k_cache[layer_idx] is None:
            return None
        
        k = self.k_cache[layer_idx]
        v = self.v_cache[layer_idx]
        
        if self.use_8bit_cache:
            k = self._dequantize_8bit(k, self.k_scales[layer_idx])
            v = self._dequantize_8bit(v, self.v_scales[layer_idx])
            
        return k.to(self.device, non_blocking=True), v.to(self.device, non_blocking=True)

    def clean(self):
        """Clear all cache."""
        self.k_cache = [None] * self.num_layers
        self.v_cache = [None] * self.num_layers
        self.k_scales = [None] * self.num_layers
        self.v_scales = [None] * self.num_layers
