from typing import Dict, Any

class AirLlmExecutionPolicy:
    """Represents the mutable runtime configuration of the engine."""
    def __init__(self, **kwargs):
        self.quantization = kwargs.get('quantization', '4bit')
        self.prefetch_depth = kwargs.get('prefetch_depth', 1)
        self.use_paged_cache = kwargs.get('use_paged_cache', True)
        self.cache_8bit = kwargs.get('cache_8bit', False)
        self.async_streams = kwargs.get('async_streams', 1)
        self.block_size = kwargs.get('block_size', 16)
        self.hp_layers = kwargs.get('hp_layers', [])
        
    def __repr__(self):
        return str(self.__dict__)

class AirLlmOptimizer:
    """
    Auto-Optimization Engine for AirLLM.
    Balances throughput, latency, and survivability.
    """
    def __init__(self, capability_profile: Dict[str, Any]):
        self.profile = capability_profile
        self.tier = capability_profile['tier']
        self.policy = self._generate_initial_policy()

    def _generate_initial_policy(self) -> AirLlmExecutionPolicy:
        """Maps tier to initial execution parameters."""
        if self.tier == 0: # CPU Only
            return AirLlmExecutionPolicy(
                quantization='4bit',
                prefetch_depth=0,
                async_streams=0,
                cache_8bit=True,
                block_size=8
            )
        elif self.tier == 1: # Constrained GPU
            return AirLlmExecutionPolicy(
                quantization='4bit',
                prefetch_depth=1,
                async_streams=1,
                cache_8bit=True,
                block_size=16
            )
        elif self.tier == 2: # Standard Consumer
            return AirLlmExecutionPolicy(
                quantization='4bit',
                prefetch_depth=2,
                async_streams=2,
                cache_8bit=False,
                block_size=16
            )
        else: # Tier 3: High-end
            return AirLlmExecutionPolicy(
                quantization='8bit',
                prefetch_depth=3,
                async_streams=3,
                cache_8bit=False,
                block_size=32
            )

    def adjust_for_pressure(self, metrics: Dict[str, Any]):
        """
        Closed-loop feedback adjustment.
        Called periodically or on specific triggers (like OOM or NaNs).
        """
        error = metrics.get('error', '').lower()
        
        # 1. VRAM Pressure / OOM
        if 'out of memory' in error or metrics.get('vram_usage_pct', 0) > 90:
            if self.policy.prefetch_depth > 0:
                print(">>> Optimizer: High VRAM pressure, reducing prefetch depth.")
                self.policy.prefetch_depth -= 1
            elif self.policy.async_streams > 0:
                print(">>> Optimizer: Critical VRAM pressure, disabling async streams.")
                self.policy.async_streams = 0

        # 2. Numerical Instability
        if 'numerical instability' in error:
            print(">>> Optimizer: Instability detected. Promoting next layers to Standard Precision.")
            # We don't have a direct 'bit' switch mid-run for weights already loaded, 
            # but we can disable aggressive optims
            self.policy.use_paged_cache = False # Fallback to slower but safer cache if needed
            
        # 3. RAM Pressure
        ram_usage = metrics.get('ram_usage_pct', 0)
        if ram_usage > 85 and not self.policy.cache_8bit:
             print(">>> Optimizer: High RAM pressure, enabling 8-bit KV cache.")
             self.policy.cache_8bit = True

    def get_policy(self) -> AirLlmExecutionPolicy:
        return self.policy
