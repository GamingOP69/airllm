import time
from typing import Dict, List, Optional, Any
from .federation import AirLlmNode

class AirLlmNetScheduler:
    """
    Decides 'Where' and 'When' for distributed execution in AirLLM.
    Uses real-time telemetry to minimize latency and maximize throughput.
    """
    def __init__(self, local_tier: int):
        self.local_tier = local_tier
        self.weather_map: Dict[str, Dict[str, Any]] = {}

    def update_telemetry(self, node_id: str, rtt_ms: float, bw_mbps: float):
        """Updates real-time network stats for a node."""
        if node_id not in self.weather_map:
            self.weather_map[node_id] = {"rtt": [], "bw": []}
        
        # Keep sliding window of 5 samples
        self.weather_map[node_id]["rtt"] = (self.weather_map[node_id]["rtt"] + [rtt_ms])[-5:]
        self.weather_map[node_id]["bw"] = (self.weather_map[node_id]["bw"] + [bw_mbps])[-5:]

    def get_avg_rtt(self, node_id: str) -> float:
        if node_id not in self.weather_map or not self.weather_map[node_id]["rtt"]:
            return 10.0 # Default assumption
        return sum(self.weather_map[node_id]["rtt"]) / len(self.weather_map[node_id]["rtt"])

    def get_avg_bw(self, node_id: str) -> float:
        if node_id not in self.weather_map or not self.weather_map[node_id]["bw"]:
            return 100.0 # Default 100Mbps
        return sum(self.weather_map[node_id]["bw"]) / len(self.weather_map[node_id]["bw"])

    def decide_execution_node(self, layer_index: int, layer_name: str, batch_size: int, 
                             seq_len: int, active_peers: List[AirLlmNode]) -> Optional[AirLlmNode]:
        """
        Calculates if offloading is beneficial.
        Returns the target Node if offloading is chosen, else None.
        """
        if not active_peers:
            return None

        # Estimated local cost (loading + compute)
        # Tier 0 (CPU) is slow, Tier 3 is fast.
        local_compute_cost = (4.0 - self.local_tier) * 0.1 # Very rough heuristic
        
        best_candidate = None
        min_remote_cost = float('inf')

        # Tensor size estimate (batch * seq * hidden_size * bits)
        # NF4 hidden states are usually sent as FP16 for compatibility
        tensor_size_mb = (batch_size * seq_len * 4096 * 2) / (1024 * 1024) 

        for peer in active_peers:
            avg_rtt = self.get_avg_rtt(peer.node_id)
            avg_bw = self.get_avg_bw(peer.node_id)
            
            # Transfer cost: (Size/BW) + overhead
            transfer_cost = (tensor_size_mb / (avg_bw / 8)) + (2 * (avg_rtt / 1000.0))
            
            # Remote compute cost (based on peer tier)
            remote_compute_cost = (4.0 - peer.tier) * 0.08 # Peers are optimized workers
            
            total_remote_cost = transfer_cost + remote_compute_cost
            
            if total_remote_cost < min_remote_cost:
                min_remote_cost = total_remote_cost
                best_candidate = peer

        # Only offload if it's significantly better or local is Tier 0
        if self.local_tier == 0 or min_remote_cost < (local_compute_cost * 0.8):
            return best_candidate
        
        return None
