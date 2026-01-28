import os
import time
import json
import socket
import threading
import uuid
import hashlib
import hmac
import requests
import io
import torch
from flask import Flask, request, jsonify
from zeroconf import IPVersion, ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange
from typing import Dict, List, Optional, Any
from .env_manager import AirLlmEnvManager

class AirLlmNode:
    """Represents a remote AirLLM node in the federation."""
    def __init__(self, node_id: str, addr: str, port: int, profile: Dict[str, Any]):
        self.node_id = node_id
        self.addr = addr
        self.port = port
        self.profile = profile
        self.tier = profile.get('tier', 0)
        self.last_seen = time.time()
        self.is_active = True
        self.stability_score = 1.0

    def __repr__(self):
        return f"AirLlmNode(id={self.node_id}, tier={self.tier}, addr={self.addr})"

class AirLlmFederationManager:
    """
    Handles node discovery and federation for AirLLM.
    Implements UDP beaconing for effortless discovery.
    """
    def __init__(self, port: int = 5588, rpc_port: int = 5589, swarm_key: str = "default_swarm"):
        self.node_id = str(uuid.uuid4())[:8]
        self.port = port
        self.rpc_port = rpc_port
        self.swarm_key = swarm_key
        self.peers: Dict[str, AirLlmNode] = {}
        self.running = False
        self.model_proxy = None # Reference to AirLLMBaseModel instance
        
        # Local stats
        self.env = AirLlmEnvManager()
        self.local_profile = self.env.profile
        self.local_profile['rpc_port'] = self.rpc_port
        
        # Threads
        self._beacon_thread: Optional[threading.Thread] = None
        self._listener_thread: Optional[threading.Thread] = None
        self._rpc_thread: Optional[threading.Thread] = None
        self._zeroconf: Optional[Zeroconf] = None
        self._service_info: Optional[ServiceInfo] = None

    def start(self, model_proxy=None):
        """Starts discovery and heartbeat."""
        if self.running: return
        self.running = True
        self.model_proxy = model_proxy
        
        # 1. Start Listener
        self._listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener_thread.start()
        
        # 2. Start Beacon
        self._beacon_thread = threading.Thread(target=self._beacon_loop, daemon=True)
        self._beacon_thread.start()

        # 3. Start Zeroconf (mDNS)
        self._start_zeroconf()

        # 4. Start RPC Worker
        if self.model_proxy:
            self._rpc_thread = threading.Thread(target=self._rpc_worker_loop, daemon=True)
            self._rpc_thread.start()
        
        print(f">>> Federation: Node {self.node_id} online (Swarm: {self.swarm_key}, RPC: {self.rpc_port})")

    def _rpc_worker_loop(self):
        """Simple Flask-based RPC server for remote execution."""
        app = Flask(__name__)

        @app.route('/execute_layer', methods=['POST'])
        def execute_layer():
            # Security Check
            auth_key = request.headers.get('X-AirLLM-Swarm-Key')
            if auth_key != self.swarm_key:
                return jsonify({"error": "Unauthorized"}), 401

            if not self.model_proxy:
                return jsonify({"error": "No model loaded"}), 500
            
            # Receive and Verify Hash
            remote_hash = request.headers.get('X-AirLLM-Tensor-Hash')
            data = request.data
            local_hash = hashlib.sha256(data).hexdigest()
            
            if remote_hash and local_hash != remote_hash:
                return jsonify({"error": "Tensor integrity check failed"}), 400
            buffer = io.BytesIO(data)
            payload = torch.load(buffer)
            
            layer_index = payload['layer_index']
            layer_name = payload['layer_name']
            tensors = payload['tensors'] # List of hidden states
            kwargs = payload['kwargs']
            
            
            try:
                # locally execute
                results = []
                # ensure the layer is loaded on the worker node
                # but we must use load_layer_to_gpu which is safe and idempotent
                self.model_proxy.load_layer_to_gpu(layer_name)
                layer = self.model_proxy.layers[layer_index]
                
                # Execute via model proxy's resilience loop but forced to local
                # Note: We handle the case where the layer returns (hidden_states, kv, attentions)
                res_batch = []
                for j, seq in enumerate(tensors):
                    # We reuse the internal _run_layer_with_resilience-like logic but simplified
                    # We ignore local failure since we are the worker
                    
                    # For now, let's just use a simple execution loop
                    # But we need to handle the return format
                    if layer_name == self.model_proxy.layer_names_dict['embed']:
                        res = layer(seq)
                    elif layer_name == self.model_proxy.layer_names_dict['norm']:
                        res = self.model_proxy.run_norm(layer, seq)
                    elif layer_name == self.model_proxy.layer_names_dict['lm_head']:
                        res = self.model_proxy.run_lm_head(layer, seq)
                    else:
                        res = layer(seq, **kwargs) # This returns a tuple
                    res_batch.append(res)
                
                # Return results
                out_buffer = io.BytesIO()
                torch.save(res_batch, out_buffer)
                return out_buffer.getvalue(), 200
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        app.run(host='0.0.0.0', port=self.rpc_port, debug=False, use_reloader=False)

    def dispatch_remote(self, node: AirLlmNode, layer_index, layer_name, batch, **kwargs):
        """Sends a layer execution task to a remote node."""
        payload = {
            "layer_index": layer_index,
            "layer_name": layer_name,
            "tensors": batch,
            "kwargs": kwargs
        }
        
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        
        # Calculate Hash
        tensor_hash = hashlib.sha256(data).hexdigest()
        
        headers = {
            'X-AirLLM-Swarm-Key': self.swarm_key,
            'X-AirLLM-Tensor-Hash': tensor_hash,
            'Content-Type': 'application/octet-stream'
        }
        
        url = f"http://{node.addr}:{node.profile.get('rpc_port', 5589)}/execute_layer"
        try:
            start_t = time.time()
            response = requests.post(url, data=data, headers=headers, timeout=30)
            latency = time.time() - start_t
            
            if response.status_code == 200:
                res_buffer = io.BytesIO(response.content)
                # verify response hash if present (in a real implemention we'd add headers to response too)
                return torch.load(res_buffer)
            else:
                node.stability_score *= 0.9 # Penalty
                raise Exception(f"Remote execution failed: {response.text}")
        except Exception as e:
            node.stability_score *= 0.8 # Penalty
            raise e

    def _start_zeroconf(self):
        """Registers node via mDNS for Zero-Config discovery."""
        try:
            self._zeroconf = Zeroconf()
            desc = {'node_id': self.node_id, 'swarm_key': self.swarm_key}
            self._service_info = ServiceInfo(
                "_airllm._tcp.local.",
                f"{self.node_id}._airllm._tcp.local.",
                addresses=[socket.inet_aton("0.0.0.0")], # will be resolved by receiver
                port=self.rpc_port,
                properties=desc,
                server=f"{self.node_id}.local."
            )
            self._zeroconf.register_service(self._service_info)
            
            # Browser to find others
            ServiceBrowser(self._zeroconf, "_airllm._tcp.local.", handlers=[self._on_service_state_change])
            print(">>> Federation: mDNS Zero-Config Discovery active.")
        except Exception as e:
            print(f">>> Federation: mDNS Error: {e}")

    def _on_service_state_change(self, zeroconf, service_type, name, state_change):
        if state_change is ServiceStateChange.Added:
            info = zeroconf.get_service_info(service_type, name)
            if info:
                pid = info.properties.get(b'node_id', b'').decode()
                s_key = info.properties.get(b'swarm_key', b'').decode()
                if s_key == self.swarm_key and pid != self.node_id:
                    addr = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
                    if addr and pid not in self.peers:
                        print(f">>> Federation: Discovered peer {pid} via mDNS at {addr}")
                        # Fetch profile via a small RPC call or just use default for now
                        self.peers[pid] = AirLlmNode(pid, addr, self.port, {"tier": 1})

    def stop(self):
        self.running = False
        if self._zeroconf:
            self._zeroconf.unregister_service(self._service_info)
            self._zeroconf.close()

    def _beacon_loop(self):
        """Broadcasts availability to the network."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        
        payload = {
            "type": "HEARTBEAT",
            "node_id": self.node_id,
            "swarm_key": self.swarm_key,
            "profile": self.local_profile,
            "port": self.port
        }
        
        while self.running:
            try:
                data = json.dumps(payload).encode('utf-8')
                sock.sendto(data, ('<broadcast>', self.port))
            except Exception as e:
                pass # Silently handle network errors
            time.sleep(5)
            self._cleanup_stale_peers()

    def _listen_loop(self):
        """Listens for peer heartbeats."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', self.port))
        sock.settimeout(1.0)
        
        while self.running:
            try:
                data, addr = sock.recvfrom(4096)
                msg = json.loads(data.decode('utf-8'))
                
                if msg.get('swarm_key') != self.swarm_key:
                    continue
                
                peer_id = msg.get('node_id')
                if peer_id == self.node_id:
                    continue
                
                if peer_id not in self.peers:
                    print(f">>> Federation: Discovered new peer {peer_id} at {addr[0]}")
                    self.peers[peer_id] = AirLlmNode(peer_id, addr[0], msg.get('port'), msg.get('profile'))
                else:
                    self.peers[peer_id].last_seen = time.time()
                    self.peers[peer_id].is_active = True
                    
            except socket.timeout:
                continue
            except Exception as e:
                continue

    def _cleanup_stale_peers(self):
        """Removes peers that haven't been seen in a while."""
        now = time.time()
        for pid in list(self.peers.keys()):
            if now - self.peers[pid].last_seen > 15:
                # print(f">>> Federation: Peer {pid} lost (timeout)")
                self.peers[pid].is_active = False

    def get_active_peers(self) -> List[AirLlmNode]:
        return [p for p in self.peers.values() if p.is_active]

    def print_swarm_summary(self):
        active = self.get_active_peers()
        print("="*40)
        print(f" AIRLLM FEDERATION: {len(active)} PEERS ")
        print(f" Local Node: {self.node_id} (Tier {self.local_profile['tier']})")
        for p in active:
            print(f" - Peer {p.node_id}: Tier {p.tier} @ {p.addr}")
        print("="*40)
