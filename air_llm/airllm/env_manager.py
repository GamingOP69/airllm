import os
import time
import json
import torch
import psutil
import platform
import shutil

class AirLlmEnvManager:
    """
    Environment Intelligence Layer for AirLLM.
    Detects hardware capabilities and classifies the environment into tiers.
    """
    
    def __init__(self):
        self.profile = self._discover()
        self.tier = self.profile['tier']

    def _discover(self):
        # 1. CPU & OS
        cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count(logical=True)
        ram_total_gb = psutil.virtual_memory().total / (1024**3)
        
        # 2. GPU Detection
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                mem_total = prop.total_memory / (1024**2) # MB
                gpus.append({
                    "id": i,
                    "name": prop.name,
                    "vram_total_mb": mem_total,
                    "cc": f"{prop.major}.{prop.minor}"
                })
        
        # 3. Disk Performance Probe
        disk_speed_mbps = self._probe_disk_speed()
        
        # 4. Capability Classification
        tier = 0
        if not gpus:
            tier = 0 # CPU Only
        else:
            max_vram = max([g['vram_total_mb'] for g in gpus])
            if max_vram <= 4096:
                tier = 1 # Constrained
            elif max_vram <= 12288:
                tier = 2 # Standard
            else:
                tier = 3 # High-end
                
        # 5. Build Profile
        profile = {
            "version": "2.0.0-runtime",
            "tier": tier,
            "os": platform.system(),
            "cpu": {
                "physical_cores": cpu_count,
                "logical_cores": logical_cpu_count,
                "ram_gb": round(ram_total_gb, 2)
            },
            "gpus": gpus,
            "disk": {
                "read_mbps": round(disk_speed_mbps, 2)
            },
            "software": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
        }
        return profile

    def _probe_disk_speed(self):
        """Measures disk sequential read speed using a small burst."""
        temp_file = ".disk_probe.tmp"
        size_mb = 64
        chunk_size = 1024 * 1024 # 1MB
        
        try:
            # Write 64MB junk
            data = os.urandom(chunk_size)
            with open(temp_file, "wb") as f:
                for _ in range(size_mb):
                    f.write(data)
            
            # Flush OS cache if possible (difficult on Windows without ctypes, but we'll approximate)
            # Measure read
            start = time.time()
            with open(temp_file, "rb") as f:
                while f.read(chunk_size):
                    pass
            end = time.time()
            
            speed = size_mb / (end - start)
            return speed
        except Exception:
            return 100.0 # Fallback default
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def print_summary(self):
        print("="*40)
        print(" AIRLLM ENVIRONMENT INTELLIGENCE ")
        print(f" Detected Tier: {self.tier}")
        print(f" OS: {self.profile['os']}")
        print(f" RAM: {self.profile['cpu']['ram_gb']} GB")
        if self.profile['gpus']:
            for g in self.profile['gpus']:
                print(f" GPU[{g['id']}]: {g['name']} ({round(g['vram_total_mb']/1024, 1)} GB VRAM)")
        else:
            print(" GPU: [NONE/NOT_DETECTED]")
        print(f" Disk Read Speed: {self.profile['disk']['read_mbps']} MB/s")
        print("="*40)

    def get_capability_json(self):
        return json.dumps(self.profile, indent=2)
