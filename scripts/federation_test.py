import time
import sys
import os

# Add parent dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from air_llm.airllm.federation import AirLlmFederationManager

def test_federation():
    print(">>> STARTING FEDERATION DISCOVERY TEST")
    
    # 1. Start Node A
    print("\nStarting Node A...")
    node_a = AirLlmFederationManager(swarm_key="test_swarm")
    node_a.start()
    
    try:
        # 2. Start Node B
        print("\nStarting Node B...")
        node_b = AirLlmFederationManager(swarm_key="test_swarm")
        node_b.start()
        
        try:
            print("\nWaiting for discovery (10s)...")
            time.sleep(10)
            
            # 3. Check peers
            print("\nNode A Swarm Summary:")
            node_a.print_swarm_summary()
            
            print("\nNode B Swarm Summary:")
            node_b.print_swarm_summary()
            
            peers_a = node_a.get_active_peers()
            peers_b = node_b.get_active_peers()
            
            if len(peers_a) > 0 and len(peers_b) > 0:
                print("\n>>> SUCCESS: Bidirectional peer discovery confirmed.")
            else:
                print("\n>>> FAILURE: Peers did not discover each other. Check local network/firewall.")
        finally:
            node_b.stop()
    finally:
        node_a.stop()

if __name__ == "__main__":
    test_federation()
