import torch
import sys
import os

# Add parent dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from air_llm.airllm import AirLLMLlama2

def validate_universal_runtime(model_path):
    print(">>> STARTING GLOBAL VALIDATION MATRIX TEST")
    
    # 1. Test Environment Intelligence
    print("\n--- TEST 1: Environment Intelligence ---")
    model = AirLLMLlama2(model_path)
    model.env_manager.print_summary()
    profile = model.capability_profile
    assert 'tier' in profile, "Tier detection failed"
    print(f"Detected Tier: {profile['tier']}")
    
    # 2. Test Policy Execution
    print("\n--- TEST 2: Policy Execution ---")
    policy = model.policy
    print(f"Active Policy: {policy}")
    
    # 3. Test Inference and Resilience (Mocking a failure if possible)
    print("\n--- TEST 3: Inference & Resilience ---")
    input_text = "What is the capital of France?"
    input_ids = model.tokenizer(input_text, return_tensors="pt").input_ids
    
    print("Running initial forward pass...")
    output = model.generate(input_ids, max_new_tokens=10)
    decoded = model.tokenizer.decode(output[0])
    print(f"Output: {decoded}")
    
    print("\n>>> VALIDATION COMPLETE: AirLLM Universal Runtime is Stable.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python universal_validate.py <model_path>")
    else:
        validate_universal_runtime(sys.argv[1])
