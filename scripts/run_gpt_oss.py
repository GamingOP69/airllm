import os
import torch
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from air_llm.airllm.airllm_base import AirLLMBaseModel

def run_gpt_oss():
    model_id = "openai/gpt-oss-20b"
    
    # Non-negotiable: All downloads stay in the project dir
    base_dir = "C:\\Github\\airllm"
    shards_path = os.path.join(base_dir, "model_shards", model_id.replace("/", "_"))
    
    print(f">>> Initializing AirLLM for {model_id}")
    print(f">>> Shards being saved to: {shards_path}")
    
    # 1. Initialize Model
    # enable_federation=True enables the distributed substrate
    model = AirLLMBaseModel(
        model_id,
        layer_shards_saving_path=shards_path,
        enable_federation=True,
        compression='4bit', 
        max_seq_len=2048
    )
    
    # 2. Prepare Input
    prompt = "Large Language Models are revolutionizing"
    input_tokens = model.tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. Generate
    print("\n>>> Generating output...")
    output = model.generate(
        input_tokens['input_ids'],
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9
    )
    
    print("\n>>> Final Result:")
    print(model.tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_gpt_oss()
