
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, GenerationMixin, LlamaForCausalLM, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights

from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.quantizers import AutoHfQuantizer, HfQuantizer

from .profiler import LayeredProfiler

try:
    from optimum.bettertransformer import BetterTransformer
    better_transformer_available = True
except (ImportError, RuntimeError):
    better_transformer_available = False

from .utils import clean_memory, load_layer, \
    find_or_create_local_splitted_path

from .cache_manager import AirLLMCacheManager
from .env_manager import AirLlmEnvManager
from .optimizer import AirLlmOptimizer
from .federation import AirLlmFederationManager
from .net_scheduler import AirLlmNetScheduler

try:
    import bitsandbytes as bnb

    bitsandbytes_installed = True
    print('>>>> bitsandbytes installed')
except ImportError:
    bitsandbytes_installed = False



try:
    from transformers.cache_utils import Cache, DynamicCache

    cache_utils_installed = True
    print('>>>> cache_utils installed')
except ImportError:
    cache_utils_installed = False






class AirLLMBaseModel(GenerationMixin):

    # customize layer names here
    def set_layer_names_dict(self):
        self.layer_names_dict = {'embed': 'model.embed_tokens',
                       'layer_prefix': 'model.layers',
                       'norm': 'model.norm',
                       'lm_head': 'lm_head',}



    def __init__(self, model_local_path_or_repo_id, device="cuda:0", dtype=torch.float16, max_seq_len=512,
                 layer_shards_saving_path=None, profiling_mode=False, compression=None,
                 hf_token=None, prefetching=True, delete_original=False, hp_layers=None,
                 enable_federation=False, swarm_key="default_swarm"):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM.

        Parameters
        ----------
        model_local_path_or_repo_id : str or Path
            path to the local model checkpoint or huggingface repo id
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        max_seq_len : int, optional
            max seq lenght, by default 512
        layer_shards_saving_path : str, optional
            optional path to save layered shards model file, by default just save to the local cache of model, subdir named splitted_model will be saved
        profiling_mode : book, optional
            if to profile the model loading time, default to False
        compression: str, optinal
            setting to '4bit' or '8bit' to enable compression from 16 bits to 4 bits/8 bits which speeed up 4x or 2x inference time with a tiny accuracy loss.
        hf_token: str, optional
            huggingface api token could be provided, by default None
        """


        self.profiling_mode = profiling_mode
        self.profiler = LayeredProfiler()

        self.total_disk_loading_time = None
        self.total_gpu_loading_time = None
        self.total_compression_overhead_time = None
        self._supports_cache_class = False
        self.hf_quantizer = None

        if compression is not None:
            if not bitsandbytes_installed:
                raise ImportError('WARNING: bitsandbytes not found. Compression needs bitsandbytes. To use compression, please install bitsandbytes: `pip install bitsandbytes`')


        self.compression = compression
        self.hf_token = hf_token
        self.hp_layers = hp_layers if hp_layers is not None else []

        # Phase 11: Initialize Node Federation
        self.federation = None
        if enable_federation:
            self.federation = AirLlmFederationManager(swarm_key=swarm_key)
            self.federation.start(model_proxy=self)
            self.net_scheduler = AirLlmNetScheduler(local_tier=self.env_manager.tier)
        else:
            self.net_scheduler = None
        self.env_manager = AirLlmEnvManager()
        self.env_manager.print_summary()
        self.capability_profile = self.env_manager.profile

        # Phase 7: Initialize Auto-Optimization Engine
        self.optimizer = AirLlmOptimizer(self.capability_profile)
        self.policy = self.optimizer.get_policy()

        # Apply Policy Overrides
        if compression is None:
            self.compression = self.policy.quantization
        self.prefetch_depth = self.policy.prefetch_depth
        
        # If user explicitly passed hp_layers, respect them; otherwise use policy
        if not self.hp_layers and self.policy.hp_layers:
            self.hp_layers = self.policy.hp_layers

        # Save parameters

        self.set_layer_names_dict()


        self.model_local_path, self.checkpoint_path = find_or_create_local_splitted_path(model_local_path_or_repo_id,
                                                                                         layer_shards_saving_path,
                                                                                         compression=compression,
                                                                                         layer_names=self.layer_names_dict,
                                                                                         hf_token=hf_token,
                                                                                         delete_original=delete_original)
        self.running_device = device
        self.device = torch.device(self.running_device)
        self.running_dtype = dtype
        self.dtype = self.running_dtype

        # Create model
        if hf_token is not None:
            self.config = AutoConfig.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            self.config = AutoConfig.from_pretrained(self.model_local_path, trust_remote_code=True)

        self.generation_config = self.get_generation_config()
        #print(f"using generation_config: {self.generation_config}")

        self.tokenizer = self.get_tokenizer(hf_token=hf_token)


        self.init_model()

        # get layer count:
        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        layers_count = len(model_attr)


        self.layer_names = [self.layer_names_dict['embed']] + [f'{self.layer_names_dict["layer_prefix"]}.{i}' for i in
                                                               range(layers_count)] + \
                           [self.layer_names_dict['norm'], self.layer_names_dict['lm_head']]

        self.max_seq_len = max_seq_len

        self.main_input_name = "input_ids"

        # model weights prefetch cuda stream
        # If policy says prefetch_depth > 0, we enable prefetching
        self.prefetching = prefetching and (self.prefetch_depth > 0)

        if self.prefetching:
             print(f"enabling prefetching with depth {self.prefetch_depth}")

        # this operation should run only if gpu is available
        if self.prefetching and device.startswith("cuda"):
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        self.cache_manager = None

    # if derived class needs to create generation config differently, like Mistrial, this function can be overridden
    def get_generation_config(self):
        # protective on generation config

        try:
            return GenerationConfig.from_pretrained(self.model_local_path)
        except Exception as e:
            return GenerationConfig()

    # a chance to customize tokenizer
    def get_tokenizer(self, hf_token=None):
        if hf_token is not None:
            return AutoTokenizer.from_pretrained(self.model_local_path, token=hf_token, trust_remote_code=True)
        else:
            return AutoTokenizer.from_pretrained(self.model_local_path, trust_remote_code=True)

    def get_use_better_transformer(self):
        return True

    def init_model(self):

        # try way 1 better transformers...
        # Load meta model (no memory used)
        self.model = None

        if self.get_use_better_transformer():
            # Path 1: try BetterTransformer (if allowed and available)
            if better_transformer_available:
                try:
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
                        self.model = BetterTransformer.transform(self.model)  # enable flash attention
                except Exception as ve:
                    print(f"BetterTransformer transformation failed: {ve}")
                    self.model = None
                    clean_memory()

            # Path 2: try SDPA (Scale Dot Product Attention) - native to modern transformers
            if self.model is None:
                try:
                    print(f"Attempting native SDPA optimization...")
                    self.config.attn_implementation = "sdpa"
                    with init_empty_weights():
                        self.model = AutoModelForCausalLM.from_config(self.config, attn_implementation="sdpa", trust_remote_code=True)
                    print(f"SDPA optimization enabled.")
                except Exception as ve:
                    print(f"SDPA optimization failed: {ve}")
                    self.model = None
                    clean_memory()

        # fallback to original way
        if self.model is None:
            print(f"either BetterTransformer or attn_implementation='sdpa' is available, creating model directly")
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)

        quantization_config = getattr(self.config, "quantization_config", None)

        if quantization_config is not None:
            self.hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=True)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model = self.model, device_map = device_map)

        if self.compression is not None:
            print(f"converting model to {self.compression}...")
            self._replace_layers(self.model)

        self.model.eval()
        self.model.tie_weights()

        self.set_layers_from_layer_names()

        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.running_device, value=buffer,
                                        dtype=self.running_dtype)

        if 'rotary_pos_emb' in self.layer_names_dict:
            # for glm keep rotary_pos_emb in gpu
            self.load_rotary_pos_emb_to_device()

    def set_layers_from_layer_names(self):

        self.layers = []

        model_attr = self.model
        for attr_name in self.layer_names_dict["embed"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["layer_prefix"].split("."):
            model_attr = getattr(model_attr, attr_name)

        self.layers.extend(list(model_attr))

        model_attr = self.model
        for attr_name in self.layer_names_dict["norm"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

        model_attr = self.model
        for attr_name in self.layer_names_dict["lm_head"].split("."):
            model_attr = getattr(model_attr, attr_name)
        self.layers.append(model_attr)

    def _replace_layers(self, model, name_prefix=""):
        from bitsandbytes.nn import Linear4bit, Linear8bitLt

        for name, module in model.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            # Check if this module belongs to a high-precision layer
            is_hp = False
            for hp_idx in self.hp_layers:
                hp_prefix = f"{self.layer_names_dict['layer_prefix']}.{hp_idx}"
                if full_name.startswith(hp_prefix):
                    is_hp = True
                    break
            
            if isinstance(module, torch.nn.Linear) and not is_hp:
                if self.compression == '4bit':
                    new_module = Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=self.running_dtype, quant_type="nf4")
                elif self.compression == '8bit':
                    new_module = Linear8bitLt(module.in_features, module.out_features, bias=module.bias is not None, has_fp16_weights=False)
                setattr(model, name, new_module)
            else:
                self._replace_layers(module, full_name)

    def load_rotary_pos_emb_to_device(self):
        state_dict = load_layer(self.checkpoint_path, self.layer_names_dict['rotary_pos_emb'])
        self.move_layer_to_device(state_dict)

    def load_layer_to_cpu(self, layer_name):

        t = time.time()

        # Determine if this layer should be dequantized (e.g. if it's an hp_layer)
        dequantize = (self.compression is None)
        if not dequantize:
            for hp_idx in self.hp_layers:
                hp_prefix = f"{self.layer_names_dict['layer_prefix']}.{hp_idx}"
                if layer_name == hp_prefix:
                    dequantize = True
                    break

        load_layer_output = load_layer(self.checkpoint_path, layer_name, self.profiling_mode, dequantize=dequantize)
        elapsed_time = time.time() - t

        if self.profiling_mode:
            state_dict, compression_time = load_layer_output
            disk_loading_time = elapsed_time - compression_time

            self.profiler.add_profiling_time('load_safe_tensor', disk_loading_time)

            self.profiler.add_profiling_time('compression_time', compression_time)
        else:
            state_dict = load_layer_output

        # pin memory:
        if self.prefetching:
            t = time.time()
            if torch.cuda.is_available():  # Check if CUDA is available
                for k in state_dict.keys():
                    state_dict[k] = state_dict[k].pin_memory()
            else:
                # For CPU, no action is needed, but you could optionally add a log or message
                print("Prefetching is enabled, but no pin_memory operation is needed for CPU.")

            elapsed_time = time.time() - t
            if self.profiling_mode:
                self.profiler.add_profiling_time('pin_memory_to_trigger_load', elapsed_time)

        return state_dict

    def move_layer_to_device(self, state_dict, non_blocking=False):
        layers = []

        for param_name in state_dict.keys():
            if (self.compression == '4bit' and '.4bit.' in param_name) or (self.compression == '8bit' and '.8bit.' in param_name):
                continue

            # check if it's a quantized weight
            is_quant = False
            if self.compression == '4bit' and param_name + ".4bit.absmax" in state_dict:
                is_quant = True
                prefix = param_name + ".4bit."
                q_type = '4bit'
            elif self.compression == '8bit' and param_name + ".8bit.absmax" in state_dict:
                is_quant = True
                prefix = param_name + ".8bit."
                q_type = '8bit'

            if is_quant:
                # get module and param
                module_path = param_name.split(".")
                obj = self.model
                for i in range(len(module_path)-1):
                    obj = getattr(obj, module_path[i])
                param_attr = module_path[-1]

                # reconstitute
                qs_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

                weight_data = state_dict[param_name].to(self.running_device, non_blocking=non_blocking)

                # For 4bit
                if q_type == '4bit':
                     quant_state = bnb.functional.QuantState.from_dict(qs_dict=qs_dict, device=self.running_device)
                     getattr(obj, param_attr).data = weight_data
                     getattr(obj, param_attr).quant_state = quant_state
                else:
                    # For 8bit
                    set_module_tensor_to_device(self.model, param_name, self.running_device, value=weight_data, dtype=self.running_dtype)

                layers.append(param_name)
            else:
                 # regular param or HF quantizer
                 if (self.hf_quantizer is None or
                     not self.hf_quantizer.check_quantized_param(self.model, param_value=None, param_name=param_name, state_dict={})
                    ):
                     set_module_tensor_to_device(self.model, param_name, self.running_device, value=state_dict[param_name],
                                                dtype=self.running_dtype, non_blocking=non_blocking)
                 else:
                     self.hf_quantizer.create_quantized_param(self.model, state_dict[param_name], param_name, self.running_device, state_dict)
                 layers.append(param_name)
        return layers

    # make GenerationMixin happy
    def can_generate(self):
        return True

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = self.get_past_key_values_cache_seq_len(past_key_values) #[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_past_key_values_cache_seq_len(self, past_key_values):
        return past_key_values[0][0].shape[2]
    def get_sequence_len(self, seq):
        return seq.shape[1]

    def get_pos_emb_args(self, len_p, len_s):
        return {}

    def get_past_key_value_args(self, k_cache, v_cache):
        return {'past_key_value': (k_cache, v_cache)}

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s):
        return {'attention_mask': full_attention_mask[:, :, -len_s:, -len_p - len_s:]}

    def get_position_ids_args(self, full_position_ids, len_p, len_s):

        return {'position_ids': full_position_ids[:, len_p:len_p + len_s]}


    def run_lm_head(self, layer, seq):
        return layer(seq).float()

    def run_norm(self, layer, seq):
        return layer(seq)

    def _execute_local_layer(self, layer, layer_name, batch, i, **kwargs):
        """Helper to execute a layer locally without failover logic (used by RPC)."""
        res_batch = []
        for seq in batch:
            if layer_name == self.layer_names_dict['embed']:
                res = layer(seq)
            elif layer_name == self.layer_names_dict['norm']:
                res = self.run_norm(layer, seq)
            elif layer_name == self.layer_names_dict['lm_head']:
                res = self.run_lm_head(layer, seq)
            else:
                layer_outputs = layer(seq, **kwargs)
                res = layer_outputs[0]
            res_batch.append(res)
        return res_batch

    def _run_layer_with_resilience(self, layer, layer_name, batch, i, use_cache, 
                                    position_ids, attention_mask, output_attentions, all_self_attns, all_hidden_states):
        """Phase 8: Execution wrapper with hot-swap failover capability."""
        
        # Phase 8 & 12/13: EXECUTION MODES
        # Order: 1. REMOTE (if optimal), 2. GPU, 3. CPU (Survival)
        execution_modes = []
        
        # Phase 13: Let the NetScheduler decide if we should offload
        if self.federation and self.net_scheduler:
            peers = self.federation.get_active_peers()
            if peers:
                # Get sequence and batch info
                len_s = self.get_sequence_len(batch[0])
                batch_size = len(batch)
                
                offload_node = self.net_scheduler.decide_execution_node(
                    i, layer_name, batch_size, len_s, peers
                )
                if offload_node:
                    execution_modes.append(("REMOTE", offload_node))
        
        execution_modes.extend(["GPU", "CPU"])
        
        # Save input state for rollback if needed
        input_snapshot = [seq.detach().clone() for seq in batch]
        
        for mode_info in execution_modes:
            if isinstance(mode_info, tuple):
                mode, target_node = mode_info
            else:
                mode = mode_info
                target_node = None
                
            try:
                if mode == "REMOTE":
                    peer = target_node
                    print(f">>> AIRLLM DISTRIBUTED: Offloading layer {layer_name} to Peer {peer.node_id}")
                    
                    # Prepare kwargs for remote
                    len_s = self.get_sequence_len(batch[0])
                    kv_past = self.cache_manager.get_layer_kv(i) if use_cache else None
                    if kv_past:
                        len_p = kv_past[0].shape[2]
                        k_past, v_past = kv_past
                        # Ensure cache is on CPU for transport
                        k_past, v_past = k_past.to("cpu"), v_past.to("cpu")
                    else:
                        len_p = 0
                        k_past, v_past = None, None
                    
                    kwargs = {'use_cache': use_cache, 'output_attentions': output_attentions}
                    kwargs.update(self.get_position_ids_args(position_ids, len_p, len_s))
                    kwargs.update(self.get_attention_mask_args(attention_mask, len_p, len_s))
                    kwargs.update(self.get_pos_emb_args(len_p, len_s))
                    if kv_past:
                         kwargs.update(self.get_past_key_value_args(k_past, v_past))
                    
                    # Dispatch to remote node
                    remote_results = self.federation.dispatch_remote(peer, i, layer_name, batch, **kwargs)
                    
                    # Unpack results and update local state
                    new_batch = []
                    for j, res in enumerate(remote_results):
                        if isinstance(res, (tuple, list)):
                            # Complex layer output (hidden, kv, attentions)
                            new_seq = res[0]
                            if output_attentions:
                                all_self_attns[i].append(res[1])
                            if use_cache:
                                kv_new = res[2 if output_attentions else 1]
                                self.cache_manager.update(i, kv_new[0][..., -len_s:, :], kv_new[1][..., -len_s:, :])
                            new_batch.append(new_seq)
                        else:
                            # Simple layer output (embed, norm, etc)
                            new_batch.append(res)
                    
                    # Move results back to local running device
                    new_batch = [r.to(self.running_device) for r in new_batch]
                    
                    # Phase 9: Numerical Integrity Check
                    for r in new_batch:
                        if torch.isnan(r).any() or torch.isinf(r).any():
                             raise ValueError("Numerical instability in remote output")
                             
                    return new_batch

                elif mode == "CPU":
                    print(f">>> AIRLLM FAILOVER: Transitioning layer {layer_name} to CPU-only survival mode.")
                    layer.to("cpu")
                    batch = [seq.to("cpu") for seq in batch]
                    # We might need to move position_ids/mask too if they are used
                    pos_ids_mode = position_ids.to("cpu")
                    att_mask_mode = attention_mask.to("cpu")
                else:
                    pos_ids_mode = position_ids
                    att_mask_mode = attention_mask

                # Execute
                new_batch = []
                for j, seq in enumerate(batch):
                    if layer_name == self.layer_names_dict['embed']:
                        res = layer(seq)
                    elif layer_name == self.layer_names_dict['norm']:
                        res = self.run_norm(layer, seq)
                        if output_attentions and all_hidden_states is not None:
                             all_hidden_states[i].append(res)
                    elif layer_name == self.layer_names_dict['lm_head']:
                        res = self.run_lm_head(layer, seq)
                    else:
                        len_s = self.get_sequence_len(seq)
                        kv_past = self.cache_manager.get_layer_kv(i) if use_cache else None

                        if kv_past:
                            k_past, v_past = kv_past
                            len_p = k_past.shape[2]
                        else:
                            len_p = 0
                            k_past, v_past = None, None

                        # Make sure cache is on correct device if in CPU mode
                        if mode == "CPU" and kv_past:
                            # Cache manager usually keeps it on CPU or designated device
                            # but we ensure consistency here
                            k_past = k_past.to("cpu")
                            v_past = v_past.to("cpu")

                        kwargs = {'use_cache': use_cache}
                        kwargs.update(self.get_position_ids_args(pos_ids_mode, len_p, len_s))
                        kwargs.update(self.get_attention_mask_args(att_mask_mode, len_p, len_s))
                        kwargs.update(self.get_pos_emb_args(len_p, len_s))

                        if kv_past:
                             kwargs.update(self.get_past_key_value_args(k_past, v_past))

                        layer_outputs = layer(seq, **kwargs)
                        res = layer_outputs[0]

                        if output_attentions:
                            all_self_attns[i].append(layer_outputs[1])

                        if use_cache:
                            kv_new = layer_outputs[2 if output_attentions else 1]
                            self.cache_manager.update(i, kv_new[0][..., -len_s:, :], kv_new[1][..., -len_s:, :])

                    new_batch.append(res)
                
                # If we were in CPU mode, move result back to target device for next layers
                if mode == "CPU":
                    new_batch = [r.to(self.running_device) for r in new_batch]
                
                # Phase 9: Numerical Integrity - Check for NaNs
                for r in new_batch:
                    if torch.isnan(r).any() or torch.isinf(r).any():
                        print(f"!!! AIRLLM INTEGRITY: NaN/Inf detected in layer {layer_name} output ({mode} mode).")
                        raise ValueError("Numerical instability detected")

                return new_batch

            except (torch.cuda.OutOfMemoryError, Exception) as e:
                if mode == "CPU":
                    raise e # Terminal fail
                
                print(f"!!! AIRLLM RECOVERY: Hardware exception in {mode} mode: {e}")
                
                # Notify optimizer of the failure to adjust global policy for future layers
                self.optimizer.adjust_for_pressure({'vram_usage_pct': 95, 'error': str(e)}) # Assume high pressure
                self.policy = self.optimizer.get_policy() # Update local policy reference
                
                clean_memory()
                torch.cuda.empty_cache()
                # Restore state for retry
                batch = [s.detach().clone() for s in input_snapshot]
                continue

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        if self.profiling_mode:
            self.profiler.clear_profiling_time()

            forward_start = time.process_time()
            forward_start_wall = time.time()

        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()

        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        kv_cache_list = [] if use_cache else None
        if use_cache:
            if self.cache_manager is None:
                # Initialize cache manager: layers are (core layers + embed + norm + head)
                # But KV cache is only for transformer layers.
                # Actually, self.layers contains all.
                # Transformer layers are the ones between embed and norm.
                self.cache_manager = AirLLMCacheManager(len(self.layers), self.config.num_attention_heads, 
                                                        self.config.hidden_size // self.config.num_attention_heads,
                                                        device=self.running_device, dtype=self.running_dtype,
                                                        use_8bit_cache=self.policy.cache_8bit)

            # if past_key_values is provided, it means we are continuing.
            # AirLLM current logic for past_key_values is a bit different.
            # Let's assume we use cache_manager to manage EVERYTHING.
            if past_key_values is None:
                self.cache_manager.clean()

        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        with torch.inference_mode(), ThreadPoolExecutor() as executor:

            if self.prefetching:
                # Pre-load first layer
                state_dict_curr = self.load_layer_to_cpu(self.layer_names[0])
                if len(self.layer_names) > 1:
                    future = executor.submit(self.load_layer_to_cpu, self.layer_names[1])
                
                # Start moving first layer to GPU
                if self.stream:
                    with torch.cuda.stream(self.stream):
                        moved_layers_curr = self.move_layer_to_device(state_dict_curr, non_blocking=True)
                else:
                    moved_layers_curr = self.move_layer_to_device(state_dict_curr)

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)),
                                               desc=f'running layers({self.running_device})',
                                               total=len(self.layers)):

                if self.prefetching:
                    if self.profiling_mode:
                        t = time.time()
                    
                    # Wait for THIS layer to be ready on GPU
                    if self.stream:
                        torch.cuda.current_stream().wait_stream(self.stream)
                    
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('load_gpu_wait', elapsed_time)

                    moved_layers = moved_layers_curr

                    # Kick off NEXT move and load
                    if (i + 1) < len(self.layer_names):
                        state_dict_curr = future.result()
                        if (i + 2) < len(self.layer_names):
                            future = executor.submit(self.load_layer_to_cpu, self.layer_names[i+2])
                        
                        if self.stream:
                            with torch.cuda.stream(self.stream):
                                moved_layers_curr = self.move_layer_to_device(state_dict_curr, non_blocking=True)
                        else:
                            moved_layers_curr = self.move_layer_to_device(state_dict_curr)

                else:
                    state_dict = self.load_layer_to_cpu(layer_name)
                    if self.profiling_mode:
                        t = time.time()
                    moved_layers = self.move_layer_to_device(state_dict)
                    if self.profiling_mode:
                        elapsed_time = time.time() - t
                        self.profiler.add_profiling_time('create_layer_from_safe_tensor', elapsed_time)

                # Run layer

                # Run layer with Failover Resilience (Phase 8)
                try:
                    batch = self._run_layer_with_resilience(
                        layer, layer_name, batch, i, use_cache, 
                        position_ids, attention_mask, output_attentions, all_self_attns, all_hidden_states
                    )
                except Exception as e:
                    print(f"!!! FATAL ERROR during layer {layer_name} execution: {e}")
                    raise e

                if output_hidden_states:
                    all_hidden_states += (torch.cat(batch, 0),)

                # Remove previous layer from memory (including buffers)

                if self.hf_quantizer is not None:
                    for param_name in moved_layers:#param_name, param in state_dict.items():
                        set_module_tensor_to_device(self.model, param_name,'meta')
                else:
                    layer.to("meta")

                layer.to("meta")
                clean_memory()  # proposed by CPMP

        logits = torch.cat(batch, 0)
        if use_cache:
            kv_cache_list = []
            for i in range(len(self.layers)):
                kv = self.cache_manager.get_layer_kv(i)
                if kv:
                    kv_cache_list.append(kv)
            # kv_cache_list = kv_cache_list[1:-2] # No longer needed as we only store what's needed
            #print(f"returning kvcache size: {kv_cache_list[0][0].shape}")

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        if not return_dict:
            return tuple(v for v in [logits,
                                     tuple(kv_cache_list) if kv_cache_list is not None else None,
                                     tuple(all_hidden_states) if all_hidden_states is not None else None,
                                     tuple(all_self_attns) if all_self_attns is not None else None] if v is not None)
        if self.profiling_mode:
            forward_elapsed_time = time.process_time() - forward_start
            forward_elapsed_time_wall = time.time() - forward_start_wall
            self.profiler.print_profiling_time()


            print(f"total infer process time(including all above plus gpu compute): {forward_elapsed_time:.04f}")
            print(f"total infer wall time(including all above plus gpu compute): {forward_elapsed_time_wall:.04f}")

            self.profiler.clear_profiling_time()


        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list is not None else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_self_attns) if all_hidden_states is not None else None,
        )