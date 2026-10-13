"""Model loading shared by the Householder entry point and offline smoke tests."""
import os

import torch

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer
from peft import HouseholderConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from model.llama import LlamaForCausalLM_with_lossmask


def load_model_and_tokenizer(args, training_args=None):
    path = args.model_name_or_path
    # Local adapters need not be named "adapter" or contain the base model type.
    is_adapter = os.path.isfile(os.path.join(path, "adapter_config.json"))
    adapter_config = PeftConfig.from_pretrained(path) if is_adapter else None
    base_path = adapter_config.base_model_name_or_path if is_adapter else path
    load_kwargs = dict(cache_dir=args.cache_dir, revision=args.model_revision,
                       use_auth_token=True if args.use_auth_token else None)
    config = AutoConfig.from_pretrained(args.config_name or base_path, **load_kwargs)
    is_llama = config.model_type == "llama"
    task_type = TaskType.CAUSAL_LM if is_llama else TaskType.SEQ_2_SEQ_LM
    tokenizer_class = LlamaTokenizer if is_llama else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name or base_path, use_fast=args.use_fast_tokenizer, **load_kwargs)
    if is_llama:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.bos_token or tokenizer.eos_token
        config.pad_token_id = tokenizer.pad_token_id

    model_class = LlamaForCausalLM_with_lossmask if is_llama else AutoModelForSeq2SeqLM
    # Preserve the original T5 loading precision; Llama scripts default to BF16.
    if is_llama and training_args is not None:
        load_kwargs["torch_dtype"] = (torch.bfloat16 if training_args.bf16 else
                                      torch.float16 if training_args.fp16 else torch.float32)
    model = model_class.from_pretrained(base_path, config=config, **load_kwargs)
    if is_adapter:
        if adapter_config.task_type != task_type:
            raise ValueError(
                f"Adapter task_type={adapter_config.task_type} is incompatible with "
                f"{config.model_type}; expected {task_type}. Use a compatible adapter.")
        model = PeftModel.from_pretrained(model, path)
    else:
        peft_config = HouseholderConfig(task_type=task_type, inference_mode=False, r_sum=args.r_sum)
        model = get_peft_model(model, peft_config)
    model.resize_token_embeddings(len(tokenizer))
    if is_llama:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
