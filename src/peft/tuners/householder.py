# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class HouseholderConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`HouseholderModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
                    "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
                    "For example, in Sequence Classification or Token Classification tasks, "
                    "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the Lora layers."},
    )
    r_sum: int = field(default=0)  # modified. This argument represents the dim of the previous LoRA parameters.
    total_step: Optional[int] = field(default=None, metadata={"help": "The total training steps."})
    global_step: Optional[int] = field(default=None, metadata={"help": "The global step."})
    save_loranew: bool = field(
        default=False)  # modified. This arguments represents whether modules named of 'loranew_A/B' are saved independently, rather than being combined with "lora_A/B".

    def __post_init__(self):
        self.peft_type = PeftType.HOUSEHOLDER


class StepAllocator(object):
    def __init__(self, peft_config, adapter_name):
        self.peft_config = peft_config
        self.adapter_name = adapter_name

    def set_total_step(self, total_step):
        self.peft_config[self.adapter_name].total_step = total_step

    def set_global_step(self, global_step):
        self.peft_config[self.adapter_name].global_step = global_step


class HouseholderModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`HouseholderConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, HouseholderConfig
        >>> from peft import HouseholderModel, HouseholderConfig

        >>> config = HouseholderConfig(
        ...     peft_type="LORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = HouseholderModel(config, model)
        ```

        ```py
        >>> import transformers
        >>> from peft import HouseholderConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = HouseholderConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`HouseholderConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__()
        self.model = model
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])
        self.log_sigma1 = torch.nn.Parameter(torch.tensor(0.0))
        self.log_sigma2 = torch.nn.Parameter(torch.tensor(0.0))  # 0.25
        self.adapter_name = adapter_name
        self.step_allocator = StepAllocator(self.peft_config, adapter_name)

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)
        memory = getattr(self, "feature_memory", None)
        if self.training and memory is not None and getattr(outputs, "loss", None) is not None:
            feature_loss = memory.loss()
            if feature_loss is not None:
                outputs.loss = (
                    0.5 * torch.exp(-2 * self.log_sigma1) * outputs.loss
                    + 0.5 * torch.exp(-2 * self.log_sigma2) * feature_loss
                    + self.log_sigma1 + self.log_sigma2
                )
        return outputs

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "HouseholderModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                if hasattr(target, "bias"):
                    bias = target.bias is not None

                if isinstance(target, HouseholderLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_lora_weights,
                        lora_config.r_sum
                    )
                else:
                    if isinstance(target, torch.nn.Linear):
                        in_features, out_features = target.in_features, target.out_features
                        if kwargs["fan_in_fan_out"]:
                            warnings.warn(
                                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                "Setting fan_in_fan_out to False."
                            )
                            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                    else:
                        raise ValueError(
                            f"Target module {target} is not supported. "
                            f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                        )
                    new_module = Linear(adapter_name, in_features, out_features, bias=bias, r_sum=lora_config.r_sum,
                                        **kwargs)  # modified

                    self._replace_module(parent, target_name, new_module, target)
                # Preserve the original RNG stream for existing T5 experiments.
                getattr(parent, target_name).use_torch_rng = self.model.config.model_type != "t5"
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "rotate_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, HouseholderLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, HouseholderLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.active_adapter = adapter_name

    def merge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, HouseholderLayer):
                module.merge()

    def unmerge_adapter(self):
        for module in self.model.modules():
            if isinstance(module, HouseholderLayer):
                module.unmerge()

    @staticmethod
    def _prepare_lora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def merge_and_unload(self):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        """
        if getattr(self.config, "model_type", None) == "gpt2":
            raise ValueError("GPT2 models are not supported for merging LORA layers")

        if getattr(self.model, "is_loaded_in_8bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, HouseholderLayer):
                bias = target.bias is not None
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def add_weighted_adapter(self, adapters, weights, adapter_name):
        self.peft_config[adapter_name] = self.peft_config[adapters[0]]
        self._find_and_replace(adapter_name)
        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        _freeze_adapter(self.model, adapter_name)
        key_list = [key for key, _ in self.model.named_modules() if "rotate" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "rotate_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, HouseholderLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class HouseholderLayer:
    def __init__(
            self,
            in_features: int,
            out_features: int,
    ):
        self.rotate_U = nn.ParameterDict({})
        self.rotate_V = nn.ParameterDict({})
        self.rotatenew_U = nn.ParameterDict({})  # modified
        self.rotatenew_V = nn.ParameterDict({})  # modified
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.use_torch_rng = True
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, init_lora_weights, r_sum):  # modified
        # Actual trainable parameters
        self.rotatenew_U.update(
            nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.in_features, 1))}))  # modified
        self.rotatenew_V.update(
            nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.in_features, 1))}))  # modified
        self.rotate_U.update(
            nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.in_features, r_sum))}))  # modified
        self.rotate_V.update(
            nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.in_features, r_sum))}))  # modified
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):

        # modified
        if adapter_name in self.rotate_U.keys():
            nn.init.zeros_(self.rotate_U[adapter_name])
            nn.init.zeros_(self.rotate_V[adapter_name])

        # modified
        if adapter_name in self.rotatenew_U.keys():
            nn.init.normal_(self.rotatenew_U[adapter_name], mean=0.0, std=1.0)
            from copy import deepcopy
            self.rotatenew_V[adapter_name].data = deepcopy(self.rotatenew_U[adapter_name].data)


class Linear(nn.Linear, HouseholderLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            adapter_name: str,
            in_features: int,
            out_features: int,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            r_sum: int = 0,  # modified
            **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        HouseholderLayer.__init__(self, in_features=in_features, out_features=out_features)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False
        self.r_sum = r_sum
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, init_lora_weights, r_sum)  # modified
        self.active_adapter = adapter_name

    def merge(self):
        if self.active_adapter not in self.rotate_U.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        pre_dtype = self.weight.data.dtype
        if self.r_sum > 0:
            with torch.no_grad():
                weight = self.weight.data.clone()
                raw_U = F.normalize(self.rotate_U[self.active_adapter], dim=0).to(weight.dtype)
                raw_V = F.normalize(self.rotate_V[self.active_adapter], dim=0).to(weight.dtype)
                for i in range(self.r_sum):
                    weight = weight - 2 * torch.matmul(
                        torch.matmul(weight, raw_U[:, i].unsqueeze(1)),
                        raw_U[:, i].unsqueeze(1).T)
                    weight = weight - 2 * torch.matmul(
                        torch.matmul(weight, raw_V[:, i].unsqueeze(1)),
                        raw_V[:, i].unsqueeze(1).T)
                self.weight.data.copy_(weight)
        self.merged = True

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype

        if self.active_adapter not in self.rotate_U.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.active_adapter and not self.merged:
            if self.r_sum > 0:
                raw_U = F.normalize(self.rotate_U[self.active_adapter], dim=0)  # n*r
                raw_V = F.normalize(self.rotate_V[self.active_adapter], dim=0)  # n*r
                for i in range(self.r_sum):
                    if i == 0:
                        W = self.weight - 2 * torch.matmul(torch.matmul(self.weight, raw_U[:, i].unsqueeze(1)),
                                                           raw_U[:, i].unsqueeze(1).T)
                        W = W - 2 * torch.matmul(torch.matmul(W, raw_V[:, i].unsqueeze(1)), raw_V[:, i].unsqueeze(1).T)
                    else:
                        W = W - 2 * torch.matmul(torch.matmul(W, raw_U[:, i].unsqueeze(1)), raw_U[:, i].unsqueeze(1).T)
                        W = W - 2 * torch.matmul(torch.matmul(W, raw_V[:, i].unsqueeze(1)), raw_V[:, i].unsqueeze(1).T)
            else:
                W = self.weight
            if self.training:
                dropout_rate = 0.1
                # Torch RNG is checkpoint-safe; T5 retains its original Python RNG.
                p = torch.rand((), device=x.device).item() if self.use_torch_rng else random.random()
                if p < dropout_rate:
                    raw_U = F.normalize(self.rotatenew_U[self.active_adapter], dim=0)
                    W = W - 2 * torch.matmul(torch.matmul(W, raw_U), raw_U.T)
                    result = F.linear(x, transpose(W, self.fan_in_fan_out), bias=self.bias)
                else:
                    raw_U = F.normalize(self.rotatenew_U[self.active_adapter], dim=0)
                    raw_V = F.normalize(self.rotatenew_V[self.active_adapter], dim=0)
                    W = W - 2 * torch.matmul(torch.matmul(W, raw_U), raw_U.T)
                    W = W - 2 * torch.matmul(torch.matmul(W, raw_V), raw_V.T)
                    result = F.linear(x, transpose(W, self.fan_in_fan_out), bias=self.bias)
            else:
                raw_U = F.normalize(self.rotatenew_U[self.active_adapter], dim=0)
                raw_V = F.normalize(self.rotatenew_V[self.active_adapter], dim=0)
                W = W - 2 * torch.matmul(torch.matmul(W, raw_U), raw_U.T)
                W = W - 2 * torch.matmul(torch.matmul(W, raw_V), raw_V.T)
                result = F.linear(x, transpose(W, self.fan_in_fan_out), bias=self.bias)
        elif self.merged:
            if self.training:
                dropout_rate = 0.01
                # Torch RNG is checkpoint-safe; T5 retains its original Python RNG.
                p = torch.rand((), device=x.device).item() if self.use_torch_rng else random.random()
                if p < dropout_rate:
                    raw_U = F.normalize(self.rotatenew_U[self.active_adapter], dim=0)
                    W = self.weight - 2 * torch.matmul(torch.matmul(self.weight, raw_U), raw_U.T)
                    result = F.linear(x, transpose(W, self.fan_in_fan_out), bias=self.bias)
                else:
                    raw_U = F.normalize(self.rotatenew_U[self.active_adapter], dim=0)
                    raw_V = F.normalize(self.rotatenew_V[self.active_adapter], dim=0)
                    W = self.weight - 2 * torch.matmul(torch.matmul(self.weight, raw_U), raw_U.T)
                    W = W - 2 * torch.matmul(torch.matmul(W, raw_V), raw_V.T)
                    result = F.linear(x, transpose(W, self.fan_in_fan_out), bias=self.bias)
            else:
                raw_U = F.normalize(self.rotatenew_U[self.active_adapter], dim=0)
                raw_V = F.normalize(self.rotatenew_V[self.active_adapter], dim=0)
                W = self.weight - 2 * torch.matmul(torch.matmul(self.weight, raw_U), raw_U.T)
                W = W - 2 * torch.matmul(torch.matmul(W, raw_V), raw_V.T)
                result = F.linear(x, transpose(W, self.fan_in_fan_out), bias=self.bias)
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
