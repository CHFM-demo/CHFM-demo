"""Feature memory for the final CHFM objective.

The memory stores one running low-rank feature factor per Householder module;
raw training examples are never retained.
"""
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def input_feature_drift(u, v, factor):
    """Relative ||R U - U||_F^2 for R=H_u H_v."""
    u, v = F.normalize(u.float(), dim=0), F.normalize(v.float(), dim=0)
    factor = factor.detach().to(device=u.device, dtype=torch.float32)
    # R U = H_u(H_v U): apply v first to match W_new = W H_u H_v.
    delta_v = -2 * v @ (v.T @ factor)
    delta = delta_v - 2 * u @ (u.T @ (factor + delta_v))
    return delta.square().sum() / factor.square().sum().clamp_min(1e-12)


class RepresentationMemory:
    def __init__(self, model, rank=32, max_tokens=512):
        if rank < 1 or max_tokens < rank:
            raise ValueError("Feature memory requires max_tokens >= rank >= 1")
        self.model_type = getattr(getattr(model, "config", None), "model_type", None)
        self.sampling_strategy = (
            "llama_balanced_source_target_v1"
            if self.model_type == "llama" else "all_valid_v1"
        )
        self.factors = {}
        self.task_ids = []
        self.rank, self.max_tokens = rank, max_tokens
        self.layers = {
            name: layer for name, layer in model.named_modules()
            if hasattr(layer, "rotatenew_U")
            and name.rsplit(".", 1)[-1] in ("q", "v", "q_proj", "v_proj")
        }
        if not self.layers:
            raise ValueError("Feature memory found no Householder Q/V projections")
        if any(layer.fan_in_fan_out for layer in self.layers.values()):
            raise ValueError("Feature memory requires ordinary linear weight layout")
        self._device_factors = {}

    @property
    def task_count(self):
        return len(self.task_ids)

    def has_task(self, task_id):
        return task_id in self.task_ids

    def _validate_factors(self, factors):
        if set(factors) != set(self.layers):
            raise ValueError("Feature-memory module names do not match the loaded model")
        for name, factor in factors.items():
            if (factor.ndim != 2 or factor.shape[0] != self.layers[name].in_features
                    or factor.shape[1] < 1):
                raise ValueError("Invalid feature factor shape for " + name)
            if not torch.isfinite(factor).all():
                raise ValueError("Nonfinite feature factor for " + name)

    def load(self, path):
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "householder_feature_memory_v1":
            raise ValueError("Unsupported feature-memory format")
        saved_strategy = payload.get("sampling_strategy", "all_valid_v1")
        if saved_strategy != self.sampling_strategy:
            raise ValueError(
                "Feature-memory sampling strategy does not match this model: saved "
                f"{saved_strategy}, expected {self.sampling_strategy}. "
                "Start the Llama sequence again to build balanced source/target memory."
            )
        ids, factors = payload["task_ids"], payload["factors"]
        if (not isinstance(ids, list) or not all(isinstance(x, str) for x in ids)
                or len(set(ids)) != len(ids) or payload["task_count"] != len(ids)):
            raise ValueError("Invalid feature-memory task metadata")
        if payload["rank"] != self.rank:
            raise ValueError("Feature-memory rank must match the saved memory")
        if ids:
            self._validate_factors(factors)
        elif factors:
            raise ValueError("Feature factors require task history")
        self.task_ids, self.factors = ids, factors
        self._device_factors.clear()

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {"format": "householder_feature_memory_v1",
                   "sampling_strategy": self.sampling_strategy,
                   "rank": self.rank, "task_count": self.task_count,
                   "task_ids": self.task_ids, "factors": self.factors}
        torch.save(payload, path)

    @torch.no_grad()
    def add_task(self, task_id, factors):
        """Commit collected factors into an equal-task-weight running memory."""
        if self.has_task(task_id):
            raise ValueError("Feature-memory task already exists: " + task_id)
        self._validate_factors(factors)
        factors = {name: value.detach().float().cpu().clone()
                   for name, value in factors.items()}
        count = self.task_count
        merged = {}
        for name, current in factors.items():
            if count:
                current = torch.cat(((count / (count + 1)) ** 0.5 * self.factors[name],
                                     (1 / (count + 1)) ** 0.5 * current), dim=1)
            if current.shape[1] > self.rank:
                left, singular, _ = torch.linalg.svd(current, full_matrices=False)
                current = left[:, :self.rank] * singular[:self.rank]
            merged[name] = current.contiguous()
        self.factors = merged
        self.task_ids.append(task_id)
        self._device_factors.clear()

    def loss(self):
        losses = []
        histories = (("feature_memory", self.factors),) if self.task_count else ()
        for task, factors in histories:
            for name, layer in self.layers.items():
                if not layer.merged:
                    raise RuntimeError("Merge historical adapters before computing feature loss")
                adapter = layer.active_adapter
                key = (task, name, layer.weight.device)
                if key not in self._device_factors:
                    self._device_factors[key] = factors[name].to(layer.weight.device)
                losses.append(input_feature_drift(
                    layer.rotatenew_U[adapter],
                    layer.rotatenew_V[adapter], self._device_factors[key]))
        return torch.stack(losses).mean() if losses else None

    @torch.no_grad()
    def collect(self, model, batches, prepare_inputs, task_id, max_batches=32,
                forward_context=nullcontext):
        """Bounded random token sample, then truncated uncentered SVD.

        Call in eval mode after training. CPU storage is bounded per module.
        Decoder inputs use the shifted label mask when no explicit mask exists.
        forward_context supplies the training precision context for each forward;
        sampled features and SVD remain FP32.
        """
        if self.has_task(task_id):
            raise ValueError("Feature-memory task already exists: " + task_id)
        if max_batches < 1:
            raise ValueError("Feature-memory max_batches must be positive")
        samples, priorities, handles = {}, {}, []
        masks = {}
        generator = torch.Generator().manual_seed(0)

        def hook(name):
            def capture(layer, args):
                x = args[0].detach()
                decoder = "decoder" in name and not (
                    "EncDecAttention" in name and name.endswith(".v"))
                mask = masks["decoder" if decoder else "encoder"]
                if mask is None or tuple(mask.shape) != tuple(x.shape[:-1]):
                    raise ValueError("Cannot determine valid-token mask for " + name)
                masks_to_sample = {"all": mask.bool()}
                if self.sampling_strategy == "llama_balanced_source_target_v1":
                    target = masks.get("target")
                    labels = masks.get("labels")
                    if (target is None or labels is None
                            or tuple(target.shape) != tuple(x.shape[:-1])
                            or tuple(labels.shape) != tuple(x.shape[:-1])):
                        raise ValueError("Cannot determine Llama source/target masks for " + name)
                    target = target.bool() & mask.bool()
                    # Keep source context and supervised prediction positions separate.
                    # Answer states that predict the next answer token belong to target;
                    # the final answer/EOS position is not used because it predicts no
                    # supervised token.
                    source = mask.bool() & labels.eq(-100) & ~target
                    masks_to_sample = {"source": source, "target": target}
                for group, sample_mask in masks_to_sample.items():
                    selected = x[sample_mask].float().cpu()
                    if selected.shape[0] == 0:
                        continue
                    key = (name, group)
                    random_keys = torch.rand(selected.shape[0], generator=generator)
                    if key in samples:
                        selected = torch.cat((samples[key], selected))
                        random_keys = torch.cat((priorities[key], random_keys))
                    keep = random_keys.topk(min(self.max_tokens, random_keys.numel())).indices
                    samples[key], priorities[key] = selected[keep], random_keys[keep]
            return capture

        was_training = model.training
        try:
            model.eval()
            for name, layer in self.layers.items():
                handles.append(layer.register_forward_pre_hook(hook(name)))
            for index, batch in enumerate(batches):
                if index >= max_batches:
                    break
                inputs = prepare_inputs(batch)
                masks["encoder"] = inputs.get("attention_mask")
                masks["decoder"] = inputs.get("decoder_attention_mask")
                masks["labels"] = inputs.get("labels")
                masks["target"] = inputs.get("loss_mask")
                if masks["decoder"] is None and "labels" in inputs:
                    valid = inputs["labels"].ne(-100)
                    masks["decoder"] = torch.cat((torch.ones_like(valid[:, :1]), valid[:, :-1]), dim=1)
                with forward_context():
                    model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)

        factors = {}
        # Isolate randomized SVD from the training RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            for name in self.layers:
                if self.sampling_strategy == "llama_balanced_source_target_v1":
                    source = samples.get((name, "source"))
                    target = samples.get((name, "target"))
                    if source is None or target is None:
                        raise ValueError("No Llama source/target feature tokens collected for " + name)
                    target_count = min(target.shape[0], self.max_tokens // 2)
                    source_count = min(source.shape[0], self.max_tokens - target_count)
                    # If source is unusually short, use available target positions to
                    # fill the single factor without exceeding max_tokens.
                    target_count = min(target.shape[0], self.max_tokens - source_count)
                    x = torch.cat((source[:source_count], target[:target_count]))
                else:
                    x = samples.get((name, "all"))
                if x is None or x.shape[0] == 0:
                    raise ValueError("No valid feature tokens collected for " + name)
                q = min(self.rank + 8, min(x.shape))
                _, singular, vectors = torch.svd_lowrank(x, q=q, niter=2)
                r = min(self.rank, singular.numel())
                factors[name] = (vectors[:, :r] * singular[:r] / x.shape[0] ** 0.5).contiguous()
        self.add_task(task_id, factors)
