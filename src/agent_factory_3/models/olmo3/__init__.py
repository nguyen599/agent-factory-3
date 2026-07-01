"""OLMo3 model loading helpers for agent-factory training."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_olmo3_causal_lm(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
    use_attention_sink: bool = False,
    sink_cache_dir: str | None = None,
    sink_init_value: float = -10.0,
    sinks_npz: str | None = None,
    sink_force_convert: bool = False,
    sink_dtype: str = "bfloat16",
    apply_liger: bool = False,
):
    """Load an OLMo3 causal LM for Accelerate/FSDP training."""
    active_model_path = model_path
    if use_attention_sink:
        if not sink_cache_dir:
            raise ValueError("sink_cache_dir is required when use_attention_sink=True")
        from .sink_support import prepare_olmo3_sink_model_path

        active_model_path = prepare_olmo3_sink_model_path(
            model_path,
            cache_dir=sink_cache_dir,
            sink_init_value=sink_init_value,
            dtype=sink_dtype,
            sinks_npz=sinks_npz,
            force_convert=sink_force_convert,
        )
        logger.info("Loading OLMo3 attention-sink model: %s", active_model_path)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            active_model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
    except Exception:
        if not use_attention_sink or attn_implementation == "eager":
            raise
        logger.exception(
            "Failed to load OLMo3 sink with attn_implementation=%s; retrying eager backend",
            attn_implementation,
        )
        model = AutoModelForCausalLM.from_pretrained(
            active_model_path,
            dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
    if apply_liger:
        try:
            from olmo3_sink import apply_liger as apply_olmo3_sink_liger
        except Exception:
            logger.exception("OLMo3 sink Liger patch requested but unavailable")
            raise
        apply_olmo3_sink_liger(model)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model


__all__ = ["load_olmo3_causal_lm"]
