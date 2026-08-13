"""Model construction for LoRA fine-tuning.

Loads a Hugging Face causal-LM base and wraps it with a PEFT LoRA adapter. PEFT and
transformers are imported lazily inside build_finetune_model so this module is importable
on machines that don't have those packages -- important for CPU-only unit tests, which
never call build_finetune_model.

The analytical-count and layer-selection functions ARE pure Python and are always
importable, so tests can pin the LoRA arithmetic without pulling PEFT into CI.
"""
import re


def _resolve_torch_dtype(base_storage_precision):
    """base_storage_precision -> torch.dtype for AutoModelForCausalLM.from_pretrained.

    Import torch inside the function so this module stays importable on the (extremely
    thin) test path that doesn't need it. int8/int4 flow through different
    from_pretrained kwargs (load_in_8bit/load_in_4bit or a BitsAndBytesConfig), handled
    below.
    """
    import torch
    return {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }.get(base_storage_precision)


def _layers_to_transform(n_layers_total, adapter_layers):
    """Adapter-placement config -> list of layer indices for LoraConfig.layers_to_transform.

    'all' -> None (PEFT convention: no restriction).
    'upper-N' -> the top N indices (n_layers_total - N .. n_layers_total - 1).
    """
    if adapter_layers == "all":
        return None
    m = re.match(r"upper-(\d+)$", adapter_layers)
    if not m:
        raise ValueError(f"unknown adapter_layers: {adapter_layers!r}")
    n = min(int(m.group(1)), n_layers_total)
    return list(range(n_layers_total - n, n_layers_total))


def build_finetune_model(config):
    """Load the base, apply LoRA, return (model, base_arch_dict).

    base_arch_dict carries the architecture values sweep_config_finetune deliberately
    does NOT store on Config -- they're a function of the base model, not a lever.

    ONLY call this on CUDA. transformers + peft + huggingface_hub must be installed.
    """
    # Lazy imports: nothing above this line depends on the HF stack.
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    base_arch = AutoConfig.from_pretrained(config.base_model_name)

    # Build the base with the requested storage precision. int8/int4 need bitsandbytes and
    # a separate loading path; those config rows are gated at parent-startup in
    # benchmark_finetune.py so this branch only sees precisions it can handle here.
    if config.base_storage_precision in ("int8", "int4"):
        # BitsAndBytesConfig accepts the specific precision as a flag; this path only
        # runs if bitsandbytes was importable at parent-startup, so the import here is
        # safe.
        from transformers import BitsAndBytesConfig  # noqa: WPS433
        quant = BitsAndBytesConfig(
            load_in_8bit=(config.base_storage_precision == "int8"),
            load_in_4bit=(config.base_storage_precision == "int4"),
        )
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            torch_dtype=_resolve_torch_dtype(config.base_storage_precision),
        )

    # Determine layer count from the loaded config so upper-N resolves against the real
    # architecture, not a hard-coded value.
    n_layers_total = base_arch.num_hidden_layers
    layers_to_transform = _layers_to_transform(n_layers_total, config.lora_adapter_layers)

    # LoRA wrap. bias='none' matches the field convention and keeps the trainable set
    # cleanly equal to adapter A/B factors -- no bias params sneak into the trainable
    # pool, so measured_trainable_param_bytes reads as the adapter total.
    from peft import LoraConfig, get_peft_model  # noqa: WPS433

    lora_kwargs = dict(
        r=config.lora_rank,
        lora_alpha=config.lora_rank * 2,  # common default: alpha = 2r
        target_modules=list(config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(base, lora_config)

    # Explicit assertion of the requires_grad split -- fail loudly here rather than let a
    # PEFT version regression silently unfreeze the base.
    _assert_only_lora_params_trainable(model)

    return model, {
        "num_hidden_layers": n_layers_total,
        "hidden_size": base_arch.hidden_size,
        "num_attention_heads": base_arch.num_attention_heads,
        "num_key_value_heads": getattr(base_arch, "num_key_value_heads", base_arch.num_attention_heads),
        "intermediate_size": base_arch.intermediate_size,
        "vocab_size": base_arch.vocab_size,
        "attention_dropout": getattr(base_arch, "attention_dropout", 0.0),
        "hidden_dropout": getattr(base_arch, "hidden_dropout", 0.0),
        "n_layers_needing_backward": len(layers_to_transform) if layers_to_transform else n_layers_total,
    }


def _assert_only_lora_params_trainable(model):
    """Guard: every trainable parameter must be a LoRA parameter (name contains 'lora_'),
    and every base weight must be frozen. Fails loudly if PEFT's default behavior
    regresses in a way that would silently double the trainable count."""
    for name, p in model.named_parameters():
        is_lora = "lora_" in name
        if p.requires_grad and not is_lora:
            raise RuntimeError(f"non-LoRA parameter {name!r} has requires_grad=True")
        if not p.requires_grad and is_lora:
            raise RuntimeError(f"LoRA parameter {name!r} has requires_grad=False")


def count_trainable_params(model):
    """Same shape as model.count_parameters(), filtered to requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_params(model):
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def analytical_param_count_finetune(base_arch, lora_rank, target_modules, adapter_layers):
    """LoRA arithmetic: each decomposed linear of shape (d_in, d_out) adds r*(d_in + d_out)
    trainable parameters. Returns dict with n_base, n_adapter, n_layers_needing_backward.

    base_arch is either the dict returned by build_finetune_model or the equivalent
    (hidden_size, num_attention_heads, num_key_value_heads, intermediate_size,
    num_hidden_layers, vocab_size). Signature accepts a dict for both to keep tests
    simple.
    """
    d = base_arch["hidden_size"]
    nh = base_arch["num_attention_heads"]
    kvh = base_arch["num_key_value_heads"]
    ff = base_arch["intermediate_size"]
    L = base_arch["num_hidden_layers"]

    # In a Llama-family GQA architecture:
    #   q_proj: (d, d)                    -- Q keeps full width
    #   k_proj: (d, d * kvh/nh)           -- K/V narrowed by GQA
    #   v_proj: (d, d * kvh/nh)
    #   o_proj: (d, d)
    #   gate_proj: (d, ff)                -- SwiGLU
    #   up_proj: (d, ff)
    #   down_proj: (ff, d)
    module_dims = {
        "q_proj": (d, d),
        "k_proj": (d, d * kvh // nh),
        "v_proj": (d, d * kvh // nh),
        "o_proj": (d, d),
        "gate_proj": (d, ff),
        "up_proj": (d, ff),
        "down_proj": (ff, d),
    }

    layers_to_transform = _layers_to_transform(L, adapter_layers)
    n_layers_with_adapters = L if layers_to_transform is None else len(layers_to_transform)

    per_layer_adapter = sum(
        lora_rank * (d_in + d_out)
        for name, (d_in, d_out) in module_dims.items()
        if name in target_modules
    )
    n_adapter = per_layer_adapter * n_layers_with_adapters

    # For n_base we use the base's own parameter count -- but that's read from the loaded
    # model, not derived from arch alone (it depends on tied embeddings, biases, and
    # norms). Pass a base_param_count if you want a specific value; otherwise use the
    # rough analytical formula from Article 1.
    return {
        "n_adapter": n_adapter,
        "n_layers_with_adapters": n_layers_with_adapters,
        "n_layers_needing_backward": n_layers_with_adapters,
        "per_layer_adapter_params": per_layer_adapter,
    }
