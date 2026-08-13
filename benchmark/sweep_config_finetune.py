"""Config dataclass and sweep generation for LoRA fine-tuning.

Parallels sweep_config.py: one-lever-at-a-time from a fixed baseline. Architecture fields
(d, layers, heads, ff_intermediate, vocab, num_kv_heads) are dropped -- those come from
the loaded Hugging Face config at runtime, not from this file. Fields kept and their
Article 1 meaning: batch, seq_len, precision (autocast compute), optimizer, checkpointing,
steps.

New fields for fine-tuning: base_model_name, base_storage_precision, lora_rank,
lora_target_modules, lora_adapter_layers.

TARGET HARDWARE: A100 80GB. Sweep endpoints are sized so the baseline and single-lever
rows fit; the demanding-corner combined rows are predicted to OOM. That is the article's
"the wall moves" evidence.
"""
import dataclasses

BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# The seven target modules that constitute a standard LoRA-on-Llama configuration.
# Attention (q/k/v/o_proj) and SwiGLU MLP (gate/up/down_proj) linears -- adapters touch
# every parameter path, so the sweep's activation formula covers both branches.
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclasses.dataclass
class Config:
    base_model_name: str
    base_storage_precision: str  # "bf16" | "fp32" | "int8" | "int4"
    lora_rank: int
    # Tuple of module-name substrings that get adapters. Kept as a tuple (not list) so the
    # dataclass is hashable and the JSON round-trip through the subprocess boundary is
    # exact.
    lora_target_modules: tuple
    lora_adapter_layers: str  # "all" | "upper-N"
    batch: int
    seq_len: int
    precision: str  # "amp_bf16" | "fp32"
    optimizer: str  # "adam" | "adam8bit"
    checkpointing: bool
    steps: int
    lever: str
    lever_value: str

    def as_dict(self):
        return dataclasses.asdict(self)


def make_baseline_finetune(
    base_model_name=BASE_MODEL_NAME,
    base_storage_precision="bf16",
    lora_rank=8,
    lora_target_modules=DEFAULT_LORA_TARGET_MODULES,
    lora_adapter_layers="all",
    batch=8,
    seq_len=512,
    precision="amp_bf16",
    optimizer="adam",
    checkpointing=False,
    steps=7,
):
    """Baseline picked for the article's pedagogy (see plan file § "Fine-tune baseline"):
    realistic LoRA batch/seq on A100 80GB, all three levers exercisable from here."""
    return Config(
        base_model_name=base_model_name,
        base_storage_precision=base_storage_precision,
        lora_rank=lora_rank,
        lora_target_modules=tuple(lora_target_modules),
        lora_adapter_layers=lora_adapter_layers,
        batch=batch,
        seq_len=seq_len,
        precision=precision,
        optimizer=optimizer,
        checkpointing=checkpointing,
        steps=steps,
        lever="baseline",
        lever_value="baseline",
    )


def _variant(baseline, lever, lever_value, **overrides):
    if "lora_target_modules" in overrides:
        overrides["lora_target_modules"] = tuple(overrides["lora_target_modules"])
    cfg = dataclasses.replace(baseline, **overrides)
    cfg.lever = lever
    cfg.lever_value = str(lever_value)
    return cfg


def batch_sweep(baseline, values=(4, 8, 16, 32)):
    return [_variant(baseline, "batch_size", v, batch=v) for v in values]


def seq_sweep(baseline, values=(256, 512, 1024, 2048)):
    return [_variant(baseline, "seq_len", v, seq_len=v) for v in values]


def rank_sweep(baseline, values=(4, 8, 16, 64)):
    """Rank moves adapter weights/gradients/optimizer state -- three static columns each
    already <1% of param count. Expected result: nearly flat. That flatness is the whole
    point of running this sweep."""
    return [_variant(baseline, "lora_rank", v, lora_rank=v) for v in values]


def checkpointing_sweep(baseline, values=(False, True)):
    return [_variant(baseline, "checkpointing", v, checkpointing=v) for v in values]


def adapter_placement_sweep(baseline, values=("all", "upper-11", "upper-6", "upper-3")):
    """Adapter placement lever: number of layers the backward pass reaches through.
    'upper-11' is half-depth for TinyLlama (22 layers); 'upper-3' is the aggressive corner.
    Real memory saving in activations, real accuracy cost -- no counterpart in Article 1."""
    return [_variant(baseline, "adapter_placement", v, lora_adapter_layers=v) for v in values]


def base_precision_sweep(baseline, values=("bf16", "fp32")):
    """Base storage precision. fp32 needs no extra dependency and is included by default;
    int8/int4 rows appear only if bitsandbytes is importable (benchmark_finetune.py's
    parent-startup gate filters them out otherwise, same pattern as adam8bit).
    fp32-base is the point that reintroduces Article 1's ~n_params*2 autocast cache."""
    return [_variant(baseline, "base_precision", v, base_storage_precision=v) for v in values]


def combined_sweep(baseline, demanding_batch, demanding_seq):
    """The 'how do I make it fit' set: baseline / adam8bit / checkpointing / both, all at
    the most demanding batch x seq_len point. Stacks levers on purpose -- excluded from
    SINGLE_LEVER_TAGS. Note: adam8bit rows are auto-filtered if bitsandbytes isn't
    importable, so the effective count can be 4 or 2 depending on install."""
    demanding = dataclasses.replace(baseline, batch=demanding_batch, seq_len=demanding_seq)
    return [
        _variant(demanding, "combined", "baseline"),
        _variant(demanding, "combined", "adam8bit", optimizer="adam8bit"),
        _variant(demanding, "combined", "checkpointing", checkpointing=True),
        _variant(demanding, "combined", "both", optimizer="adam8bit", checkpointing=True),
    ]


def build_all_configs_finetune(
    baseline=None,
    batch_values=(4, 8, 16, 32),
    seq_values=(256, 512, 1024, 2048),
    rank_values=(4, 8, 16, 64),
    checkpointing_values=(False, True),
    placement_values=("all", "upper-11", "upper-6", "upper-3"),
    base_precision_values=("bf16", "fp32"),
    demanding_batch=None,
    demanding_seq=None,
):
    baseline = baseline or make_baseline_finetune()
    demanding_batch = demanding_batch or max(batch_values)
    demanding_seq = demanding_seq or max(seq_values)

    configs = [baseline]
    configs += batch_sweep(baseline, batch_values)
    configs += seq_sweep(baseline, seq_values)
    configs += rank_sweep(baseline, rank_values)
    configs += checkpointing_sweep(baseline, checkpointing_values)
    configs += adapter_placement_sweep(baseline, placement_values)
    configs += base_precision_sweep(baseline, base_precision_values)
    configs += combined_sweep(baseline, demanding_batch, demanding_seq)
    return configs


# Levers expected to vary at most one field from baseline. combined is excluded by design
# (stacks multiple levers on purpose).
SINGLE_LEVER_TAGS = {
    "baseline", "batch_size", "seq_len", "lora_rank",
    "checkpointing", "adapter_placement", "base_precision",
}


def differing_fields(baseline, cfg):
    """Fields that differ between baseline and cfg, excluding CSV-shape metadata.

    base_model_name and lora_target_modules are excluded from the diff: they are keyed off
    the base model choice, not off a lever. If they were included, any sweep changing
    the model would fail the single-lever guard.
    """
    excluded = {"lever", "lever_value", "base_model_name", "lora_target_modules"}
    base_fields = {f.name for f in dataclasses.fields(Config)} - excluded
    return [name for name in base_fields if getattr(baseline, name) != getattr(cfg, name)]
