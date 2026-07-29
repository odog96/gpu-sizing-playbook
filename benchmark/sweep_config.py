"""Config dataclass and sweep generation.

Sweeps are built one-lever-at-a-time from a fixed baseline so results are interpretable:
hold everything constant, change one lever, observe the memory delta. Only the combined-
levers set (built for the "how do I make it fit" chart) intentionally varies more than one
field at once -- it's a distinct, separately-labeled group.
"""
import dataclasses


@dataclasses.dataclass
class Config:
    d: int
    layers: int
    heads: int
    ff_mult: int
    vocab: int
    batch: int
    seq_len: int
    precision: str  # "amp_bf16" | "fp32"
    optimizer: str  # "adam" | "adam8bit"
    checkpointing: bool
    steps: int
    lever: str  # which lever this config varies, e.g. "batch_size"
    lever_value: str  # the value of that lever, as a string (for uniform CSV columns)

    def as_dict(self):
        return dataclasses.asdict(self)


def make_baseline(d=2048, layers=20, heads=16, ff_mult=4, vocab=1000, steps=7):
    return Config(
        d=d,
        layers=layers,
        heads=heads,
        ff_mult=ff_mult,
        vocab=vocab,
        batch=256,
        seq_len=100,
        precision="amp_bf16",
        optimizer="adam",
        checkpointing=False,
        steps=steps,
        lever="baseline",
        lever_value="baseline",
    )


def _variant(baseline, lever, lever_value, **overrides):
    cfg = dataclasses.replace(baseline, **overrides)
    cfg.lever = lever
    cfg.lever_value = str(lever_value)
    return cfg


def batch_sweep(baseline, values=(256, 1024, 4096)):
    return [_variant(baseline, "batch_size", v, batch=v) for v in values]


def seq_sweep(baseline, values=(100, 512, 1024, 2048)):
    return [_variant(baseline, "seq_len", v, seq_len=v) for v in values]


def optimizer_sweep(baseline, values=("adam", "adam8bit")):
    return [_variant(baseline, "optimizer", v, optimizer=v) for v in values]


def checkpointing_sweep(baseline, values=(False, True)):
    return [_variant(baseline, "checkpointing", v, checkpointing=v) for v in values]


def checkpointing_batch_sweep(baseline, values=(256, 1024, 4096)):
    """checkpointing=True repeated across every value already in the batch sweep, not just
    baseline's batch -- the plain checkpointing_sweep only tells you the effect at one
    point; this maps how far checkpointing pushes the batch ceiling. Deliberately varies
    two fields (batch + checkpointing) at once, so it's excluded from SINGLE_LEVER_TAGS,
    same reasoning as the combined sweep."""
    return [_variant(baseline, "checkpointing_batch", v, batch=v, checkpointing=True) for v in values]


def checkpointing_seq_sweep(baseline, values=(100, 512, 1024, 2048)):
    """checkpointing=True repeated across every value already in the seq_len sweep --
    same reasoning as checkpointing_batch_sweep, for the sequence-length axis."""
    return [_variant(baseline, "checkpointing_seq", v, seq_len=v, checkpointing=True) for v in values]


def combined_sweep(baseline, demanding_batch, demanding_seq):
    """The 'how do I make it fit' set: baseline / 8-bit Adam only / checkpointing only / both,
    all at the most demanding batch x seq_len combination."""
    demanding = dataclasses.replace(baseline, batch=demanding_batch, seq_len=demanding_seq)
    return [
        _variant(demanding, "combined", "baseline"),
        _variant(demanding, "combined", "adam8bit", optimizer="adam8bit"),
        _variant(demanding, "combined", "checkpointing", checkpointing=True),
        _variant(demanding, "combined", "both", optimizer="adam8bit", checkpointing=True),
    ]


def fp32_reference(baseline):
    return [_variant(baseline, "precision", "fp32", precision="fp32")]


def build_all_configs(
    baseline=None,
    batch_values=(256, 1024, 4096),
    seq_values=(100, 512, 1024, 2048),
    optimizer_values=("adam", "adam8bit"),
    checkpointing_values=(False, True),
    demanding_batch=None,
    demanding_seq=None,
):
    baseline = baseline or make_baseline()
    demanding_batch = demanding_batch or max(batch_values)
    demanding_seq = demanding_seq or max(seq_values)

    configs = [baseline]
    configs += batch_sweep(baseline, batch_values)
    configs += seq_sweep(baseline, seq_values)
    configs += optimizer_sweep(baseline, optimizer_values)
    configs += checkpointing_sweep(baseline, checkpointing_values)
    configs += checkpointing_batch_sweep(baseline, batch_values)
    configs += checkpointing_seq_sweep(baseline, seq_values)
    configs += combined_sweep(baseline, demanding_batch, demanding_seq)
    configs += fp32_reference(baseline)
    return configs


# Levers that are expected to vary at most one field from baseline. The combined sweep and
# the checkpointing x batch/seq crosses are excluded by design -- they stack multiple
# levers on purpose.
SINGLE_LEVER_TAGS = {"baseline", "batch_size", "seq_len", "optimizer", "checkpointing", "precision"}


def differing_fields(baseline, cfg):
    base_fields = {f.name for f in dataclasses.fields(Config)} - {"lever", "lever_value"}
    return [name for name in base_fields if getattr(baseline, name) != getattr(cfg, name)]
