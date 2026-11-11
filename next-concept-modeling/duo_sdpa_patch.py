"""Stand-in for the `flash_attn` package that DUO's models/dit.py imports (duo @ 7c9b498).

DUO's non-causal DiT already computes attention with torch's F.scaled_dot_product_attention
(models/dit.py regular_attention_multi_headed). The only flash_attn code its non-causal path calls is
flash_attn.layers.rotary.apply_rotary_emb_torch, a pure-PyTorch rotary helper. flash-attn has no
Blackwell (sm_120) build pinned by DUO, so this module provides that one helper, reproduced from
flash-attn 2.x (non-interleaved form, the only one DUO uses), and makes every other flash_attn entry
point raise. Rows that load it are labelled "DUO, SDPA attention".

install() must run before DUO's models package is imported.
"""
import sys
import types

import torch

SHIM_VERSION = "duo-sdpa-shim-1"
CALLS = {"apply_rotary_emb_torch": 0}


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """flash-attn 2.x semantics. x: (batch, seqlen, nheads, headdim); cos, sin: (seqlen, rotary_dim / 2)."""
    if interleaved:
        raise NotImplementedError("duo_sdpa_patch: interleaved rotary is not used by DUO and not provided")
    CALLS["apply_rotary_emb_torch"] += 1
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], (ro_dim, x.shape)
    # einops "... d -> ... 1 (2 d)": the half-width table repeated twice, with a heads axis of 1
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    return torch.cat([x[..., :ro_dim] * cos + _rotate_half(x[..., :ro_dim]) * sin, x[..., ro_dim:]], dim=-1)


def _refuse(modname):
    def __getattr__(name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(f"duo_sdpa_patch: {modname}.{name} is not available (no flash-attn on this machine)")
    return __getattr__


def install():
    if "flash_attn" in sys.modules and getattr(sys.modules["flash_attn"], "__version__", "") != SHIM_VERSION:
        raise RuntimeError("a real flash_attn module is already imported; refusing to mix it with the shim")
    root = types.ModuleType("flash_attn")
    root.__version__ = SHIM_VERSION
    root.__path__ = []
    layers = types.ModuleType("flash_attn.layers")
    layers.__path__ = []
    rotary = types.ModuleType("flash_attn.layers.rotary")
    rotary.apply_rotary_emb_torch = apply_rotary_emb_torch

    def apply_rotary_emb_qkv_(*_a, **_k):
        raise RuntimeError("duo_sdpa_patch: apply_rotary_emb_qkv_ (causal path) is not available")

    rotary.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_
    iface = types.ModuleType("flash_attn.flash_attn_interface")
    iface.__getattr__ = _refuse("flash_attn.flash_attn_interface")
    root.layers = layers
    layers.rotary = rotary
    root.flash_attn_interface = iface
    sys.modules.update({
        "flash_attn": root,
        "flash_attn.layers": layers,
        "flash_attn.layers.rotary": rotary,
        "flash_attn.flash_attn_interface": iface,
    })
    return root


def is_installed(mod):
    return getattr(mod, "__version__", "") == SHIM_VERSION
