"""Gated DeltaNet-2 token-mixer layer (pure JAX).

Implements the block design of Hatamizadeh, Choi, Kautz, "Gated DeltaNet-2:
Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791),
Section 3.5 / Figure 1 (right), on top of the Gated Delta Rule-2 recurrence
cores in rule.py.

Layer structure (per token representation x_t ∈ R^D):

    q path:  Linear -> short causal Conv -> SiLU -> L2 norm   (per head)
    k path:  Linear -> short causal Conv -> SiLU -> L2 norm   (per head)
    v path:  Linear -> short causal Conv -> SiLU
    decay:   g_t = −exp(a) ⊙ softplus(W_f x_t + δ)             Eq. 12
             (computed in fp32 before the kernel consumes it)
    erase:   b_t = σ(W_b x_t) ∈ [0,1]^dk  (or [0,2] variant)   Eq. 11
    write:   w_t = σ(W_w x_t) ∈ [0,1]^dv                       Eq. 11
    core:    Gated Delta Rule-2 recurrence (rule.py), Eq. 10/29
    output:  RMSNorm(o) ⊙ SiLU(W_g x)  ->  Linear -> R^D

The L2 normalization of q and k is load-bearing, not cosmetic: the erase
matrix I − k e ᵀ is contractive only when ‖k‖₂ ≤ 1; unnormalized keys make
the recurrence explode.

Streaming state. The layer is a recurrent map over segments: it consumes
and returns a `state = (S, (cache_q, cache_k, cache_v))`, where
S: [B, H, dk, dv] is the associative memory of Eq. 10 and each cache holds
the last conv_width−1 pre-convolution inputs of its path. Carrying BOTH
parts makes segmented processing exactly equal to one full-sequence call;
carrying S alone would let the convs see zero-padding at each segment
start, and those corrupted q/k/v would be written into S, contaminating
everything after the boundary. `state=None` starts a fresh sequence.

Parameters live in a plain dict (init_gated_deltanet2 /
gated_deltanet2_layer), matching the framework-free style of rule.py.
Shape conventions: x: [B, L, D]; per head dk (key/query) and dv (value);
H heads.

The paper's grouped value heads (q/k/g/b repeated across value-head
groups) and the hybrid SWA block are not implemented here; this file is
the single token-mixer layer.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

from rule import (
    D_TYPE,
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)

State = tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]


# --------------------------------------------------------------------------- #
#  Small building blocks
# --------------------------------------------------------------------------- #
def _causal_depthwise_conv(
    x: jax.Array, kernel: jax.Array, cache: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal 1-D convolution (the paper's "short convolution").

    Each channel is filtered independently with its own length-W kernel.
    Causality comes from prepending the W−1 inputs that precede this
    segment (`cache`) rather than zero-padding: position t then depends
    only on tokens ≤ t, and segmented processing is exactly equivalent to
    one full-sequence call when the returned cache is threaded through. A
    zero cache reproduces the usual start-of-sequence left-padding.

    x: [B, L, C]   kernel: [W, C]   cache: [B, W-1, C]
    ->  (out: [B, L, C], new_cache: [B, W-1, C])
    """
    W = kernel.shape[0]
    full = jnp.concatenate([cache, x], axis=1)  # [B, W-1+L, C]
    out = lax.conv_general_dilated(
        full,
        kernel[:, None, :],                     # [W, 1, C]: 1 input per group
        window_strides=(1,),
        padding="VALID",
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=kernel.shape[-1],   # depthwise: C groups
    )
    return out, full[:, -(W - 1):]  # last W-1 inputs feed the next segment


def _l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """L2-normalize the last axis: x / ‖x‖₂ (paper block design, Fig. 1)."""
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def _rmsnorm(x: jax.Array, scale: jax.Array, eps: float = 1e-6) -> jax.Array:
    """RMSNorm over the last axis with a learned scale (no mean subtraction).

    x: [..., d]   scale: broadcastable to x   ->   same shape as x
    """
    rms = jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x * rms * scale


# --------------------------------------------------------------------------- #
#  Parameter initialization
# --------------------------------------------------------------------------- #
def init_gated_deltanet2(
    key: jax.Array,
    d_model: int,
    n_heads: int = 4,
    head_dk: int | None = None,
    head_dv: int | None = None,
    conv_width: int = 4,
) -> dict:
    """Initialize a Gated DeltaNet-2 layer parameter dict.

    Args:
      key:        PRNG key.
      d_model:    model width D of the residual stream.
      n_heads:    number of recurrence heads H.
      head_dk:    per-head query/key width dk (default D / H).
      head_dv:    per-head value width dv (default D / H).
      conv_width: kernel length W of the short causal convolutions.
    Returns:
      dict of parameters, all fp32.
    """
    head_dk = head_dk or d_model // n_heads
    head_dv = head_dv or d_model // n_heads
    dqk = n_heads * head_dk   # total query/key width
    dv_ = n_heads * head_dv   # total value width

    k_ = iter(jax.random.split(key, 16))
    lin = lambda k, m, n: jax.random.normal(k, (m, n), D_TYPE) / math.sqrt(m)
    conv = lambda k, n: (jax.random.normal(k, (conv_width, n), D_TYPE)
                         / math.sqrt(conv_width))

    # Decay branch init, following the Gated DeltaNet parameterization of
    # Eq. 12: a = log U(1, 16) so exp(a) ∈ [1, 16] spreads per-channel base
    # forgetting rates; δ = softplus⁻¹(dt) with dt log-uniform in
    # [1e-3, 1e-1] so the decay magnitude at init (where W_f x ≈ 0) starts
    # small — long memory — with a spread of time scales.
    a = jnp.log(jax.random.uniform(next(k_), (dqk,), D_TYPE, 1.0, 16.0))
    dt = jnp.exp(jax.random.uniform(
        next(k_), (dqk,), D_TYPE, math.log(1e-3), math.log(1e-1)))
    delta = dt + jnp.log(-jnp.expm1(-dt))  # softplus⁻¹(dt)

    return {
        # q/k/v paths: Linear -> Conv -> SiLU (Fig. 1)
        "W_q": lin(next(k_), d_model, dqk), "conv_q": conv(next(k_), dqk),
        "W_k": lin(next(k_), d_model, dqk), "conv_k": conv(next(k_), dqk),
        "W_v": lin(next(k_), d_model, dv_), "conv_v": conv(next(k_), dv_),
        # decay branch (Eq. 12): g = −exp(a) ⊙ softplus(W_f x + δ)
        "W_f": lin(next(k_), d_model, dqk), "a": a, "delta": delta,
        # erase / write gates (Eq. 11): zero bias -> gates start at 0.5
        "W_b": lin(next(k_), d_model, dqk), "b_bias": jnp.zeros((dqk,), D_TYPE),
        "W_w": lin(next(k_), d_model, dv_), "w_bias": jnp.zeros((dv_,), D_TYPE),
        # output: per-head RMSNorm scale, SiLU gate projection, out projection
        "rms_scale": jnp.ones((n_heads, head_dv), D_TYPE),
        "W_g": lin(next(k_), d_model, dv_),
        "W_o": lin(next(k_), dv_, d_model),
    }


def init_state(params: dict, batch: int, n_heads: int = 4) -> State:
    """Fresh streaming state (zeros) for a batch: (S, conv caches).

    S: [B, H, dk, dv]; each cache: [B, conv_width-1, channels of its path].
    """
    H = n_heads
    dqk = params["W_q"].shape[1]
    dv_ = params["W_v"].shape[1]
    Wm1 = params["conv_q"].shape[0] - 1
    S = jnp.zeros((batch, H, dqk // H, dv_ // H), D_TYPE)
    caches = (jnp.zeros((batch, Wm1, dqk), D_TYPE),
              jnp.zeros((batch, Wm1, dqk), D_TYPE),
              jnp.zeros((batch, Wm1, dv_), D_TYPE))
    return S, caches


# --------------------------------------------------------------------------- #
#  Layer forward
# --------------------------------------------------------------------------- #
def gated_deltanet2_layer(
    params: dict,
    x: jax.Array,
    state: State | None = None,
    n_heads: int = 4,
    chunk_size: int = 64,
    allow_neg_eigval: bool = False,
) -> tuple[jax.Array, State]:
    """Gated DeltaNet-2 token mixer forward (Fig. 1 right / Sec. 3.5).

    Args:
      params:   dict from init_gated_deltanet2.
      x:        [B, L, D] token representations (residual stream).
      state:    optional streaming state (S, conv caches) returned by a
                previous call on the same sequence; None starts fresh.
      n_heads:  H, must match the value used at init.
      chunk_size: C for the chunkwise training path. If L is not divisible
                by C the layer falls back to the token-by-token core
                (identical math: Eq. 9 vs Eqs. 18-25).
      allow_neg_eigval: negative-eigenvalue variant (Sec. 3.1) — scales the
                erase gate to [0, 2] so the state transition can have
                eigenvalues in [−1, 1]. The write gate stays in [0, 1]
                because the spectral effect concerns the transition, not
                the value magnitude.
    Returns:
      (y: [B, L, D], new state (S: [B, H, dk, dv], conv caches))
    """
    B, L, D = x.shape
    H = n_heads
    dk = params["W_q"].shape[1] // H
    dv = params["W_v"].shape[1] // H
    x = x.astype(D_TYPE)

    if state is None:
        state = init_state(params, B, n_heads=H)
    S0, (cq, ck, cv) = state

    def heads(t, d):  # [B, L, H*d] -> [B, H, L, d] for the recurrence cores
        return t.reshape(B, L, H, d).transpose(0, 2, 1, 3)

    # ---- q/k/v paths: Linear -> short causal Conv -> SiLU ------------------
    # The conv gives each path a small local receptive field (shifts,
    # n-gram features) before the recurrence compresses history.
    q, cq = _causal_depthwise_conv(x @ params["W_q"], params["conv_q"], cq)
    k, ck = _causal_depthwise_conv(x @ params["W_k"], params["conv_k"], ck)
    v, cv = _causal_depthwise_conv(x @ params["W_v"], params["conv_v"], cv)
    q, k, v = jax.nn.silu(q), jax.nn.silu(k), jax.nn.silu(v)

    # L2 norm per head on q, k: keeps ‖k‖₂ = 1 so the erase matrix
    # I − k e ᵀ is contractive and the state cannot blow up.  [B, H, L, dk]
    q = _l2norm(heads(q, dk))
    k = _l2norm(heads(k, dk))
    v = heads(v, dv)                                        # [B, H, L, dv]

    # ---- gate branches ------------------------------------------------------
    # Eq. 12: g = −exp(a) ⊙ softplus(W_f x + δ) ≤ 0 — the log-decay. exp(a)
    # sets each channel's base forgetting rate; softplus makes it input-
    # dependent. fp32 as the paper requires for the cumulative sums in the
    # core.                                                   [B, L, H*dk]
    g = -jnp.exp(params["a"]) * jax.nn.softplus(
        (x @ params["W_f"]).astype(jnp.float32) + params["delta"])

    # Eq. 11: erase gate b ∈ [0,1]^dk — key side, WHICH channels of the old
    # association to erase; ×2 under the negative-eigenvalue variant.
    b = jax.nn.sigmoid(x @ params["W_b"] + params["b_bias"])
    if allow_neg_eigval:
        b = b * 2.0

    # Eq. 11: write gate w ∈ [0,1]^dv — value side, WHICH channels to write.
    w = jax.nn.sigmoid(x @ params["W_w"] + params["w_bias"])

    g, b, w = heads(g, dk), heads(b, dk), heads(w, dv)      # [B, H, L, ·]

    # ---- Gated Delta Rule-2 core (rule.py) ----------------------------------
    if L % chunk_size == 0:
        o, S_final = chunkwise_gated_delta_rule_2(
            q, k, v, g, b, w, S0, chunk_size=chunk_size)
    else:  # same recurrence token-by-token (also the decoding path)
        o, S_final = recurrent_gated_delta_rule_2(q, k, v, g, b, w, S0)
    # o: [B, H, L, dv]

    # ---- output: RMSNorm -> SiLU output gate -> projection (Fig. 1) ---------
    # Per-head RMSNorm stabilizes the read magnitude; the SiLU gate lets
    # each token modulate how much of the recurrent read enters the
    # residual stream; W_o mixes heads back to model width.
    o = _rmsnorm(o, params["rms_scale"][None, :, None, :])   # [B, H, L, dv]
    o = o.transpose(0, 2, 1, 3).reshape(B, L, H * dv)        # [B, L, H*dv]
    gate = jax.nn.silu(x @ params["W_g"])                    # [B, L, H*dv]
    y = (o * gate) @ params["W_o"]                           # [B, L, D]

    return y, (S_final, (cq, ck, cv))
