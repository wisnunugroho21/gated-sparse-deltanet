"""Sparse Delta Memory (SDM) token-mixer layer in Flax NNX, ANNOTATED against
Cabannes et al., "Sparse Delta Memory", arXiv:2607.07386, Sec. 3.1 (the four
SDM steps) and Sec. 3.2-3.3 (isoFLOP sizing).

Block design (paper Fig. 2; Sec. 3.1):
  k' = W_k x  -> split (k'_1, k'_2)     # pre-PKM WRITE key halves, each R^√N
  q' = W_q x  -> split (q'_1, q'_2)     # pre-PKM READ  query halves
  v  = W_v x                            # value (no conv / SiLU — see note)
  g  = -exp(A)·softplus(W_a x + b_dt)   # per-head log forget gate, fp32 (Eq. 3)
  b  = sigmoid(W_b x)                   # per-head input gate β (Eq. 4)
  y  = sparse_delta_memory(k'_1,k'_2, q'_1,q'_2, v, g, b, M)   # Eqs. 3-5 + PKM
  out = W_o( RMSNorm(y) * SiLU(W_g x) )  # RMSNorm + SiLU gate + head-mix (step 4)

FAITHFUL SDM (default). Built on the GDN base with a SINGLE input gate β; the
forget/input gates are PER-HEAD SCALARS (not per-channel as in GDN-2), matching
the sparse recurrence in sparse_rule.py.

GATED SPARSE DELTANET-2 (decouple_erase_write=True). Re-adds the GDN-2
erase/write decoupling on top of the sparse memory:
  * value-side WRITE gate w = σ(W_w x) ∈ R^{dv} per head, gating z = w ⊙ v
    (exactly GDN-2's write gate);
  * key-side ERASE gate b — the sparse analogue of GDN-2's per-key-channel b —
    as a per-SELECTED-SLOT gate from a Product-Key erase projection
    W_e: d → 2√N, gathered at the write slots (κ_e = b ⊙ κ on the recall).
β is retained as the overall input gate, so this is a strict superset of both
faithful SDM (b≡1, w≡1) and the GDN-2 gating.

DIFFERENCES vs the GDN-2 layer (layer.py), all from the paper:
  * NO short causal convolution — "the only difference [from GDN] is the lack
    of 1D convolutions on the qkv vectors, present in GDN but not in SDM"
    (Sec. 3.1). Streaming state is therefore JUST the memory table (no conv
    cache).
  * NO SiLU / L2-normalization on the q, k, v paths: the PKM addressing is a
    softmax over the selected slot scores (done inside sparse_rule.py).
  * The dense per-head state S ∈ R^{dk×dv} is replaced by a large explicit
    memory table M ∈ R^{N×dv} per head (N = root²), addressed sparsely.
  * M0 is a LEARNABLE parameter (paper's default "parametric memory"): it can
    store knowledge at pretraining and is reused at test time for free
    (learn_initial_state=True). Set False for the null-initialized baseline.

isoFLOP sizing (Sec. 3.2-3.3). The paper ties W_q,W_k ∈ R^{d×d/2},
W_v ∈ R^{d×d}, W=R=64, and N=(d/4H)² so SDM matches GDN's parameters and
FLOPs. Here root (=√N), head_v_dim, W, R, and num_heads are free knobs; pick
them to hit that relationship if you want the isoFLOP setting. num_heads is the
state-size dial (Sec. 3.3): fewer heads → larger memory at NO extra FLOPs.

Scope: the recurrent TOKEN MIXER only (paper Fig. 2). The hybrid model (Sec. 4)
interleaves SDM global layers with sliding-window attention and gated MLPs;
those wrappers are not implemented here.
"""

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

# Reuse the GDN-2 layer's shared pieces: fp32 alias, Xavier init, the plain and
# gated RMSNorms, and the paper-family decay init helpers are re-derived below.
from layer import F32, _XAVIER, GatedRMSNorm
from sparse_rule import (
    chunkwise_sparse_delta_memory,
    recurrent_sparse_delta_memory,
)


# --------------------------------------------------------------------------- #
#  Inference cache for streaming decode.
#
#  Unlike a softmax KV-cache, SDM's whole history collapses into the FIXED-SIZE
#  memory table M [B, H, N, dv] — it does not grow with sequence length. And
#  because faithful SDM has NO short convolution, that table is the ENTIRE
#  streaming state (contrast GDN2Cache, which also caches conv left-context).
# --------------------------------------------------------------------------- #
class SDMCache(NamedTuple):
    recurrent_state: jax.Array  # [B, H, N, dv]  the sparse memory table M


class SparseDeltaMemory(nnx.Module):
    """Sparse Delta Memory recurrent token mixer (paper Fig. 2; Sec. 3.1)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,        # H SDM heads; few heads => larger state (Sec. 3.3)
        head_v_dim: int = 128,     # d_v per head (memory row width)
        n_slots_root: int = 256,   # root = √N; the table has N = root² slots
        num_write: int = 64,       # W write slots per token (paper W=64)
        num_read: int = 64,        # R read  slots per token (paper R=64)
        chunk_size: int = 64,      # C for the chunkwise training core
        learn_initial_state: bool = True,  # parametric memory M0 (paper default)
        decouple_erase_write: bool = False,  # GDN-2 erase/write gates (Gated Sparse DeltaNet-2)
        compute_dtype: jnp.dtype = jnp.float32,
        core: str = "chunkwise",   # "chunkwise" (parallel) training path
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.H = num_heads
        self.dv = head_v_dim
        self.root = n_slots_root
        self.N = n_slots_root * n_slots_root
        self.W = num_write
        self.R = num_read
        self.chunk_size = chunk_size
        self.learn_initial_state = learn_initial_state
        self.decouple_erase_write = decouple_erase_write
        self.compute_dtype = compute_dtype
        self.core = core

        # PKM top-k selects among `root` per half, so W, R must not exceed root.
        assert self.W <= self.root and self.R <= self.root, (
            f"W={self.W}, R={self.R} must be <= n_slots_root={self.root}")

        # Pre-PKM score projections: per head, k' and q' live in R^{2·root}
        # (two √N halves whose outer sum scores the N slots). Flat width H·2·root
        # (= d/2 per the isoFLOP tie when root = d/4H).
        score_dim = self.H * 2 * self.root
        self.k_proj = nnx.Linear(
            d_model, score_dim, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs)  # write keys k'
        self.q_proj = nnx.Linear(
            d_model, score_dim, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs)  # read queries q'

        # Value projection: v_t = W_v x_t (Eq. 4), flat width H·dv (= d when dv=d/H).
        self.v_proj = nnx.Linear(
            d_model, self.H * self.dv, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs)

        # Per-head forget gate α_t = exp(-exp(A)·softplus(W_a x + δ)) (Eq. 3) and
        # input gate β_t = σ(W_b x) (Eq. 4). Both project to H scalars per token.
        self.f_proj = nnx.Linear(
            d_model, self.H, use_bias=False, kernel_init=_XAVIER,
            dtype=F32, param_dtype=F32, rngs=rngs)   # decay branch stays fp32
        self.b_proj = nnx.Linear(
            d_model, self.H, use_bias=True, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs)

        # GDN-2 decoupling gates (only when decouple_erase_write): the value-side
        # write gate w = σ(W_w x) ∈ R^{H·dv}, and the Product-Key ERASE score
        # projection W_e: d → H·2·root (scores every slot; gathered per selected
        # write slot inside the core to form b = σ(ew1[a] + ew2[c])).
        if decouple_erase_write:
            self.e_proj = nnx.Linear(
                d_model, self.H * 2 * self.root, use_bias=False,
                kernel_init=_XAVIER, dtype=compute_dtype, param_dtype=F32,
                rngs=rngs)
            self.w_proj = nnx.Linear(
                d_model, self.H * self.dv, use_bias=True, kernel_init=_XAVIER,
                dtype=compute_dtype, param_dtype=F32, rngs=rngs)

        # Decay parameters, GDN/FLA family init (paper: "matching GDN/FLA
        # conventions"), per head: exp(A_log) ∈ [1,16] and softplus(δ) ∈
        # [1e-3, 1e-1] so the per-token decay at init is small (long memory).
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (self.H,), F32, 1.0, 16.0)))
        dt = jnp.exp(jax.random.uniform(
            rngs.params(), (self.H,), F32, jnp.log(1e-3), jnp.log(1e-1)))
        self.dt_bias = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)))  # δ = softplus⁻¹(dt)

        # Learnable initial memory M0 [H, N, dv] (paper's parametric memory).
        # Zero-init = the null-initialized baseline as a learning start point;
        # per-slot gradients differ via the learned addressing. Broadcast over
        # batch at call time. When not learned, forward uses zeros directly.
        if learn_initial_state:
            self.M0 = nnx.Param(jnp.zeros((self.H, self.N, self.dv), F32))

        # Output stage: head-wise RMSNorm of y, SiLU output gate, head-mix W_o
        # (paper step 4). Reuses the GDN-2 GatedRMSNorm (identical structure).
        self.o_norm = GatedRMSNorm(
            head_dim=self.dv, d_model=d_model, inner_dim=self.H * self.dv,
            rngs=rngs)
        self.o_proj = nnx.Linear(
            self.H * self.dv, d_model, use_bias=False, kernel_init=_XAVIER,
            dtype=compute_dtype, param_dtype=F32, rngs=rngs)

    # -- projections -------------------------------------------------------- #
    def _split_scores(self, x: jax.Array, proj: nnx.Linear, B: int, L: int):
        """Project x -> per-head pre-PKM score halves.

        proj(x): [B, L, H·2·root] -> [B, H, L, root] × 2 (the two √N halves
        whose outer sum scores the N slots). Returns (half1, half2)."""
        s = proj(x).reshape(B, L, self.H, 2 * self.root).swapaxes(1, 2)  # [B,H,L,2root]
        return s[..., :self.root], s[..., self.root:]

    def _project(self, x: jax.Array):
        """Front-end shared by training and streaming: PKM score halves, value,
        the per-head log-decay g and input gate β, and (when decoupled) the
        GDN-2 erase score halves ew1, ew2 and write gate w. Returns
        (kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w) on the [B,H,L,...] axis;
        the last three are None unless decouple_erase_write."""
        B, L, _ = x.shape

        kw1, kw2 = self._split_scores(x, self.k_proj, B, L)  # write key halves
        qr1, qr2 = self._split_scores(x, self.q_proj, B, L)  # read query halves

        v = self.v_proj(x).reshape(B, L, self.H, self.dv).swapaxes(1, 2)  # [B,H,L,dv]

        # Per-head log-decay g = -exp(A)·softplus(W_a x + δ) ≤ 0 (Eq. 3), fp32.
        f = self.f_proj(x).astype(F32) + self.dt_bias[...].astype(F32)  # [B,L,H]
        a = jnp.exp(self.A_log[...].astype(F32))                        # [H]
        g = (-a * jax.nn.softplus(f)).swapaxes(1, 2)                    # [B,H,L]

        # Per-head input gate β = σ(W_b x) ∈ [0,1] (Eq. 4).
        beta = jax.nn.sigmoid(self.b_proj(x)).astype(F32).swapaxes(1, 2)  # [B,H,L]

        # GDN-2 decoupling gates (optional): erase score halves (the core turns
        # them into a per-selected-slot b) and the value-side write gate w.
        ew1 = ew2 = w = None
        if self.decouple_erase_write:
            ew1, ew2 = self._split_scores(x, self.e_proj, B, L)         # [B,H,L,root]
            w = self.w_proj(x).reshape(B, L, self.H, self.dv).swapaxes(1, 2)  # [B,H,L,dv]

        return kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w

    def _init_memory(self, B: int) -> jax.Array:
        """Batched initial memory [B, H, N, dv]: the learned M0 broadcast over
        the batch, or zeros for the null-initialized baseline."""
        if self.learn_initial_state:
            return jnp.broadcast_to(
                self.M0[...][None], (B, self.H, self.N, self.dv)).astype(F32)
        return jnp.zeros((B, self.H, self.N, self.dv), F32)

    def _output(self, y: jax.Array, x: jax.Array) -> jax.Array:
        """RMSNorm + SiLU gate + head-mix (paper step 4). y: [B,H,L,dv]."""
        B, _, L, _ = y.shape
        y = y.swapaxes(1, 2).reshape(B, L, self.H, self.dv)  # [B,L,H,dv]
        y = self.o_norm(y, x).astype(x.dtype)                # SiLU gate inside
        return self.o_proj(y)

    def __call__(
        self,
        x: jax.Array,
        initial_state: jax.Array | None = None,
        return_state: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Full-sequence (training) forward via the chunkwise-parallel core.
        x: [B, L, d_model] -> [B, L, d_model], or (out, M_final) if
        return_state. L must be divisible by chunk_size.

        initial_state / M_final use the memory layout [B, H, N, dv]; pass
        M_final of one segment (e.g. via stop_gradient) as initial_state of the
        next for truncated-BPTT-style continuation. With no short conv, this
        carry is EXACT (unlike GDN2, there is no conv context to lose)."""
        B, _, _ = x.shape
        kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w = self._project(x)
        M0 = self._init_memory(B) if initial_state is None else initial_state

        y, M_final = chunkwise_sparse_delta_memory(
            kw1, kw2, qr1, qr2, v, g, beta, M0,
            ew1=ew1, ew2=ew2, w=w,
            chunk_size=self.chunk_size, W=self.W, R=self.R)

        out = self._output(y, x)
        return (out, M_final) if return_state else out

    # -- streaming / inference ---------------------------------------------- #
    def init_cache(self, batch_size: int, max_len: int | None = None) -> SDMCache:
        """Empty streaming cache = the (learned or zero) initial memory table.
        `max_len` is accepted for interface parity but unused — the SDM state is
        fixed-size, independent of sequence length."""
        return SDMCache(recurrent_state=self._init_memory(batch_size))

    def step(self, x: jax.Array, cache: SDMCache) -> tuple[jax.Array, SDMCache]:
        """Streaming forward. x: [B, L, d_model] (L>=1). Returns (out, cache).

        L is split as n_full + tail with n_full = (L // C)·C: the chunk-aligned
        prefix runs through the parallel chunkwise core (fast prefill), the tail
        (including the L=1 decode step) through the token-by-token recurrent
        core. Both thread the same fixed-size memory, so the split is invisible
        in the output."""
        kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w = self._project(x)
        L = x.shape[1]
        n_full = (L // self.chunk_size) * self.chunk_size
        M = cache.recurrent_state
        outs = []

        # Slice an optional gate tensor along the length axis, or pass None.
        def sl(t, lo, hi):
            return None if t is None else t[:, :, lo:hi]

        if n_full > 0:
            y_pre, M = chunkwise_sparse_delta_memory(
                kw1[:, :, :n_full], kw2[:, :, :n_full],
                qr1[:, :, :n_full], qr2[:, :, :n_full],
                v[:, :, :n_full], g[:, :, :n_full], beta[:, :, :n_full], M,
                ew1=sl(ew1, 0, n_full), ew2=sl(ew2, 0, n_full), w=sl(w, 0, n_full),
                chunk_size=self.chunk_size, W=self.W, R=self.R)
            outs.append(y_pre)

        if n_full < L:
            y_tail, M = recurrent_sparse_delta_memory(
                kw1[:, :, n_full:], kw2[:, :, n_full:],
                qr1[:, :, n_full:], qr2[:, :, n_full:],
                v[:, :, n_full:], g[:, :, n_full:], beta[:, :, n_full:], M,
                ew1=sl(ew1, n_full, L), ew2=sl(ew2, n_full, L), w=sl(w, n_full, L),
                W=self.W, R=self.R)
            outs.append(y_tail)

        y = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=2)
        return self._output(y, x), SDMCache(M)
