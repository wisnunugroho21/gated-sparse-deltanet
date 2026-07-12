"""Sparse Delta Memory (SDM) token-mixer layer in Flax NNX, ANNOTATED
against the paper (Cabannes et al., arXiv:2607.07386): Section 3.1 (the
four-step layer), 3.2 (isoFLOP parameterization), 3.3 (head count vs state
size), Section 4 "SDM Configuration", and Figure 2.

Block design (Fig. 2; Sec. 3.1), paper-faithful variant:
  k' = Linear_k(x), q' = Linear_q(x)          # PKM scores, d -> 2√N per head
  I_w, k = softmax(top-W(k'₁ ⊕ k'₂))          # sparse write address (Sec. 3.1 step 1)
  I_r, q = softmax(top-R(q'₁ ⊕ q'₂))          # sparse read address
  v  = Linear_v(x)                            # value (Fig. 2: plain W_v, no conv/SiLU)
  g  = -exp(A) · softplus(Linear_a(x) + b_dt) # log-decay, scalar per head (Sec. 3.1 step 2)
  β  = sigmoid(Linear_b(x))                   # input gate, scalar per head
  y  = recurrent_sdm(...)                     # sparse gated delta rule (Eqs. 3-5)
  out = Linear_o( RMSNorm(y) * SiLU(Linear_g(x)) )  # norm+gate+head mix (Sec. 3.1 step 4)

Differences from the GDN-2 mixer (layer.py) — all from the paper:
  * NO short convolutions and NO L2 norm / SiLU on the q/k paths: the
    "Connection to GDN" paragraph notes SDM drops the 1D convs, and the
    sparse addresses are softmax-normalized scores, not feature vectors.
  * Decay and input gate are per-head SCALARS (GDN-1 style), not GDN-2's
    per-channel gates — a per-slot gate would need a d -> N projection,
    exactly the dense cost SDM removes.
  * The state is a memory table M ∈ R^{N×dv} per head, with N = (√N)²
    slots addressed by product keys. Only W written + R read per token.
  * The initial state M₀ is (by default) a LEARNED parameter — the paper's
    default (Sec. 3.1 "Learned Initial State"): the table doubles as
    parametric memory storing pretraining knowledge at zero extra FLOPs.

Parameterization notes (Sec. 3.2 / 3.3 / 4):
  * Product-key projections W_q, W_k: d -> H·2√N. Paper default
    √N = d_qk_total/(2H) = d/(4H)  (from d_qk_total = d/2, Sec. 3.2), used
    here when sqrt_n is not given. N = (d/4H)² per head.
  * W = R = 64 in the paper (= GDN's d_qk, the isoFLOP match, Sec. 3.2).
  * H is a state-size knob with NO FLOPs impact (Sec. 3.3): total memory
    per layer is (d_qk_total)²·d_v_total / 4H², so FEWER heads mean a
    BIGGER table. The paper uses H = 1-2.
  * Decay init "matching GDN/FLA conventions" (Sec. 4): A_log = log U(1,16)
    (the FLA convention this repo already uses; the paper's "A uniform in
    [0,16]" phrase refers to the same family recipe) and
    b_dt = softplus⁻¹(dt), dt log-uniform in [1e-3, 1e-1].
  * Projection init: the SDM paper is silent; we keep this repo's
    convention (Xavier-uniform, gain 2^{-2.5}, from GDN-2 App. D.5).
  * M₀ init: zeros (empty memory; the paper does not specify — zeros makes
    the null-M₀ ablation the exact same model at step 0).

Execution cores (core=): "recurrent" (default) scans token-by-token — no
length constraint, the oracle; "chunkwise" runs the chunkwise-parallel WY
core (paper App. A; sdm_rule.chunkwise_sdm) — sequential depth L/C
instead of L, requires L % chunk_size == 0 in __call__. step() mirrors
the GDN-2 layer: with core="chunkwise" the chunk-aligned prefix takes the
parallel core and the ragged tail (incl. L=1 decode) the recurrent one;
both compute the same recurrence, so the split is invisible.

Memory cost: the state is H·N·dv floats PER SEQUENCE — e.g. √N=64,
dv=128, H=1 is 2 MB fp32; the paper-scale √N=1024 is 512 MB. Size sqrt_n
to your hardware. The chunkwise core additionally materializes
[C, C, W, W] index-match tensors per chunk (see sdm_rule.py).
"""

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import lax

from layer import GatedRMSNorm
from sdm_rule import chunkwise_sdm, recurrent_sdm

F32 = jnp.float32

# Repo convention (GDN-2 App. D.5): Xavier-uniform, gain 2^{-2.5}
# (variance_scaling scale = gain² = 2^{-5}). The SDM paper does not
# specify projection inits, so the house rule stands.
_XAVIER = nnx.initializers.variance_scaling(2**-5, "fan_avg", "uniform")


# --------------------------------------------------------------------------- #
#  Inference cache. The ENTIRE state of an SDM layer is the memory table —
#  there are no short convolutions, so unlike GDN2Cache there is no conv
#  left-context to carry. Fixed-size, but LARGE: [B, H, N, dv] fp32.
# --------------------------------------------------------------------------- #
class SDMCache(NamedTuple):
    memory: jax.Array  # [B, H, N, dv]  the sparse delta memory table M


class SparseDeltaMemory(nnx.Module):
    """Sparse Delta Memory token mixer (paper Fig. 2; Sec. 3.1)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,  # H; paper uses 1-2 — FEWER heads = BIGGER table (Sec. 3.3)
        sqrt_n: int | None = None,  # √N per head; default d/(4H) (Sec. 3.2)
        num_writes: int = 64,  # W; paper: 64 = GDN's d_qk (isoFLOP, Sec. 3.2)
        num_reads: int = 64,  # R; paper: 64
        head_v_dim: int = 128,  # d_v per head
        learned_init: bool = True,  # learned M₀ (paper default, Sec. 3.1)
        compute_dtype: jnp.dtype = jnp.float32,
        core: str = "recurrent",  # "recurrent" (oracle) or "chunkwise" (App. A)
        chunk_size: int = 64,  # C for core="chunkwise"; ignored otherwise
        *,
        rngs: nnx.Rngs,
    ):
        # Matmul dtype for the q/k/v/b/o projections. The core (sdm_rule.py)
        # upcasts to fp32 regardless, and the decay branch is kept fp32
        # below — same numerical policy as the GDN-2 layer.
        self.compute_dtype = compute_dtype
        self.d_model = d_model
        self.H = num_heads
        self.sqrt_n = sqrt_n if sqrt_n is not None else d_model // (4 * num_heads)
        self.N = self.sqrt_n**2
        self.W = num_writes
        self.R = num_reads
        self.dv = head_v_dim

        assert self.sqrt_n >= 2, "sqrt_n must be >= 2 (N = sqrt_n**2 slots)"
        assert 1 <= self.W <= self.N, "num_writes must be in [1, N]"
        assert 1 <= self.R <= self.N, "num_reads must be in [1, N]"
        assert core in ("recurrent", "chunkwise"), f"unknown core {core!r}"
        self.core = core
        self.chunk_size = chunk_size

        v_proj_dim = self.H * self.dv

        # Product-key score projections (Sec. 3.1 step 1): W_k, W_q map
        # d -> 2√N PER HEAD; each output splits into two √N halves whose
        # outer SUM scores all N slots.
        self.k_proj = nnx.Linear(
            d_model,
            self.H * 2 * self.sqrt_n,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.q_proj = nnx.Linear(
            d_model,
            self.H * 2 * self.sqrt_n,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )
        self.v_proj = nnx.Linear(
            d_model,
            v_proj_dim,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # v_t = W_v x_t (Fig. 2 — no conv, no SiLU)
        self.b_proj = nnx.Linear(
            d_model,
            self.H,
            use_bias=True,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )  # β_t = σ(W_b x_t), scalar per head (Sec. 3.1 step 2)
        self.a_proj = nnx.Linear(
            d_model,
            self.H,
            use_bias=False,  # the bias is b_dt, stored separately per head
            kernel_init=_XAVIER,
            dtype=F32,  # decay branch stays fp32 end-to-end
            param_dtype=F32,
            rngs=rngs,
        )  # W_a in α_t = exp(-A·softplus(W_a x_t + b_dt))

        # Decay parameters, per head (Sec. 4: "matching GDN/FLA
        # conventions" — the same family recipe layer.py uses, here at
        # head granularity because α is a per-head scalar):
        #   A_log = log U(1, 16)  ->  A = exp(A_log) ∈ [1, 16]
        #   b_dt  = softplus⁻¹(dt), dt log-uniform in [1e-3, 1e-1]
        self.A_log = nnx.Param(
            jnp.log(jax.random.uniform(rngs.params(), (self.H,), F32, 1.0, 16.0))
        )
        dt = jnp.exp(
            jax.random.uniform(
                rngs.params(), (self.H,), F32, jnp.log(1e-3), jnp.log(1e-1)
            )
        )
        self.dt_bias = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)))  # softplus⁻¹(dt)

        # Learned initial state M₀ (Sec. 3.1): the paper's default. Zeros
        # init — the model starts identical to the null-M₀ ablation and
        # learns what to preload. When learned_init=False the initial
        # memory is a fixed zero table (no parameter).
        self.M0 = (
            nnx.Param(jnp.zeros((self.H, self.N, self.dv), F32))
            if learned_init
            else None
        )

        # Output stage (Sec. 3.1 step 4): head-wise RMSNorm × SiLU gate,
        # then W_o mixes the H heads back to d_model. Same module as the
        # GDN-2 layer — the paper's output stage is identical.
        self.o_norm = GatedRMSNorm(
            head_dim=self.dv,
            d_model=d_model,
            inner_dim=v_proj_dim,
            rngs=rngs,
        )
        self.o_proj = nnx.Linear(
            v_proj_dim,
            d_model,
            use_bias=False,
            kernel_init=_XAVIER,
            dtype=compute_dtype,
            param_dtype=F32,
            rngs=rngs,
        )

    # ------------------------------------------------------------------ #
    #  Product-key sparse selection (Sec. 3.1 step 1 / PKM background).
    # ------------------------------------------------------------------ #
    def _select(
        self, scores: jax.Array, k: int
    ) -> tuple[jax.Array, jax.Array]:
        """Top-k product-key selection with softmax-normalized values.

        scores: [B, L, H·2√N] (raw projection output). Returns
        (idx: [B, H, L, k] int32 slot ids in [0, N),
         val: [B, H, L, k] fp32 softmax weights, descending).

        Uses the PKM identity topk(s₁ ⊕ s₂) = topk(topk(s₁) ⊕ topk(s₂)):
        only kk = min(k, √N) candidates per half are kept, so the full
        [√N × √N] score grid is never materialized beyond kk² entries.
        The k selected (i₁, i₂) pairs are distinct, hence the k slot ids
        i₁·√N + i₂ are distinct — the core's scatter relies on this.
        """
        B, L, _ = scores.shape
        s = scores.astype(F32).reshape(B, L, self.H, 2, self.sqrt_n)
        s1, s2 = s[..., 0, :], s[..., 1, :]  # halves        [B, L, H, √N]

        kk = min(k, self.sqrt_n)
        v1, i1 = lax.top_k(s1, kk)  # per-half candidates    [B, L, H, kk]
        v2, i2 = lax.top_k(s2, kk)

        grid = v1[..., :, None] + v2[..., None, :]  # outer sum   [.., kk, kk]
        vals, flat = lax.top_k(grid.reshape(B, L, self.H, kk * kk), k)

        r1, r2 = flat // kk, flat % kk  # positions within the candidate grid
        idx = (
            jnp.take_along_axis(i1, r1, axis=-1) * self.sqrt_n
            + jnp.take_along_axis(i2, r2, axis=-1)
        )  # slot id = i₁·√N + i₂                             [B, L, H, k]

        val = jax.nn.softmax(vals, axis=-1)  # softmax over the SELECTED
        # scores only (Sec. 4: "read and write activations use
        # softmax-normalization")
        return idx.swapaxes(1, 2), val.swapaxes(1, 2)  # -> [B, H, L, k]

    def _project(self, x: jax.Array):
        """Front-end shared by __call__ and step: projections -> sparse
        addresses + values + gates, all on the [B, H, L, ...] layout the
        core consumes. Returns (iw, kw, ir, qr, v, g, beta)."""
        B, L, _ = x.shape

        iw, kw = self._select(self.k_proj(x), self.W)  # write address
        ir, qr = self._select(self.q_proj(x), self.R)  # read address

        v = self.v_proj(x).reshape(B, L, self.H, self.dv).swapaxes(1, 2)

        # Log-decay branch, fp32 (Sec. 3.1 step 2):
        #   g_t = -exp(A_log) · softplus(W_a x_t + b_dt) ≤ 0, per head.
        f = self.a_proj(x).astype(F32) + self.dt_bias[...]  # [B, L, H]
        g = -jnp.exp(self.A_log[...]) * jax.nn.softplus(f)
        g = g.swapaxes(1, 2)  # [B, H, L]

        beta = jax.nn.sigmoid(self.b_proj(x).astype(F32)).swapaxes(1, 2)

        return iw, kw, ir, qr, v, g, beta

    def _initial_memory(self, batch_size: int) -> jax.Array:
        """M₀ broadcast to the batch: the learned table (gradients flow
        through the broadcast) or zeros. [B, H, N, dv] fp32."""
        if self.M0 is not None:
            return jnp.broadcast_to(
                self.M0[...], (batch_size, self.H, self.N, self.dv)
            )
        return jnp.zeros((batch_size, self.H, self.N, self.dv), F32)

    def _output(self, y: jax.Array, x: jax.Array) -> jax.Array:
        """RMSNorm × SiLU gate × W_o (Sec. 3.1 step 4). y: [B, H, L, dv]."""
        o = y.swapaxes(1, 2)  # [B, L, H, dv] — per-head axis for the norm
        o = self.o_norm(o, x).astype(x.dtype)
        return self.o_proj(o)

    def __call__(
        self,
        x: jax.Array,
        initial_state: jax.Array | None = None,
        return_state: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Full-sequence forward. x: [B, L, d_model] -> out: [B, L, d_model],
        or (out, M_final) with return_state=True.

        core="recurrent" has no length constraint; core="chunkwise"
        requires L % chunk_size == 0 (the core validates and raises) — pad
        training batches to a multiple of C, or use `step`, which handles
        ragged lengths via its recurrent tail. `initial_state` / M_final
        are [B, H, N, dv]; passing one OVERRIDES the learned M₀ — state
        carried across segment calls is EXACT here (no conv caveat, unlike
        the GDN-2 layer: an SDM layer's entire state is the table)."""
        B, _, _ = x.shape
        iw, kw, ir, qr, v, g, beta = self._project(x)

        M0 = (
            self._initial_memory(B)
            if initial_state is None
            else initial_state.astype(F32)
        )
        if self.core == "chunkwise":
            y, M_final = chunkwise_sdm(
                iw, kw, ir, qr, v, g, beta, M0, chunk_size=self.chunk_size)
        else:
            y, M_final = recurrent_sdm(iw, kw, ir, qr, v, g, beta, M0)

        out = self._output(y, x)
        if return_state:
            return out, M_final
        return out

    # ------------------------------------------------------------------ #
    #  Streaming / inference. Same math as __call__ threading the table
    #  in -> out; both prefill and decode take the recurrent core.
    # ------------------------------------------------------------------ #
    def init_cache(
        self, batch_size: int, max_len: int | None = None, dtype=None
    ) -> SDMCache:
        """Fresh streaming cache: the initial memory table (learned M₀ or
        zeros). `max_len` / `dtype` accepted for interface parity with
        GDN2Cache but UNUSED — the table is fixed-size and always fp32."""
        return SDMCache(memory=self._initial_memory(batch_size))

    def step(self, x: jax.Array, cache: SDMCache) -> tuple[jax.Array, SDMCache]:
        """Streaming forward. x: [B, L, d_model] (any L ≥ 1).
        Returns (out, new_cache). Equals the corresponding slice of a full
        pass — there is no conv context, so segmentation is invisible
        (verified in test_sdm_layer.py).

        With core="chunkwise" the length splits as L = n_full + tail
        (n_full = (L // C)·C): the aligned prefix runs the parallel
        chunkwise core (fast prefill), the ragged tail — including the
        L=1 decode step — the recurrent core, both threading the same
        memory table (the GDN-2 layer's step pattern). With
        core="recurrent" everything takes the recurrent core."""
        iw, kw, ir, qr, v, g, beta = self._project(x)
        M = cache.memory.astype(F32)

        L = x.shape[1]
        n_full = (
            (L // self.chunk_size) * self.chunk_size
            if self.core == "chunkwise"
            else 0
        )
        outs = []

        if n_full > 0:
            # Chunkwise prefill of the aligned prefix, warm-started from —
            # and updating — the running memory table.
            y_head, M = chunkwise_sdm(
                iw[:, :, :n_full], kw[:, :, :n_full],
                ir[:, :, :n_full], qr[:, :, :n_full],
                v[:, :, :n_full], g[:, :, :n_full], beta[:, :, :n_full],
                M, chunk_size=self.chunk_size)
            outs.append(y_head)

        if n_full < L:
            # Ragged tail (or the whole input when L < C, e.g. decode):
            # recurrent core, token-by-token.
            y_tail, M = recurrent_sdm(
                iw[:, :, n_full:], kw[:, :, n_full:],
                ir[:, :, n_full:], qr[:, :, n_full:],
                v[:, :, n_full:], g[:, :, n_full:], beta[:, :, n_full:],
                M)
            outs.append(y_tail)

        y = outs[0] if len(outs) == 1 else jnp.concatenate(outs, axis=2)
        return self._output(y, x), SDMCache(memory=M)
