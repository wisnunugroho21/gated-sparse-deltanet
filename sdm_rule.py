"""Sparse Delta Memory (SDM) — recurrent core in JAX.

Implements the sparse gated delta rule of Cabannes et al., "Sparse Delta
Memory: Scaling the State of Linear RNNs through Sparsity"
(arXiv:2607.07386), Section 3.1, paper-faithful variant.

The state is an explicit memory table M ∈ R^{N×dv} with N slots (N is
orders of magnitude larger than GDN's dk). Each token touches only the
W slots selected for writing and the R slots selected for reading; every
other slot is left EXACTLY unchanged — including no decay ("Unselected
slots remain unchanged", Sec. 3.1). Per token t (Eqs. 3-5, Fig. 2):

    M̃_t[i] = α_t · M_{t-1}[i]                    i ∈ I_w   (forget, Eq. 3)
    ṽ_t    = Σ_{i∈I_w} k⁽ⁱ⁾ · M̃_t[i]                       (retrieve)
    M_t[i] = M̃_t[i] + β_t · k⁽ⁱ⁾ · (v_t − ṽ_t)   i ∈ I_w   (delta write, Eq. 4)
    y_t    = Σ_{i∈I_r} q⁽ⁱ⁾ · M_t[i]                        (read, Eq. 5)

α_t ∈ (0,1) and β_t ∈ (0,1) are per-head SCALARS (unlike GDN-2's
per-channel gates); k⁽ⁱ⁾, q⁽ⁱ⁾ are the softmax-normalized top-W / top-R
PKM scores produced by the layer (sdm_layer.py). The read happens AFTER
the write, matching o_t = S_tᵀ q_t in rule.py.

NOTE on Eq. (4) as printed: the paper writes the delta as (v_t − M̃_t[i])
per slot, but Fig. 2 (ṽ_t = M̃_tᵀ k_t, Δv_t = v_t − ṽ_t) and the WY math of
Appendix A (write-write interaction matrix A, the B·retrieved correction)
use the FULL retrieval ṽ_t. We implement the ṽ_t form — it is also the
only one that satisfies the paper's own "Connection to GDN": with N = dk,
W = R = N and dense key values, the update reduces exactly to Gated
DeltaNet,  M ← Diag(α)M − β k kᵀ Diag(α) M + β k vᵀ,  which equals
rule.py's Gated Delta Rule-2 with tied gates b = w = β and scalar decay
(verified in test_sdm_rule.py).

Scope: token-by-token scan only — the Phase-1 oracle and the inference
path. The chunkwise-parallel training core (paper Appendix A: sparse
interaction matrices + triangular solve, the sparse analogue of rule.py's
WY cores) is future work; this core is O(L·(W+R)·dv) sequential.

Shape conventions (one head): iw, kw: [L, W]; ir, qr: [L, R];
v: [L, dv]; g, beta: [L]; M0: [N, dv]. The public entry point adds
leading [B, H] axes via vmap, exactly like rule.py's _batchify.
All math runs in fp32 (the memory table is the recurrent state — same
policy as rule.py / App. D of the GDN-2 paper).
"""

import jax
import jax.numpy as jnp
from jax import lax

# Compute dtype for all internal math; the memory table stays fp32.
D_TYPE = jnp.float32


def _recurrent_sdm_single(
    iw: jax.Array,
    kw: jax.Array,
    ir: jax.Array,
    qr: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    M0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token SDM forward for ONE (batch, head) pair.

    Args:
      iw: [L, W] int   write slot indices (top-W, distinct per token)
      kw: [L, W]       sparse write key values k⁽ⁱ⁾ (softmax weights)
      ir: [L, R] int   read slot indices (top-R, distinct per token)
      qr: [L, R]       sparse read query values q⁽ⁱ⁾
      v:  [L, dv]      value vectors
      g:  [L]          log-decay, g ≤ 0; α = exp(g) ∈ (0,1]  (scalar/head)
      beta: [L]        input gate β ∈ (0,1)                  (scalar/head)
      M0: [N, dv]      initial memory table
    Returns:
      (y: [L, dv], M_final: [N, dv])
    """
    kw = kw.astype(D_TYPE)
    qr = qr.astype(D_TYPE)
    v = v.astype(D_TYPE)
    beta = beta.astype(D_TYPE)
    M0 = M0.astype(D_TYPE)

    alpha = jnp.exp(g.astype(D_TYPE))  # α = exp(g), g ≤ 0 so α ∈ (0,1]  [L]

    def step(M, inp):
        # M: [N, dv]; iw_t: [W], kw_t: [W], ir_t: [R], qr_t: [R],
        # a_t, b_t: scalar, v_t: [dv].
        iw_t, kw_t, ir_t, qr_t, a_t, b_t, v_t = inp

        # Eq. 3: gather the W selected rows and decay ONLY them (lazy /
        # write-triggered forgetting — untouched slots keep full strength).
        M_sel = a_t * jnp.take(M, iw_t, axis=0)  # M̃_t[I_w]        [W, dv]

        # Retrieve ṽ_t = Σ k⁽ⁱ⁾ M̃_t[i] — the sparse read along the WRITE
        # keys (Fig. 2), i.e. what the decayed memory currently returns
        # for this token's write address.                          [dv]
        v_ret = kw_t @ M_sel

        # Eq. 4: delta write — store only the residual between the target
        # v_t and the recalled ṽ_t, spread over the W slots by k⁽ⁱ⁾ and
        # scaled by the input gate β.
        M_sel = M_sel + kw_t[:, None] * (b_t * (v_t - v_ret))[None, :]

        # Scatter back. The top-W product-key indices of one token are
        # DISTINCT by construction (distinct (i₁, i₂) pairs — see
        # sdm_layer's selection), so .set has no collisions.
        M = M.at[iw_t].set(M_sel)

        # Eq. 5: read the POST-write memory at the R read slots.
        y_t = qr_t @ jnp.take(M, ir_t, axis=0)  # [dv]

        return M, y_t

    M_final, y = lax.scan(step, M0, (iw, kw, ir, qr, alpha, beta, v))
    return y, M_final


def _batchify(fn):
    """Lift the per-head core to batched [B, H, ...] inputs.

    Pure plumbing, no math (rule.py's pattern): vmap over heads (axis 1),
    then over batch (axis 0). All 8 arguments map over their leading axis;
    M0 simply has no L axis.
    """
    over_heads = jax.vmap(fn, in_axes=(0,) * 8, out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=(0,) * 8, out_axes=(0, 0))


def recurrent_sdm(
    iw: jax.Array,
    kw: jax.Array,
    ir: jax.Array,
    qr: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    M0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token Sparse Delta Memory forward (oracle / inference path).

    Args:
      iw: [B, H, L, W] int   kw: [B, H, L, W]     write indices / key values
      ir: [B, H, L, R] int   qr: [B, H, L, R]     read indices / query values
      v:  [B, H, L, dv]      g, beta: [B, H, L]   values / log-decay / gate
      M0: [B, H, N, dv]                           initial memory table
    Returns:
      (y: [B, H, L, dv], M_final: [B, H, N, dv])
    """
    return _batchify(_recurrent_sdm_single)(iw, kw, ir, qr, v, g, beta, M0)
