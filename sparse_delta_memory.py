"""Sparse Delta Memory (SDM, arXiv:2607.07386) — recurrent JAX reference.

Mirrors the structure of `gated_deltarule_2.py`: a per-(batch, head)
recurrence lifted with vmap. Intended as the float-verification baseline
for a later chunkwise-parallel implementation.

Correspondence with the dense GDN-2 kernel:
    SDM write  : M_bar + beta * k (v - M_bar^T k)^T          (k sparse, W nnz)
    GDN-2 write: S_bar  + k (w*v - S_bar^T (b*k))^T
    => SDM == GDN-2 with b = w = beta (scalar), per-slot decay,
       and k/q restricted to the top-W / top-R PKM slots.

Semantics that differ from dense GDN-2 (per the paper):
    * only the W selected write slots decay; untouched slots are frozen
    * read happens AFTER the write of the same token (as in your kernel)
    * no short conv on q/k/v; softmax normalization over selected scores
"""

import jax
import jax.numpy as jnp
from jax import lax

D_TYPE = jnp.float32


# ---------------------------------------------------------------------------
# 1. PKM product-key addressing
# ---------------------------------------------------------------------------

def pkm_topk(s1: jax.Array, s2: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    """Top-k over the outer sum s1 (+) s2 without materializing all N scores.

    s1, s2: [sqrt_N] score halves for one token.  N = sqrt_N ** 2.
    Returns (idx [k] flat slot ids, scores [k]), sorted by slot id.
    Uses: top_k(s1 (+) s2) == top_k(top_k(s1) (+) top_k(s2)).
    """
    n2 = s2.shape[-1]
    v1, i1 = lax.top_k(s1, k)
    v2, i2 = lax.top_k(s2, k)
    cand = (v1[:, None] + v2[None, :]).reshape(-1)            # [k*k]
    cand_idx = (i1[:, None] * n2 + i2[None, :]).reshape(-1)   # [k*k]
    scores, pos = lax.top_k(cand, k)
    idx = cand_idx[pos]
    order = jnp.argsort(idx)   # canonical (sorted) order, as in the paper
    return idx[order], scores[order]


def sdm_addressing(
    x: jax.Array,      # [T, d]
    Wk: jax.Array,     # [d, 2 * sqrt_N]
    Wq: jax.Array,     # [d, 2 * sqrt_N]
    W: int,
    R: int,
):
    """Sparse write keys / read queries for a whole sequence (parallel in T).

    Returns k_idx [T, W], k_val [T, W], q_idx [T, R], q_val [T, R].
    Values are softmax-normalized over the selected slots (paper default).
    """
    sqrt_n = Wk.shape[-1] // 2

    def one_token(xt):
        ks = xt @ Wk
        qs = xt @ Wq
        ki, kv = pkm_topk(ks[:sqrt_n], ks[sqrt_n:], W)
        qi, qv = pkm_topk(qs[:sqrt_n], qs[sqrt_n:], R)
        return ki, jax.nn.softmax(kv), qi, jax.nn.softmax(qv)

    return jax.vmap(one_token)(x)


def per_slot_decay_scores(
    k_idx: jax.Array,   # [T, W] selected (flat) write-slot ids
    x: jax.Array,       # [T, d]
    Wa: jax.Array,      # [d, 2 * sqrt_N]  decay projection, factorized like PKM
    sqrt_n: int,
) -> jax.Array:
    """'SDM-2' extension: per-slot decay factorized through PKM halves.

    The decay score of slot (i1, i2) is a1[i1] + a2[i2]; we only ever need it
    at the W selected slots, so we gather instead of materializing N scores.
    Returns raw scores [T, W]; feed into g = -A * softplus(scores + b_dt).
    Vanilla SDM = broadcast one scalar score per token instead.
    """
    a = x @ Wa                              # [T, 2*sqrt_N]
    a1, a2 = a[:, :sqrt_n], a[:, sqrt_n:]
    i1 = k_idx // sqrt_n
    i2 = k_idx % sqrt_n
    return jnp.take_along_axis(a1, i1, axis=1) + jnp.take_along_axis(a2, i2, axis=1)


# ---------------------------------------------------------------------------
# 2. Recurrent SDM core (single batch element, single head)
# ---------------------------------------------------------------------------

def _recurrent_sdm_single(
    q_idx: jax.Array,   # [T, R] int
    q_val: jax.Array,   # [T, R]
    k_idx: jax.Array,   # [T, W] int
    k_val: jax.Array,   # [T, W]
    v: jax.Array,       # [T, dv]
    g: jax.Array,       # [T, W]  per-selected-slot log-decay (tile a scalar
                        #         across W to recover vanilla SDM)
    b: jax.Array,       # [T]     input gate beta in [0, 1]
    M0: jax.Array,      # [N, dv]
) -> tuple[jax.Array, jax.Array]:
    q_val = q_val.astype(D_TYPE)
    k_val = k_val.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    b = b.astype(D_TYPE)
    M0 = M0.astype(D_TYPE)

    def step(M, inp):
        qi, qv, ki, kv, vt, gt, bt = inp

        # -- gated delta write, restricted to the W selected slots ---------
        alpha = jnp.exp(gt)[:, None]              # [W, 1]
        M_sel = M[ki]                             # gather      [W, dv]
        M_bar = alpha * M_sel                     # decay (touched slots only)

        r_t = kv @ M_bar                          # retrieval   [dv]
        delta_v = vt - r_t
        new_rows = M_bar + bt * kv[:, None] * delta_v[None, :]
        M = M.at[ki].set(new_rows)                # scatter (top-k ids unique)

        # -- sparse read from the post-write memory (matches GDN-2 kernel) -
        o_t = qv @ M[qi]                          # [dv]
        return M, o_t

    M_final, o = lax.scan(step, M0, (q_idx, q_val, k_idx, k_val, v, g, b))
    return o, M_final


def _batchify(fn):
    # heads: every arg carries a leading H axis (M0 is [H, N, dv])
    over_heads = jax.vmap(fn, in_axes=(0,) * 8, out_axes=(0, 0))
    # batch: learned M0 is shared across the batch -> in_axes=None
    return jax.vmap(
        over_heads,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None),
        out_axes=(0, 0),
    )


def recurrent_sparse_delta_memory(
    q_idx, q_val, k_idx, k_val, v, g, b, M0,
) -> tuple[jax.Array, jax.Array]:
    """Shapes: [B, H, T, ...] for token inputs, M0: [H, N, dv].

    Returns o: [B, H, T, dv] and M_final: [B, H, N, dv].
    """
    return _batchify(_recurrent_sdm_single)(q_idx, q_val, k_idx, k_val, v, g, b, M0)


# ---------------------------------------------------------------------------
# 3. Dense-limit equivalence check against the GDN-2 kernel
#    (N = dk, all slots selected, b = w = beta, per-channel decay)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    from gated_deltarule_2 import recurrent_gated_delta_rule_2

    rng = np.random.default_rng(0)
    B, H, T, dk, dv = 2, 3, 16, 8, 8   # N = dk in the dense limit
    N = dk

    k = rng.standard_normal((B, H, T, dk)).astype(np.float64)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)  # stability: beta*|k|^2 <= 2
    # (real SDM uses softmax-normalized key values, which satisfies this)
    q = rng.standard_normal((B, H, T, dk)).astype(np.float64)
    v = rng.standard_normal((B, H, T, dv)).astype(np.float64)
    g = -np.abs(rng.standard_normal((B, H, T, dk))) * 0.1   # per-channel decay
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, T))))
    S0 = np.zeros((B, H, dk, dv))

    # GDN-2 with b = w = beta broadcast per-channel
    beta_full = np.broadcast_to(beta[..., None], (B, H, T, dk))
    beta_v = np.broadcast_to(beta[..., None], (B, H, T, dv))
    o_ref, S_ref = recurrent_gated_delta_rule_2(
        jnp.array(q), jnp.array(k), jnp.array(v),
        jnp.array(g), jnp.array(beta_full), jnp.array(beta_v), jnp.array(S0),
    )

    # SDM in the dense limit: every slot selected, dense values as "sparse"
    idx = np.broadcast_to(np.arange(N), (B, H, T, N)).astype(np.int32)
    o_sdm, M_fin = recurrent_sparse_delta_memory(
        jnp.array(idx), jnp.array(q),
        jnp.array(idx), jnp.array(k),
        jnp.array(v), jnp.array(g), jnp.array(beta),
        jnp.zeros((H, N, dv)),
    )

    print("max |o - o_ref|:", np.max(np.abs(np.array(o_sdm) - np.array(o_ref))))
    print("max |M - S_ref|:", np.max(np.abs(np.array(M_fin) - np.array(S_ref))))
