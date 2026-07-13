"""Chunkwise-parallel Sparse Delta Memory (SDM, arXiv:2607.07386) in JAX.

Mirrors the structure of the dense chunkwise GDN-2 kernel
(`_chunkwise_single_faithful`), with each dense operation replaced by its
sparse counterpart per Appendix A of the paper:

    dense GDN-2                          sparse SDM
    -----------                          ----------
    G = cumsum(g)  (per channel)     ->  Lw, Lr: segmented per-slot cumulative
                                         log-decay (a slot only accrues decay
                                         at tokens that WRITE it)
    Ebar  = gamma * (b*k)            ->  effE  = k_val * exp(Lw)
    Kbar  = k * exp(-G)              ->  effK  = k_val * exp(-Lw)
    Qg    = gamma * q                ->  q_eff = q_val * exp(Lr)
    Ktail = k * gamma_C / gamma      ->  ktail = k_val * exp(Ltot - Lw)
    T   = tril(Ebar @ Kbar^T, -1)    ->  A[t,j]  over matching write slots
    Aqk = tril(Qg @ Kbar^T)          ->  QK[t,j] over Ir(t) matched to Iw(j)
    (I + T)^-1 triangular solve      ->  Msys = I + diag(b) tril(A,-1);
                                         dV_const = Msys^-1 diag(b) V
                                         Bmat     = Msys^-1 diag(-b)
    R = U - Y @ S                    ->  dv = dV_const + Bmat @ retrieved
    o = Qg @ S + Aqk @ R             ->  o  = inter + QK @ dv
    S_new = gamma_C*S + Ktail^T @ R  ->  scatter-multiply decay on touched
                                         slots, then scatter-add ktail * dv

The sparse inner products exploit that slot indices are SORTED within each
token (pkm_topk sorts them): the paper's two-pointer merge becomes a
searchsorted + gather, O(C^2 * W * log W) per chunk instead of a dense
C^2 * W^2 match tensor.

Beta convention (as in the recurrent reference): dv_t = beta_t (v_t - r_t),
writes are k_val * dv, i.e. SDM == GDN-2 with b = w = beta.
Decay g is per-write-occurrence [T, W] ("SDM-2" per-slot decay); broadcast a
per-token scalar across W to recover vanilla SDM.
"""

import jax
import jax.numpy as jnp
from jax import lax

D_TYPE = jnp.float32


def _match_gather(src_idx: jax.Array, src_val: jax.Array, qry_idx: jax.Array) -> jax.Array:
    """For each source row, look up the value at each queried slot id.

    src_idx: [C, W] slot ids, sorted ascending within each row.
    src_val: [C, W] values aligned with src_idx.
    qry_idx: [Q]    flat query slot ids.
    Returns [C, Q]: src_val at the matching slot, 0 where the row lacks it.
    (JAX analogue of the paper's two-pointer merge over sorted indices.)
    """
    W = src_idx.shape[-1]

    def per_row(row_idx, row_val):
        pos = jnp.clip(jnp.searchsorted(row_idx, qry_idx), 0, W - 1)
        hit = row_idx[pos] == qry_idx
        return jnp.where(hit, row_val[pos], jnp.zeros((), row_val.dtype))

    return jax.vmap(per_row)(src_idx, src_val)


def _phase1_chunk(q_idx, q_val, k_idx, k_val, v, g, b):
    """Intra-chunk parallel quantities for ONE chunk (vmapped over chunks).

    q_idx [C, R] int, q_val [C, R], k_idx [C, W] int (sorted), k_val [C, W],
    v [C, dv], g [C, W] per-slot log-decay, b [C] input gate.
    """
    C, W = k_idx.shape
    R = q_idx.shape[-1]
    eye = jnp.eye(C, dtype=v.dtype)
    incl = jnp.tril(jnp.ones((C, C), dtype=v.dtype))  # src token <= query token

    # --- segmented cumulative log-decay (per-slot analogue of cumsum(g)) ---
    # hits_w[s, t, n] = log-decay applied by source token s to slot Iw[t, n]
    hits_w = _match_gather(k_idx, g, k_idx.reshape(-1)).reshape(C, C, W)
    hits_r = _match_gather(k_idx, g, q_idx.reshape(-1)).reshape(C, C, R)
    Lw = jnp.einsum("ts,stn->tn", incl, hits_w)   # decay up to & incl. token t
    Lr = jnp.einsum("ts,stn->tn", incl, hits_r)
    Ltot = hits_w.sum(axis=0)                     # full-chunk decay per write occ.

    effE = k_val * jnp.exp(Lw)          # 'Ebar'
    effK = k_val * jnp.exp(-Lw)         # 'Kbar'
    q_eff = q_val * jnp.exp(Lr)         # 'Qg'
    ktail = k_val * jnp.exp(Ltot - Lw)  # 'Ktail'

    # --- sparse interaction matrices (two-pointer merge as searchsorted) ---
    Gw = _match_gather(k_idx, effK, k_idx.reshape(-1)).reshape(C, C, W)  # [j, t, n]
    A = jnp.tril(jnp.einsum("tn,jtn->tj", effE, Gw), k=-1)
    Gq = _match_gather(k_idx, effK, q_idx.reshape(-1)).reshape(C, C, R)
    QK = jnp.tril(jnp.einsum("tn,jtn->tj", q_eff, Gq))

    # --- triangular system: Msys = I + diag(b) * tril(A, -1) ---
    Msys = eye + b[:, None] * A
    dV_const = jax.scipy.linalg.solve_triangular(
        Msys, b[:, None] * v, lower=True, unit_diagonal=True)
    Bmat = jax.scipy.linalg.solve_triangular(
        Msys, -jnp.diag(b), lower=True, unit_diagonal=True)

    return effE, q_eff, ktail, QK, dV_const, Bmat


def _chunkwise_sdm_single(
    q_idx: jax.Array,   # [L, R] int, sorted within each token
    q_val: jax.Array,   # [L, R]
    k_idx: jax.Array,   # [L, W] int, sorted within each token
    k_val: jax.Array,   # [L, W]
    v: jax.Array,       # [L, dv]
    g: jax.Array,       # [L, W] per-selected-slot log-decay
    b: jax.Array,       # [L]    input gate beta
    M0: jax.Array,      # [N, dv]
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    L, W = k_idx.shape
    dv = v.shape[-1]
    C = chunk_size
    if L % C:
        raise ValueError(f"sequence length L={L} must be divisible by chunk_size={C}")
    Nc = L // C

    def to_chunks(x, dtype=None):
        x = x.reshape(Nc, C, *x.shape[1:])
        return x if dtype is None else x.astype(dtype)

    q_idx, k_idx = to_chunks(q_idx), to_chunks(k_idx)
    q_val, k_val = to_chunks(q_val, D_TYPE), to_chunks(k_val, D_TYPE)
    v, g = to_chunks(v, D_TYPE), to_chunks(g, D_TYPE)
    b = b.reshape(Nc, C).astype(D_TYPE)
    M0 = M0.astype(D_TYPE)

    # ---- Phase 1: batched over all chunks at once ----
    effE, q_eff, ktail, QK, dV_const, Bmat = jax.vmap(_phase1_chunk)(
        q_idx, q_val, k_idx, k_val, v, g, b)

    # ---- Phase 2: sequential over chunks (gather / correct / scatter) ----
    def chunk_step(M, inp):
        qi, qe, ki, kE, kt, QK_n, dVc, B_n, g_n = inp

        retrieved = jnp.einsum("tn,tnd->td", kE, M[ki])   # read at write slots
        inter = jnp.einsum("tn,tnd->td", qe, M[qi])       # read at read slots

        dv_t = dVc + B_n @ retrieved                      # correct intra-chunk
        o = inter + QK_n @ dv_t                           # output

        flat = ki.reshape(-1)
        # decay every touched slot: duplicate indices compose multiplicatively,
        # giving exp(sum of per-occurrence log-decays) = exp(Ltot) per slot
        M = M.at[flat].multiply(jnp.exp(g_n.reshape(-1))[:, None])
        # write-back: duplicate indices sum
        contrib = kt.reshape(-1)[:, None] * jnp.repeat(dv_t, W, axis=0)
        M = M.at[flat].add(contrib)
        return M, o

    M_final, o = lax.scan(
        chunk_step, M0,
        (q_idx, q_eff, k_idx, effE, ktail, QK, dV_const, Bmat, g),
    )
    return o.reshape(L, dv), M_final


def _batchify(fn):
    over_heads = jax.vmap(fn, in_axes=(0,) * 8, out_axes=(0, 0))
    return jax.vmap(  # learned M0 shared across the batch
        over_heads, in_axes=(0, 0, 0, 0, 0, 0, 0, None), out_axes=(0, 0))


def chunkwise_sparse_delta_memory(
    q_idx, q_val, k_idx, k_val, v, g, b, M0, chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Shapes: [B, H, L, ...] for token inputs, M0: [H, N, dv]."""
    def fun(qi, qv, ki, kv, V, G, Bt, M):
        return _chunkwise_sdm_single(qi, qv, ki, kv, V, G, Bt, M, chunk_size=chunk_size)
    return _batchify(fun)(q_idx, q_val, k_idx, k_val, v, g, b, M0)


# ---------------------------------------------------------------------------
# Verification: chunkwise vs. recurrent SDM (float64)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    import numpy as np
    import sparse_delta_memory as sdm

    D_TYPE = jnp.float64
    sdm.D_TYPE = jnp.float64

    rng = np.random.default_rng(0)
    B, H, L, N, W, R, dvdim, C = 2, 2, 64, 32, 4, 4, 8, 8

    def rand_idx(k):
        out = np.stack([
            np.sort(rng.choice(N, size=k, replace=False))
            for _ in range(B * H * L)
        ]).reshape(B, H, L, k)
        return out.astype(np.int32)

    k_idx, q_idx = rand_idx(W), rand_idx(R)
    # softmax-like positive key values -> delta-rule stability for free
    k_val = np.exp(rng.standard_normal((B, H, L, W)))
    k_val /= k_val.sum(-1, keepdims=True)
    q_val = rng.standard_normal((B, H, L, R))
    v = rng.standard_normal((B, H, L, dvdim))
    g = -np.abs(rng.standard_normal((B, H, L, W))) * 0.1   # per-slot decay
    b = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L))))
    M0 = rng.standard_normal((H, N, dvdim)) * 0.1          # learned init

    args = tuple(map(jnp.array, (q_idx, q_val, k_idx, k_val, v, g, b, M0)))
    o_ref, M_ref = sdm.recurrent_sparse_delta_memory(*args)
    o_chk, M_chk = chunkwise_sparse_delta_memory(*args, chunk_size=C)

    print("max |o_chunk - o_rec|:", float(jnp.max(jnp.abs(o_chk - o_ref))))
    print("max |M_chunk - M_rec|:", float(jnp.max(jnp.abs(M_chk - M_ref))))
