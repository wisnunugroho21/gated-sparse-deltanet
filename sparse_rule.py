"""Sparse Delta Memory (SDM) — recurrence + Product-Key addressing in JAX.

Implements the token-mixer memory of Cabannes et al., "Sparse Delta Memory:
Scaling the State of Linear RNNs through Sparsity" (arXiv:2607.07386),
Section 3.1 (Eqs. 3-5) with the Product-Key Memory addressing of Section 2 /
Lample et al. (2019).

WHAT SDM CHANGES vs Gated DeltaNet (see rule.py). GDN/GDN-2 keep a DENSE state
S ∈ R^{dk×dv}: every token reads and writes all dk key-channel rows, so state
capacity is capped by dk and the per-token cost is O(dk·dv). SDM instead keeps
a LARGE explicit memory table M ∈ R^{N×dv} with N slots (N up to millions) and
touches only W write-slots and R read-slots per token, selected by Product-Key
Memory. The dense key-channel axis dk becomes a SPARSE slot axis N: the dense
key vector k_t ∈ R^{dk} becomes a sparse weight vector supported on W slots.
Per-token cost is O((W+R)·dv) — independent of N — so state capacity scales
without adding FLOPs (paper Fig. 1, Sec. 3.2).

THIS FILE — Phase 1: the token-by-token RECURRENT reference (the correctness
oracle, analogous to rule.py's _recurrent_single) plus the PKM addressing it
uses. The chunkwise-parallel training core (paper App. A) is a later phase.

FAITHFUL SDM (this phase). Built on the GDN base with a SINGLE input gate β
(paper Eq. 4), i.e. NO GDN-2 decoupling: the key-side erase gate b and the
value-side write gate w of rule.py are both dropped here. Re-introducing the
GDN-2 decoupling on top of the sparse memory is a deliberate later extension.

Per SDM head, at token t (paper Sec. 3.1):

  1. SPARSE KEY SELECTION (PKM, Eq. of Sec. 2 / Fig. 2).
       write scores s^w = k'_1 ⊕ k'_2   (outer sum of the two √N score halves)
       I^w_t = top-W(s^w),  κ_t = softmax(s^w[I^w_t])     (W write slots+weights)
       read  scores s^r = q'_1 ⊕ q'_2
       I^r_t = top-R(s^r),  ρ_t = softmax(s^r[I^r_t])      (R read  slots+weights)
     The outer-sum top-k never materializes all N scores: because
       top-k(s1 ⊕ s2) = top-k( top-k(s1) ⊕ top-k(s2) ),
     it reduces a k×k candidate set instead (Sec. 2; see _pkm_select).

  2. GATED DELTA WRITE (Eq. 3-4), only for the selected slots i ∈ I^w_t;
     UNSELECTED slots are frozen — not even decayed (segmented/lazy decay):
       M̃_t[i] = α_t · M_{t-1}[i]                          (forget, Eq. 3)
       r_t     = Σ_{i∈I^w_t} κ_t[i] · M̃_t[i]              (aggregate delta read)
       M_t[i]  = M̃_t[i] + β_t · κ_t[i] · (v_t − r_t)       (delta write, Eq. 4)
     α_t = exp(g_t) ∈ (0,1] is the PER-HEAD forget gate (g_t = log α_t ≤ 0,
     a scalar per token, unlike GDN-2's per-channel decay); β_t ∈ [0,1] is the
     per-head input gate; v_t ∈ R^{dv} is the value. The subtraction of the
     aggregate read r_t is the delta rule — it recovers GDN exactly when N=dk
     and all slots are selected (paper "Connection to GDN").

  3. SPARSE READ (Eq. 5), from the POST-write memory:
       y_t = Σ_{i∈I^r_t} ρ_t[i] · M_t[i]

Normalization, the SiLU output gate, and cross-head mixing (paper step 4) live
in the layer, not here — exactly as rule.py returns the raw recurrent output.

Selection (step 1) depends only on the inputs, not on the memory state, so it
is computed in PARALLEL for all L tokens before the sequential scan; only the
write/read recurrence (steps 2-3) is sequential.

Shape conventions (one head): the PKM score halves k'_1,k'_2,q'_1,q'_2 are
[L, root] with root = √N; v: [L, dv]; g, beta: [L]; M0: [N, dv]. Public entry
points add leading [B, H] axes via vmap. All math runs in fp32.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

# Compute dtype for all internal math (fp32, as in rule.py / paper App. D).
D_TYPE = jnp.float32


def _pkm_select(
    s1: jax.Array, s2: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """Product-Key top-k over the outer sum of two score halves (one token).

    Selects the `topk` largest entries of the full score vector
    s = s1 ⊕ s2 ∈ R^{root·root} (s[a·root + b] = s1[a] + s2[b]) WITHOUT
    materializing all root² scores, using the PKM identity
        top-k(s1 ⊕ s2) = top-k( top-k(s1) ⊕ top-k(s2) )
    (Lample et al., 2019; paper Sec. 2). The right-hand side is exact: if
    (a, b) is among the top-k of s then a is among the top-k of s1 and b among
    the top-k of s2 (any better a' gives k strictly larger pairs (a', b)), so
    the k×k candidate grid provably contains the true top-k.

    Args:
      s1, s2: [root]   the two score halves (root = √N).
      topk:   k, the number of slots to select (k ≤ root).
    Returns:
      (idx: [topk] int32 slot indices into N = root²,
       score: [topk] the corresponding s1[a] + s2[b] values, descending).
    """
    root = s1.shape[-1]

    # Reduce each half to its own top-k first: the winning pair's coordinates
    # can only come from these (proof above), so root²  ->  k²  candidates.
    t1v, t1i = lax.top_k(s1, topk)  # [topk] values / indices into the a-axis
    t2v, t2i = lax.top_k(s2, topk)  # [topk] values / indices into the b-axis

    # Outer sum of the two shortlists: every candidate score, [topk, topk].
    cand = t1v[:, None] + t2v[None, :]

    # Top-k of the k² candidates = the true top-k of the full N scores.
    score, flat = lax.top_k(cand.reshape(-1), topk)  # [topk]
    a = flat // topk  # row in the candidate grid -> which top-k of s1
    b = flat % topk   # col in the candidate grid -> which top-k of s2

    # Map (a, b) back to the flat slot index in N: slot = a_idx·root + b_idx,
    # the same bijection used to define s = s1 ⊕ s2.
    idx = t1i[a] * root + t2i[b]
    return idx.astype(jnp.int32), score


def _recurrent_single_sdm(
    kw1: jax.Array,
    kw2: jax.Array,
    qr1: jax.Array,
    qr2: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    M0: jax.Array,
    W: int,
    R: int,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token SDM reference forward for ONE (batch, head) pair.

    Direct scan of the sparse gated delta rule (paper Eqs. 3-5) with PKM
    addressing (Sec. 2). O(L·(W+R)·dv) plus the O(L·(root + W² + R²))
    selection — a trustworthy oracle for verifying the chunkwise path, and the
    form used at inference (one token at a time).

    Args:
      kw1, kw2: [L, root]  write score halves (root = √N); s^w = kw1 ⊕ kw2.
      qr1, qr2: [L, root]  read  score halves;             s^r = qr1 ⊕ qr2.
      v:        [L, dv]    value vectors (W_v x).
      g:        [L]        per-head log-decay, g = log α ≤ 0.
      beta:     [L]        per-head input gate β ∈ [0, 1].
      M0:       [N, dv]    initial memory (N = root²); learnable or zeros.
      W, R:     ints       number of write / read slots selected per token.
    Returns:
      (y: [L, dv], M_final: [N, dv])
    """
    kw1 = kw1.astype(D_TYPE)
    kw2 = kw2.astype(D_TYPE)
    qr1 = qr1.astype(D_TYPE)
    qr2 = qr2.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    beta = beta.astype(D_TYPE)
    M0 = M0.astype(D_TYPE)

    # ---- Selection is state-independent: compute it for ALL tokens at once --
    # PKM top-k of the write / read scores. vmap runs _pkm_select over the L
    # axis in parallel — nothing here depends on the memory M.
    w_idx, w_sc = jax.vmap(partial(_pkm_select, topk=W))(kw1, kw2)  # [L, W]
    r_idx, r_sc = jax.vmap(partial(_pkm_select, topk=R))(qr1, qr2)  # [L, R]

    # Softmax-normalized addressing weights (paper: "Read and write activations
    # use softmax-normalization"). Over the selected slots only.
    kappa = jax.nn.softmax(w_sc, axis=-1)  # [L, W]  write weights κ
    rho = jax.nn.softmax(r_sc, axis=-1)    # [L, R]  read  weights ρ

    alpha = jnp.exp(g)  # α_t = exp(g_t) ∈ (0,1], per-head forget gate   [L]

    def step(M, inp):
        # M: [N, dv].  Per-token slices:
        # wi [W], ka [W], ai scalar, bi scalar, vt [dv], ri [R], rh [R].
        wi, ka, ai, bi, vt, ri, rh = inp

        # Eq. 3: forget — decay ONLY the selected write slots (segmented/lazy
        # decay; unselected slots keep their value untouched).        [W, dv]
        Msel = ai * M[wi]

        # Aggregate delta-rule read: what the decayed memory returns along the
        # write addressing κ. This is the sparse analogue of GDN's M̃ᵀk.  [dv]
        r_t = ka @ Msel

        # Eq. 4: delta write — store the residual (v_t − r_t) at the selected
        # slots, gated by the input gate β and the per-slot weight κ.  [W, dv]
        Mnew = Msel + bi * ka[:, None] * (vt[None, :] - r_t[None, :])

        # Scatter the updated slots back. The W indices are distinct (distinct
        # (a,b) pairs -> distinct N indices), so .set is well-defined.
        M = M.at[wi].set(Mnew)

        # Eq. 5: sparse read from the POST-write memory, weighted by ρ.   [dv]
        y = rh @ M[ri]

        return M, y

    M_final, y = lax.scan(
        step, M0, (w_idx, kappa, alpha, beta, v, r_idx, rho)
    )
    return y, M_final


def _batchify_sdm(fn):
    """Lift a per-head SDM function to batched [B, H, ...] inputs.

    Pure plumbing, mirroring rule.py's _batchify: vmap over heads (axis 1),
    then over batch (axis 0). The eight array arguments map over their leading
    axis; M0 simply has no L axis. W and R stay static (closed over by `fn`).
    """
    in_axes = (0, 0, 0, 0, 0, 0, 0, 0)
    over_heads = jax.vmap(fn, in_axes=in_axes, out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=in_axes, out_axes=(0, 0))


def recurrent_sparse_delta_memory(
    kw1: jax.Array,
    kw2: jax.Array,
    qr1: jax.Array,
    qr2: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    M0: jax.Array,
    W: int = 64,
    R: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token Sparse Delta Memory forward (reference / inference path).

    Faithful SDM (single input gate β; no GDN-2 erase/write decoupling), paper
    Eqs. 3-5 with PKM addressing (Sec. 2).

    Args:
      kw1, kw2: [B, H, L, root]  write score halves (root = √N).
      qr1, qr2: [B, H, L, root]  read  score halves.
      v:        [B, H, L, dv]    values.
      g:        [B, H, L]        per-head log-decay (≤ 0).
      beta:     [B, H, L]        per-head input gate (∈ [0, 1]).
      M0:       [B, H, N, dv]    initial memory (N = root²).
      W, R:     number of write / read slots per token.
    Returns:
      (y: [B, H, L, dv], M_final: [B, H, N, dv])
    """
    fn = partial(_recurrent_single_sdm, W=W, R=R)
    return _batchify_sdm(fn)(kw1, kw2, qr1, qr2, v, g, beta, M0)
