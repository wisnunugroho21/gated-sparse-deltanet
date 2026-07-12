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

Two implementations of the same recurrence:

  _recurrent_sdm_single   token-by-token scan of Eqs. 3-5. O(L·(W+R)·dv),
                          trivially correct — the verification oracle and
                          the decode path.
  _chunkwise_sdm_single   chunkwise WY form (paper Appendix A), the sparse
                          analogue of rule.py's chunkwise cores: sequential
                          depth L/C instead of L. Interaction matrices are
                          built by slot-index matching (Eq. 7) with the
                          exponent masked BEFORE exp (rule.py's "pairwise"
                          discipline) — no decay-range limit. See the
                          function docstring for the dense↔sparse mapping
                          and the O(C²·(W²+R·W)) cost of the JAX-friendly
                          equality-mask stand-in for the paper's two-pointer
                          CUDA kernel.

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


def _chunkwise_sdm_single(
    iw: jax.Array,
    kw: jax.Array,
    ir: jax.Array,
    qr: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    M0: jax.Array,
    *,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise-parallel SDM forward for ONE (batch, head) pair (App. A).

    THE DENSE↔SPARSE MAPPING. Scattered to a dense N-vector κ_r (write
    keys) with write mask m_r, the SDM recurrence is exactly rule.py's
    Gated Delta Rule-2 with N key channels, PER-SLOT decay
    log α_r[i] = g_r · m_r[i] (written slots decay, untouched slots don't)
    and tied gates b = w = β. Rule.py's WY quantities therefore carry
    over verbatim, with the per-channel cumsum G replaced by the SEGMENTED
    per-slot cumulative log-decay

        λ_{t,n} = Σ_{s≤t} g_s · 1[slot(t,n) ∈ I_w(s)]   (inclusive)

    evaluated only at the O(C·(W+R)) selected (token, slot) occurrences —
    the paper's "segmented prefix sum over slot indices". Every inner
    product over N collapses to a sum over MATCHING slot indices:

      T[r,s]   = β_r Σ_{n,m: Iw_r[n]=Iw_s[m]} k_r⁽ⁿ⁾k_s⁽ᵐ⁾ e^{λ_{r,n}−λ_{s,m}}, s<r
      Aqk[t,s] =     Σ_{n,m: Ir_t[n]=Iw_s[m]} q_t⁽ⁿ⁾k_s⁽ᵐ⁾ e^{λʳ_{t,n}−λ_{s,m}}, s≤t

    (the paper's A and QK, Eq. 7 / App. A.2 — Aqk's inclusive diagonal is
    the post-write read of a token's own write). Matched exponents are
    decays over (s, r] of a SHARED slot, hence ≤ 0: masking the exponent
    BEFORE exp (rule.py's pairwise discipline) makes the core
    overflow-proof with NO decay-range limit — important here because the
    paper observes near-zero forget gates on some slots (App. B).

    Phase 1 (parallel over chunks; App. A.1): λ, T, Aqk, and ONE unit
    lower-triangular solve with stacked RHS [diag(β)V | diag(β)] giving
      ΔVconst = (I+T)⁻¹ diag(β) V   and   Bmat = (I+T)⁻¹ diag(β)
    (the paper's ΔVconst and −B). Both are independent of the running
    memory — that is what lets all chunks precompute in parallel.

    Phase 2 (sequential scan over chunks; App. A.1): per chunk, with the
    carried memory M (rule.py's S0 per chunk):
      retrieved_t = Σ_n k⁽ⁿ⁾e^{λ_{t,n}} M[Iw_t[n]]        (gather)
      inter_t     = Σ_n q⁽ⁿ⁾e^{λʳ_{t,n}} M[Ir_t[n]]       (gather)
      δv          = ΔVconst − Bmat · retrieved             (correct)
      y           = inter + Aqk · δv                       (output)
      M[i]       *= e^{Λ_C[i]}  — realized as a log-space scatter-add of
                    g over write occurrences, then one exp·multiply
                    (per-occurrence multiplicative decay composes exactly:
                    Λ_C[i] = Σ_{s: i∈Iw_s} g_s);
      M[Iw_s[n]] += k⁽ⁿ⁾e^{λtail_{s,n}} δv_s  — scatter-add, where
                    λtail_{s,n} = Λ_C[i] − λ_{s,n} is the decay the slot
                    still accumulates AFTER token s (the paper's "apply
                    per-slot decay and scatter the delta-v back").

    Cost note: the index matching is a [C, C, W, W] (and [C, C, R, W])
    equality einsum — O(C²W²) work and memory per chunk, the JAX-friendly
    stand-in for the paper's O(C²W) two-pointer CUDA merge (App. A.2).
    Fine at this repo's scale; a sort + searchsorted formulation is the
    upgrade path if W grows.

    Args:
      iw, kw: [L, W]   ir, qr: [L, R]   v: [L, dv]   g, beta: [L]
      M0: [N, dv]      chunk_size: C (L must be divisible by C)
    Returns:
      (y: [L, dv], M_final: [N, dv])
    """
    L, W = kw.shape
    dv = v.shape[-1]
    C = chunk_size
    if L % C:
        raise ValueError(
            f"sequence length L={L} must be divisible by chunk_size={C}")
    n = L // C  # number of chunks

    def to_chunks(x):
        # [L, ...] -> [n, C, ...]: one leading axis per chunk, so every
        # chunk-local quantity below is computed for all n chunks at once.
        return x.reshape(n, C, *x.shape[1:])

    Iw, Ir = to_chunks(iw), to_chunks(ir)  # [n, C, W] / [n, C, R] (int)
    Kw = to_chunks(kw.astype(D_TYPE))  # [n, C, W]
    Qr = to_chunks(qr.astype(D_TYPE))  # [n, C, R]
    V = to_chunks(v.astype(D_TYPE))  # [n, C, dv]
    G = to_chunks(g.astype(D_TYPE))  # [n, C]  log-decay, ≤ 0
    Bt = to_chunks(beta.astype(D_TYPE))  # [n, C]
    M0 = M0.astype(D_TYPE)  # [N, dv]

    eye = jnp.eye(C, dtype=D_TYPE)
    incl = jnp.tril(jnp.ones((C, C), D_TYPE))  # s ≤ r (reads, λ cumsums)
    strict = jnp.tril(jnp.ones((C, C), D_TYPE), k=-1)  # s < r (T, λtail via .T)

    # ---- Phase 1: chunk-local precompute (parallel over the n chunks) ----

    # Slot-index matching. EQw[c,a,b,n,m] = 1 iff token a's n-th write slot
    # equals token b's m-th write slot; slots are distinct WITHIN a token,
    # so Σ_m EQw ∈ {0,1} is the membership test "slot(a,n) ∈ I_w(b)".
    EQw = Iw[:, :, None, :, None] == Iw[:, None, :, None, :]  # [n,C,C,W,W]
    hitw = jnp.sum(EQw, axis=-1).astype(D_TYPE)  # [n,C,C,W]
    EQr = Ir[:, :, None, :, None] == Iw[:, None, :, None, :]  # [n,C,C,R,W]
    hitr = jnp.sum(EQr, axis=-1).astype(D_TYPE)  # [n,C,C,R]

    # Segmented per-slot cumulative log-decay (App. A.1 "segmented
    # cumulative decay"), inclusive of the token's own write — the write
    # reads the α_r-decayed state (Eq. 3), the read is post-write:
    #   lam_w[c,r,n] = Σ_{s≤r} g_s · 1[slot(r,n) ∈ I_w(s)]      [n, C, W]
    #   lam_r[c,t,n] = Σ_{s≤t} g_s · 1[read-slot(t,n) ∈ I_w(s)] [n, C, R]
    #   lam_tail[c,s,n] = Σ_{u>s} g_u · 1[slot(s,n) ∈ I_w(u)]   [n, C, W]
    # lam_tail is the decay a slot still accumulates AFTER its write at s
    # (probe axis first, target axis contracted with g — strict upper).
    lam_w = jnp.einsum("crsn,rs,cs->crn", hitw, incl, G)
    lam_r = jnp.einsum("ctsn,ts,cs->ctn", hitr, incl, G)
    lam_tail = jnp.einsum("csun,us,cu->csn", hitw, strict, G)

    # Interaction matrices (Eq. 7 / App. A.2), pairwise log-space form.
    # Matched exponents λ_{r,n} − λ_{s,m} are the shared slot's decay over
    # (s, r] — ≤ 0 by construction. Mask BEFORE exp (unmatched or
    # anti-causal exponents can be > 0 and would give inf·0 = NaN after).
    dW = lam_w[:, :, None, :, None] - lam_w[:, None, :, None, :]
    mskW = EQw & (strict[None, :, :, None, None] > 0)
    T = Bt[:, :, None] * jnp.einsum(
        "crsnm,crn,csm->crs", jnp.exp(jnp.where(mskW, dW, -jnp.inf)), Kw, Kw
    )  # write-write "A", scaled by β_r -> Msys = I + T (App. A.1)

    dR = lam_r[:, :, None, :, None] - lam_w[:, None, :, None, :]
    mskR = EQr & (incl[None, :, :, None, None] > 0)
    Aqk = jnp.einsum(
        "ctsnm,ctn,csm->cts", jnp.exp(jnp.where(mskR, dR, -jnp.inf)), Qr, Kw
    )  # read-write "QK", inclusive diagonal (post-write read)

    # ONE stacked triangular solve (rule.py's solver discipline):
    #   (I + T) [ΔVconst | Bmat] = [diag(β)V | diag(β)]
    # ΔVconst is the intra-chunk delta-v assuming reads from the
    # chunk-initial memory; Bmat corrects it with the actual retrieved
    # rows in Phase 2 (δv = ΔVconst − Bmat·retrieved — the paper's
    # B = −Msys⁻¹diag(β)).
    X = jax.scipy.linalg.solve_triangular(
        eye + T,
        jnp.concatenate([Bt[..., None] * V, Bt[..., None] * eye[None]], axis=-1),
        lower=True,
        unit_diagonal=True,
    )
    dv_const, Bmat = X[..., :dv], X[..., dv:]  # [n, C, dv], [n, C, C]

    # Decay-absorbed sparse factors (all exponents ≤ 0 — safe):
    k_eff = Kw * jnp.exp(lam_w)  # gathers vs chunk-initial M (paper's k_eff)
    q_eff = Qr * jnp.exp(lam_r)  # inter-chunk read (paper's q_eff)
    w_coef = Kw * jnp.exp(lam_tail)  # tail-decayed scatter coefficients
    g_occ = jnp.repeat(G, W, axis=-1)  # per-occurrence log-decay  [n, C·W]
    Iw_flat = Iw.reshape(n, C * W)  # flat write occurrences (C-major)

    Nslots = M0.shape[0]

    # ---- Phase 2: sequential scan over chunks (App. A.1) ------------------
    def chunk_step(M, inp):
        Iw_c, Ir_c, keff_c, qeff_c, dvc_c, Bmat_c, Aqk_c, wcoef_c, gocc_c, iwf_c = inp

        # 1. Read (gathers from the carried memory):
        retrieved = jnp.einsum("cw,cwd->cd", keff_c, M[Iw_c])  # [C, dv]
        inter = jnp.einsum("cr,crd->cd", qeff_c, M[Ir_c])  # [C, dv]

        # 2. Correct: earlier tokens in the chunk already modified what
        # later tokens read — Bmat folds that in (App. A.1 "Correct").
        dvv = dvc_c - Bmat_c @ retrieved  # δv  [C, dv]

        # 4. Output (before the state update touches M — y only needs the
        # chunk-initial memory plus the intra-chunk correction):
        y = inter + Aqk_c @ dvv  # [C, dv]

        # 3. Write: per-slot decay then scatter-add (App. A.1 "Write").
        # Multiplicative decay composes per occurrence — a slot written at
        # s₁ < s₂ must shrink by e^{g_{s₁}+g_{s₂}} — so accumulate logs
        # with a scatter-ADD and apply one exp·multiply (scatter-add has
        # well-defined duplicate semantics and a solid gradient rule).
        log_dec = jnp.zeros((Nslots,), D_TYPE).at[iwf_c].add(gocc_c)
        M = M * jnp.exp(log_dec)[:, None]
        M = M.at[iwf_c].add(wcoef_c.reshape(-1)[:, None] * jnp.repeat(dvv, W, axis=0))

        return M, y

    M_final, Y = lax.scan(
        chunk_step,
        M0,
        (Iw, Ir, k_eff, q_eff, dv_const, Bmat, Aqk, w_coef, g_occ, Iw_flat),
    )
    return Y.reshape(L, dv), M_final


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


def chunkwise_sdm(
    iw: jax.Array,
    kw: jax.Array,
    ir: jax.Array,
    qr: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    M0: jax.Array,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise-parallel Sparse Delta Memory forward (training path).

    Exact same recurrence as recurrent_sdm with sequential depth L/C
    instead of L (paper Appendix A; see _chunkwise_sdm_single for the
    dense↔sparse mapping and the two-phase decomposition). Overflow-proof
    for any decay strength (pairwise log-space exponents, all ≤ 0).

    Args:
      iw: [B, H, L, W] int   kw: [B, H, L, W]     write indices / key values
      ir: [B, H, L, R] int   qr: [B, H, L, R]     read indices / query values
      v:  [B, H, L, dv]      g, beta: [B, H, L]   values / log-decay / gate
      M0: [B, H, N, dv]                           initial memory table
      chunk_size: intra-chunk length C (L must be divisible by C).
    Returns:
      (y: [B, H, L, dv], M_final: [B, H, N, dv])
    """
    L = kw.shape[2]
    if L % chunk_size:
        raise ValueError(
            f"sequence length L={L} must be divisible by chunk_size={chunk_size}")

    def fun(iw_, kw_, ir_, qr_, v_, g_, beta_, M0_):
        return _chunkwise_sdm_single(
            iw_, kw_, ir_, qr_, v_, g_, beta_, M0_, chunk_size=chunk_size)

    return _batchify(fun)(iw, kw, ir, qr, v, g, beta, M0)
