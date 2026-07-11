"""Gated Delta Rule-2 — training-time recurrence cores in JAX.

Implements the token mixer of Hatamizadeh, Choi, Kautz, "Gated DeltaNet-2:
Decoupling Erase and Write in Linear Attention" (arXiv:2605.22791).
Equation numbers below are cited as "main text / Appendix A" where the
equation appears in both.

The state S ∈ R^{dk×dv} is an associative memory mapping key directions to
value rows; the output reads it with the query, o_t = S_tᵀ q_t (Eq. 1).
Each token applies three edits (Eq. 10/29):

    S_r = (I − k_r e_rᵀ) Diag(α_r) S_{r−1} + k_r z_rᵀ,
    e_r = b_r ⊙ k_r,   z_r = w_r ⊙ v_r,   α_r = exp(g_r) ∈ (0,1]

  1. FORGET: Diag(α_r) shrinks each key-channel row (passive decay).
  2. ERASE:  −k_r e_rᵀ S removes what the memory returns along the gated
     read direction e_r; the erase gate b picks WHICH key channels.
  3. WRITE:  +k_r z_rᵀ inserts the gated value; the write gate w picks
     WHICH value channels.
The "-2" is the decoupling: classic (Gated) DeltaNet ties erase and write
to one scalar β_r; here b (key side) and w (value side) are independent
per-channel gates.

Three implementations of the same recurrence:

  _recurrent_single           token-by-token scan of Eq. 9/29. O(L·dk·dv),
                              trivially correct — the verification oracle.
  _chunkwise_single_faithful  chunkwise WY form exactly as printed in the
                              paper (Eqs. 18-25 / 30-44). Overflows fp32
                              when a chunk's cumulative log-decay G drops
                              below ≈ −88 (it materializes exp(−G)).
  _chunkwise_single_optim     same algebra with per-chunk exponent
                              centering: all within-chunk exponents are
                              shifted by c = G_C/2, which cancels exactly
                              in every product (they only consume
                              differences of G) and is re-attached against
                              S0 via diag(exp(c)). Doubles the safe range
                              to |G_C| ≈ 176; used by the public entry point.

Chunkwise idea (Sec. 3.3 / App. A): substituting S_r = Diag(γ_r) Ŝ_r with
γ_r = exp(Σ_{i≤r} g_i) removes the decay from the recurrence — Ŝ follows a
PURE delta rule whose C-step unroll is a product of rank-one corrections.
Stacking its residual rows ρ_r = z_r − ē_rᵀ Ŝ_{r−1} gives one unit
lower-triangular system (I + T) R = Z − Ē S0 (Eq. 39), solved by forward
substitution; the output and end-of-chunk state are then dense matmuls.
Only the cross-chunk state remains sequential.

The hand-derived backward of Appendix B is intentionally not implemented:
jax.grad differentiates through solve_triangular and the elementwise gate
products and reconstructs exactly those vector-Jacobian products. A manual
backward is only needed for a fused Triton/Pallas kernel.

Shape conventions (one head): q, k, g, b: [L, dk]; v, w: [L, dv];
S0: [dk, dv]. Public entry points add leading [B, H] axes via vmap.
All math runs in fp32 (paper App. D).
"""

import jax
import jax.numpy as jnp
from jax import lax

# Compute dtype for all internal math (paper App. D: gate/decay math in fp32).
D_TYPE = jnp.float32

def _recurrent_single(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token reference forward for ONE (batch, head) pair.

    Direct scan of Eq. 9 (three-line factored form, algebraically equal to
    the (I − k e ᵀ) Diag(α) form of Eq. 10/29). O(L·dk·dv), no triangular
    solve — a trustworthy oracle for verifying the chunkwise paths, and the
    form used at inference (one token at a time).

    Args:
      q, k, g, b: [L, dk]   v, w: [L, dv]   S0: [dk, dv]
    Returns:
      (O: [L, dv], S_final: [dk, dv])
    """
    q = q.astype(D_TYPE)
    k = k.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    b = b.astype(D_TYPE)
    w = w.astype(D_TYPE)
    S0 = S0.astype(D_TYPE)

    alpha = jnp.exp(g)  # Eq. 12/30: α = exp(g), g ≤ 0 so α ∈ (0,1]  [L, dk]
    e = b * k  # Eq. 8: e = b ⊙ k — erase gate picks key channels    [L, dk]
    z = w * v  # Eq. 8: z = w ⊙ v — write gate picks value channels  [L, dv]

    def step(S, inp):
        # S: [dk, dv]; per-token slices qt, kt, at, et: [dk], zt: [dv].
        qt, kt, at, et, zt = inp

        # Column vectors for the rank-one linear algebra below.
        qt = qt[:, None]  # [dk, 1]
        kt = kt[:, None]  # [dk, 1]
        at = at[:, None]  # [dk, 1]
        et = et[:, None]  # [dk, 1]
        zt = zt[:, None]  # [dv, 1]

        # Eq. 9:  S̄_t = Diag(α_t) S_{t−1} — passive forgetting: each
        # key-channel row shrinks by its own α ∈ (0,1].            [dk, dv]
        S_bar = at * S

        # Eq. 9:  r_t = S̄_tᵀ e_t — RECALL: what the decayed memory returns
        # along the gated erase direction. With b = 1 this is the classic
        # delta-rule read S̄ᵀk; b < 1 mutes key channels, protecting their
        # stored content from the upcoming subtraction.             [dv, 1]
        r_t = S_bar.T @ et

        # Eq. 9/15:  S_t = S̄_t + k_t (z_t − r_t)ᵀ — delta write: store only
        # the RESIDUAL between the gated target z_t and the recalled r_t, at
        # key k_t. If memory already holds the target, nothing is written
        # (no unbounded accumulation, unlike vanilla linear attention).
        # Also one gradient step on ½‖Sᵀk_t − target‖² (fast-weight view,
        # Eqs. 13-15).                                             [dk, dv]
        S_new = S_bar + kt * (zt - r_t).T

        # Eq. 1:  o_t = S_tᵀ q_t — read the post-update memory.     [dv, 1]
        o_t = S_new.T @ qt

        return S_new, o_t

    # o stacks per-token outputs: [L, dv, 1] -> squeeze to [L, dv].
    S_final, o = lax.scan(step, S0, (q, k, alpha, e, z))
    return o.squeeze(-1), S_final

def _chunkwise_single_faithful(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair, literal paper form.

    Implements Eqs. 18-25 (main text) = Eqs. 30-44 (App. A) with the exact
    factors printed in the paper. Kept as a readable reference; it
    materializes exp(-G), so it overflows fp32 once the within-chunk
    cumulative log-decay G goes below ≈ -88. Prefer _chunkwise_single_optim.

    Args:
      q, k, g, b: [L, dk]   query / key / log-decay (g ≤ 0) / erase gate
      v, w:       [L, dv]   value / write gate
      S0:         [dk, dv]  state entering the sequence
      chunk_size: C; L must be divisible by C.
    Returns:
      (O: [L, dv], S_final: [dk, dv])
    """
    L, dk = k.shape
    dv = v.shape[-1]
    C = chunk_size
    N = L // C  # number of chunks

    def to_chunks(x):
        # [L, d] -> [N, C, d]: one leading axis per chunk, so every
        # chunk-local quantity below is computed for all N chunks at once.
        return x.reshape(N, C, x.shape[-1]).astype(D_TYPE)

    q, k, v = to_chunks(q), to_chunks(k), to_chunks(v)   # [N, C, dk/dk/dv]
    g, b, w = to_chunks(g), to_chunks(b), to_chunks(w)   # [N, C, dk/dk/dv]

    eye = jnp.eye(C, dtype=D_TYPE)  # [C, C]
    S0 = S0.astype(D_TYPE)          # [dk, dv]

    # ---- Chunk-local precompute (parallel over the N chunks) -------------
    # The cumsum restarts at each chunk boundary, which realizes both
    # γ_0 = 1 (Eq. 18/30) and the normalized init Ŝ_0 = S_[n] (Eq. 31).

    # Eq. 18/30:  G_r = Σ_{i≤r} g_i (inclusive, within chunk).   [N, C, dk]
    # g ≤ 0 is the per-token log-decay, so γ_r = exp(G_r) = Π_{i≤r} α_i is
    # the total shrinkage a key channel accumulated since chunk start. A
    # write at step s read at step r survives with γ_r/γ_s = exp(G_r − G_s):
    # summing logs turns the running product of gates into a cumsum.
    G = jnp.cumsum(g, axis=1)

    # Eq. 18/30:  γ_r = exp(G_r)                                  [N, C, dk]
    gamma = jnp.exp(G)

    # Total chunk decay γ_C (last row): how much of the chunk-entry state
    # survives to the end of the chunk (the carry in Eq. 23/40).   [N, dk]
    gamma_C = gamma[:, -1]

    # --- Decay normalization: S_r = Diag(γ_r) Ŝ_r (Eq. 31) removes Diag(α)
    # from the recurrence and moves it into the rank-one factors. The ratio
    # γ_r/γ_s is split: γ^{-1} goes on the write side (K̄), γ on the read
    # side (Ē).
    # Eq. 19/32/33:  K̄ = γ^{-1} ⊙ K                              [N, C, dk]
    Kbar = k * jnp.exp(-G)

    # Eq. 20/33:  Ē = γ ⊙ (B ⊙ K), with e_r = b_r ⊙ k_r (Eq. 8). [N, C, dk]
    # Row r pairs with a K̄ row s<r as ē_rᵀ k̄_s = e_rᵀ Diag(γ_r/γ_s) k_s:
    # how much of the association written at s, decayed until r, lies along
    # the gated erase direction e_r.
    Ebar = gamma * (b * k)

    # Eq. 8, 20/33:  Z = W ⊙ V — gated write targets.             [N, C, dv]
    # No γ here: values live on the dv axis, decay acts on key channels.
    Z = w * v

    # Eq. 24/43:  Q_γ, row r = γ_r ⊙ q_r.                         [N, C, dk]
    # The query reads the chunk-entry state decayed down to its own step r.
    Qg = gamma * q

    # --- WY triangular solve (the parallelization) ---
    # Eq. 21/34:  T = tril(Ē K̄ᵀ, −1), T_rs = ē_rᵀ k̄_s (s < r).   [N, C, C]
    # T_rs measures how strongly token r's erase overlaps token s's decayed
    # write. Strictly lower triangular = causality (r only erases the past).
    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)

    # Eq. 21/34:  A = (I + T)^{-1}.                                [N, C, C]
    # Residuals obey ρ_r = (z_r − ē_rᵀ S0) − Σ_{s<r} T_rs ρ_s (Eq. 38): each
    # edit first accounts for every earlier edit it partially erases.
    # Stacked: (I + T) R = Z − Ē S0. I + T is unit lower-triangular, so one
    # batched forward substitution replaces the C-step sequential recurrence.
    A = jax.scipy.linalg.solve_triangular(
        eye + T, jnp.broadcast_to(eye, T.shape), lower=True, unit_diagonal=True
    )

    # Eq. 22/34:  Y = A Ē — erase-side auxiliary.                 [N, C, dk]
    Y = A @ Ebar

    # Eq. 22/34:  U = A Z — write-side auxiliary.                 [N, C, dv]
    # Y and U solve the SAME triangular system with different right-hand
    # sides (row recurrences Eqs. 45/46), sharing the one inverse A. Both
    # are independent of S0 — that is what lets all chunks precompute them
    # in parallel before the sequential scan.
    U = A @ Z

    # Eq. 25/43:  (A_qk)_rs = 1_{r≥s} q_rᵀ Diag(γ_r/γ_s) k_s.      [N, C, C]
    # Decay-aware causal attention scores: query r attends to the write at
    # s ≤ r with the key contracted channel-wise by the decay accumulated
    # between s and r. tril INCLUDES the diagonal (a token reads its own
    # write, ratio = 1).
    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))

    # Eq. 23/41:  (K_tail)_r = (γ_C/γ_r) ⊙ k_r.                   [N, C, dk]
    # The write at step r keeps decaying for the rest of the chunk, so it
    # enters the end-of-chunk state with the leftover factor γ_C/γ_r.
    Ktail = k * (gamma_C[:, None, :] / gamma)

    # ---- Cross-chunk recurrence: the ONLY sequential part -----------------
    # S is the raw chunk-entry state S_[n] (NOT decay-normalized); each step
    # is three small matmuls.
    def chunk_step(
        S: jax.Array,
        inp: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        # S: [dk, dv].  Per-chunk slices:
        # Y_n [C, dk], U_n [C, dv], Aqk_n [C, C], Qg_n [C, dk],
        # Ktail_n [C, dk], gamma_C_n [dk].
        Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n = inp

        # Eq. 35:  R = U − Y S0.                                    [C, dv]
        # Row r is the residual ρ_r = z_r − ē_rᵀ Ŝ_{r−1} (Eq. 37): what is
        # left to write at step r after subtracting what the decayed,
        # already-edited memory returns along the erase direction. U covers
        # the intra-chunk part; −Y S0 corrects for inherited content. R is
        # exactly the chunk's set of rank-one updates:
        # Ŝ_r = S0 + Σ_{s≤r} k̄_s ρ_sᵀ (Eq. 36).
        R = U_n - Y_n @ S

        # Eq. 24/44:  O = Q_γ S0 + A_qk R.                          [C, dv]
        # Two read paths: history (query reads the chunk-entry state through
        # its decay γ_r) + intra-chunk (causal scores against this chunk's
        # residual writes). This is o_r = S_rᵀ q_r unrolled through Eq. 36.
        o = Qg_n @ S + Aqk_n @ R

        # Eq. 23/40:  S_[n+1] = Diag(γ_C) S0 + K_tailᵀ R.          [dk, dv]
        # Hand-off to the next chunk: old state after a full chunk of decay
        # (erasures folded into R), plus every residual write re-keyed by
        # its tail-decayed key. gamma_C broadcasts over key-channel rows.
        S_new = gamma_C_n[:, None] * S + Ktail_n.T @ R

        return S_new, o

    # scan carries S across chunks; stacked outputs o: [N, C, dv].
    S_final, o = lax.scan(chunk_step, S0, (Y, U, Aqk, Qg, Ktail, gamma_C))
    return o.reshape(L, dv), S_final


def _chunkwise_single_optim(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise WY forward for ONE (batch, head) pair, exponent-centered.

    Same algebra as _chunkwise_single_faithful (Eqs. 18-25 / 30-44) but
    numerically safe: the faithful form materializes exp(-G) ∈ [1, e^{|G_C|}],
    which overflows fp32 for |G_C| > ~88 per chunk. Every product consumed
    downstream only involves the BOUNDED ratios exp(G_r − G_s), s ≤ r, and
    exp(G_C − G_r), so all within-chunk exponents are shifted by c = G_C/2
    per channel:
      * the shift cancels exactly wherever two centered factors meet
        (T, A_qk, and K_tail is formed directly in log-space);
      * it survives only where an absolute γ meets the state (Y S0, Q_γ S0),
        where it is re-attached by pre-scaling S0 with diag(exp(c)),
        exp(c) ≤ 1 — always safe.
    Result is bit-for-bit the same recurrence with the safe per-chunk decay
    range doubled to |G_C| ≈ 176; beyond that, reduce chunk_size.

    Args / returns: identical to _chunkwise_single_faithful.
    """
    L, dk = k.shape
    dv = v.shape[-1]
    C = chunk_size
    N = L // C  # number of chunks

    def to_chunks(x):
        # [L, d] -> [N, C, d]
        return x.reshape(N, C, x.shape[-1]).astype(D_TYPE)

    q, k, v = to_chunks(q), to_chunks(k), to_chunks(v)   # [N, C, dk/dk/dv]
    g, b, w = to_chunks(g), to_chunks(b), to_chunks(w)   # [N, C, dk/dk/dv]

    eye = jnp.eye(C, dtype=D_TYPE)  # [C, C]
    S0 = S0.astype(D_TYPE)          # [dk, dv]

    # Eq. 18/30:  G_r = Σ_{i≤r} g_i, within-chunk cumulative log-decay.
    # [N, C, dk]; see the faithful variant for the γ_r/γ_s reading.
    G = jnp.cumsum(g, axis=1)

    # Total chunk log-decay G_C (last row) and its exp, the carry
    # coefficient of Eq. 23/40. γ_C = exp(G_C) ≤ 1 never overflows. [N, dk]
    G_C = G[:, -1]
    gamma_C = jnp.exp(G_C)

    # Exponent centering: Gc = G − c with c = G_C/2 per channel.  [N, C, dk]
    # G spans [G_C, 0] within a chunk; Gc spans ±|G_C|/2, halving the
    # largest exponent ever materialized.
    Gc = G - 0.5 * G_C[:, None, :]

    # exp(c) ≤ 1: per-chunk state pre-scale that re-attaches the shift
    # against S0 inside chunk_step.                                  [N, dk]
    delta = jnp.exp(0.5 * G_C)

    # Eq. 19/32/33 centered:  K̄ = exp(c−G) ⊙ K (paper: γ^{-1} ⊙ K).
    # [N, C, dk]; max exponent |G_C|/2 instead of |G_C|.
    Kbar = k * jnp.exp(-Gc)

    # Eq. 20/33 centered:  Ē = exp(G−c) ⊙ (B ⊙ K), e_r = b_r ⊙ k_r (Eq. 8).
    # [N, C, dk]. Pairing rows restores the true ratio:
    # ē_rᵀ k̄_s = e_rᵀ Diag(exp(G_r − G_s)) k_s — c cancels.
    Ebar = jnp.exp(Gc) * (b * k)

    # Eq. 8, 20/33:  Z = W ⊙ V — gated write targets (no γ: decay lives on
    # the key axis).                                              [N, C, dv]
    Z = w * v

    # Eq. 24/43 centered:  Q_γ row r = exp(G_r − c) ⊙ q_r.        [N, C, dk]
    # The missing exp(c) is restored by pairing with diag(exp(c)) S0.
    Qg = jnp.exp(Gc) * q

    # Eq. 21/34:  T = tril(Ē K̄ᵀ, −1), T_rs = ē_rᵀ k̄_s (s < r).   [N, C, C]
    # Overlap of edit r with the decayed write s; centering cancels here.
    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)

    # Eq. 21/34:  A = (I + T)^{-1} via one batched forward substitution —
    # the closed form of the causal chain of corrections
    # ρ_r = (z_r − ē_rᵀ S0) − Σ_{s<r} T_rs ρ_s (Eq. 38).           [N, C, C]
    A = jax.scipy.linalg.solve_triangular(
        eye + T, jnp.broadcast_to(eye, T.shape), lower=True, unit_diagonal=True
    )

    # Eq. 22/34:  Y = A Ē (erase side), U = A Z (write side) — same
    # triangular system, two right-hand sides (Eqs. 45/46); both are
    # S0-independent, hence precomputable for all chunks in parallel.
    Y = A @ Ebar   # [N, C, dk]
    U = A @ Z      # [N, C, dv]

    # Eq. 25/43:  (A_qk)_rs = 1_{r≥s} q_rᵀ Diag(γ_r/γ_s) k_s.      [N, C, C]
    # Decay-aware causal scores; exp(Gc_r)·exp(−Gc_s) = exp(G_r − G_s), so
    # the centering cancels. tril includes the diagonal (self-read, ratio 1).
    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))

    # Eq. 23/41:  (K_tail)_r = (γ_C/γ_r) ⊙ k_r, formed in log-space as
    # exp(G_C − G_r): under strong decay the faithful ratio divides two
    # underflowed denormals (0/0 → NaN).                          [N, C, dk]
    Ktail = k * jnp.exp(G_C[:, None, :] - G)

    # ---- Cross-chunk recurrence: the ONLY sequential part -----------------
    def chunk_step(
        S: jax.Array,
        inp: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        # S: [dk, dv].  Per-chunk slices: Y_n [C, dk], U_n [C, dv],
        # Aqk_n [C, C], Qg_n [C, dk], Ktail_n [C, dk],
        # gamma_C_n [dk], delta_n [dk].
        Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n, delta_n = inp

        # diag(exp(c)) S0 — re-attaches the centering shift so that
        # Y_n S_c == (A Ē) S0 and Qg_n S_c == Q_γ S0 hold exactly. [dk, dv]
        S_c = delta_n[:, None] * S

        # Eq. 35:  R = U − Y S0 — stacked residuals ρ_r = z_r − ē_rᵀ Ŝ_{r−1}
        # (Eq. 37): the rank-one updates this chunk applies,
        # Ŝ_r = S0 + Σ_{s≤r} k̄_s ρ_sᵀ (Eq. 36).                     [C, dv]
        R = U_n - Y_n @ S_c

        # Eq. 24/44:  O = Q_γ S0 + A_qk R — history read + intra-chunk read;
        # o_r = S_rᵀ q_r (Eq. 1) unrolled through Eq. 36.            [C, dv]
        o = Qg_n @ S_c + Aqk_n @ R

        # Eq. 23/40:  S_[n+1] = Diag(γ_C) S0 + K_tailᵀ R — decayed carry of
        # the old state plus the tail-decayed residual writes.      [dk, dv]
        S_new = gamma_C_n[:, None] * S + Ktail_n.T @ R

        return S_new, o

    # scan carries S across chunks; stacked outputs o: [N, C, dv].
    S_final, o = lax.scan(chunk_step, S0, (Y, U, Aqk, Qg, Ktail, gamma_C, delta))
    return o.reshape(L, dv), S_final


def _batchify(fn):
    """Lift a per-head function to batched [B, H, ...] inputs.

    Pure plumbing, no math: vmap over heads (axis 1), then over batch
    (axis 0). All 7 arguments map over their leading axis; S0 simply has no
    L axis.
    """
    over_heads = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))


def chunkwise_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array]:
    """Chunkwise-parallel Gated Delta Rule-2 forward (training path).

    Uses the exponent-centered core (_chunkwise_single_optim); safe for
    per-chunk cumulative log-decay |G_C| up to ≈ 176 in fp32.

    Args:
      q, k, g, b: [B, H, L, dk]   v, w: [B, H, L, dv]   S0: [B, H, dk, dv]
      chunk_size: intra-chunk length C (L divisible by C).
    Returns:
      (O: [B, H, L, dv], S_final: [B, H, dk, dv])
    """
    def fun(
        Q: jax.Array,
        K: jax.Array,
        V: jax.Array,
        G: jax.Array,
        B: jax.Array,
        W: jax.Array,
        So: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _chunkwise_single_optim(Q, K, V, G, B, W, So, chunk_size=chunk_size)

    return _batchify(fun)(q, k, v, g, b, w, S0)


def recurrent_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token Gated Delta Rule-2 forward (reference / inference path).

    Same I/O contract as chunkwise_gated_delta_rule_2 (without chunk_size):
      q, k, g, b: [B, H, L, dk]   v, w: [B, H, L, dv]   S0: [B, H, dk, dv]
    Returns:
      (O: [B, H, L, dv], S_final: [B, H, dk, dv])
    """
    return _batchify(_recurrent_single)(q, k, v, g, b, w, S0)
