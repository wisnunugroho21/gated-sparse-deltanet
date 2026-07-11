"""
Gated DeltaNet-2 — chunkwise parallel training core (JAX), ANNOTATED.

Every line of the algorithm is mapped to the equations of
Hatamizadeh, Choi, Kautz, "Gated DeltaNet-2: Decoupling Erase and Write in
Linear Attention" (arXiv:2605.22791). Dual numbering is given as
"main text / Appendix A" where the equation appears in both.

State orientation follows the paper: S in R^{dk x dv}, output o_t = S_t^T q_t.

Per-head recurrence (Eq. 10 / 29):
    S_r = (I - k_r e_r^T) diag(alpha_r) S_{r-1} + k_r z_r^T,
    e_r = b_r ⊙ k_r,   z_r = w_r ⊙ v_r,   alpha_r = exp(g_r).

Chunkwise WY form version (Eqs. 18-25 / 30-44):
    G_r   = cumsum(g)             (inclusive, within chunk)        Eq. 18/30
    gamma = exp(G),  gamma_C = gamma[-1]                           Eq. 18/30
    Kbar  = gamma^{-1} ⊙ K        (decay-normalized keys)          Eq. 19/32/33
    Ebar  = gamma     ⊙ (B ⊙ K)   (decay-absorbed erase factor)    Eq. 20/33
    Z     = W ⊙ V                                                  Eq. 20/33
    T     = tril(Ebar Kbar^T, -1)                                  Eq. 21/34
    A     = (I + T)^{-1}          (unit lower-triangular solve)    Eq. 21/34
    Y, U  = A Ebar, A Z           (WY auxiliaries; share inverse)  Eq. 22/34
    R     = U - Y S0                                               Eq. 35
    O     = Qgamma S0 + Aqk R                                      Eq. 24/44
    S_C   = diag(gamma_C) S0 + Ktail^T R                           Eq. 23/40

Numerical-stability deviation from the paper's literal factors (exponent
centering): gamma^{-1} = exp(-G) is unbounded, so materializing Kbar as
written overflows fp32 once the within-chunk cumulative log-decay G drops
below ~-88. Every product consumed downstream only involves the bounded
ratios exp(G_r - G_s) (s <= r) and exp(G_C - G_r), so we shift all
within-chunk exponents by c = G_C/2 per channel: the shift cancels exactly
in T, Aqk, and Ktail, and is re-attached where an absolute gamma meets the
state (Y S0, Qgamma S0) by pre-scaling S0 with diag(exp(c)), exp(c) <= 1.
This doubles the safe per-chunk decay range to |G_C| ~ 176; beyond that,
reduce chunk_size.

The gate-aware backward (Eqs. 64-82, Appendix B) is intentionally NOT written:
jax.grad differentiates straight through solve_triangular and the elementwise
gate products and reconstructs exactly those vector-Jacobian products. The
hand-derived backward is only needed for a fused Triton/Pallas kernel.
"""

import jax
import jax.numpy as jnp
from jax import lax

# math runs in fp32 (paper App. D.1/D.3/D.4)
D_TYPE = jnp.float32

# --------------------------------------------------------------------------- #
#  Single (batch, head) sequence — the actual algorithm.
#  Everything else is vmap over (B, H) on top of this.
# --------------------------------------------------------------------------- #
def _chunkwise_single(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    """q,k,g,b: [L, dk]  v,w: [L, dv]  S0: [dk, dv]  ->  (O: [L, dv], S_final: [dk, dv]).

    Structured for GPU throughput: everything that is chunk-LOCAL (Eqs. 18-22 and
    the A_qk/K_tail factors of Eqs. 23-25 — cumsums, gate products, the triangular
    solve, the WY auxiliaries) depends only on the chunk's own inputs, so it is
    computed for ALL N chunks at once as batched [N, C, ·] ops. Only the parts
    that read the running state S remain in the sequential cross-chunk scan, and
    that scan body is just three small matmuls per chunk.
    """
    L, dk = k.shape
    dv = v.shape[-1]
    C = chunk_size
    N = L // C

    def to_chunks(x):
        return x.reshape(N, C, x.shape[-1]).astype(D_TYPE)

    q, k, v = to_chunks(q), to_chunks(k), to_chunks(v)
    g, b, w = to_chunks(g), to_chunks(b), to_chunks(w)

    eye = jnp.eye(C, dtype=D_TYPE)
    S0 = S0.astype(D_TYPE)

    # ---- Chunk-local precompute: all N chunks in parallel ([N, C, d*]) --------
    # The per-chunk cumsum (axis=1) resets at every chunk boundary, which realizes
    # both gamma_0 = 1 (Eq. 18/30) and the normalized init Ŝ_0 = S_[n] (Eq. 31 / A.1).

    # --- Cumulative decay ---
    # Eq. 18/30:  G_r = Σ_{i≤r} g_i (inclusive, within chunk)
    G = jnp.cumsum(g, axis=1)

    # G_C, total chunk log-decay (last row of each chunk); Eq. 40/41
    G_C = G[:, -1]  # [N, dk]
    gamma_C = jnp.exp(G_C)

    # Exponent centering (see module docstring): shift within-chunk exponents
    # by c = G_C/2 per channel; Gc spans ±|G_C|/2 instead of [G_C, 0].
    Gc = G - 0.5 * G_C[:, None, :]

    # exp(c): per-chunk state pre-scale (≤ 1) that re-attaches the shift
    delta = jnp.exp(0.5 * G_C)  # [N, dk]

    # --- Decay normalization (removes Diag(α) from the recurrence) ---
    # Eq. 19/32/33 centered:  K̄ = exp(c-G) ⊙ K  (paper: γ^{-1} ⊙ K)
    Kbar = k * jnp.exp(-Gc)

    # Eq. 20/33 centered:  Ē = exp(G-c) ⊙ (B ⊙ K);  b*k = e_r (Eq. 8)
    Ebar = jnp.exp(Gc) * (b * k)

    # Eq. 8, 20/33:  Z = W ⊙ V  (z_r = w_r ⊙ v_r)
    Z = w * v

    # Eq. 24/43 centered:  Q_γ, row exp(G_r-c) ⊙ q_r; the missing exp(c)
    # is restored by pairing with the pre-scaled state diag(exp(c)) S0
    Qg = jnp.exp(Gc) * q

    # --- WY triangular solve (the parallelization) ---
    # Eq. 21/34, entry Eq. 87:  T = tril(Ē K̄ᵀ, -1), T_rs = ē_rᵀ k̄_s (s<r)
    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)  # [N, C, C]

    # Eq. 21/34:  A = (I + T)^{-1}  (unit lower-tri -> forward substitution,
    # batched over the N chunks)
    A = jax.scipy.linalg.solve_triangular(
        eye + T, jnp.broadcast_to(eye, T.shape), lower=True, unit_diagonal=True
    )

    # --- WY auxiliaries ---
    # Eq. 22/34:  Y = A Ē  (erase-side auxiliary)
    Y = A @ Ebar

    # Eq. 22/34:  U = A Z  (write-side aux; SAME inverse A, two RHS — A.4)
    U = A @ Z

    # Eq. 25/43:  (A_qk)_rs = 1_{r≥s} q_rᵀ Diag(γ_r/γ_s) k_s  (tril incl. diag: s≤r)
    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))  # [N, C, C]

    # Eq. 23/41:  (K_tail)_r = (γ_C / γ_r) ⊙ k_r, formed in log-space as
    # exp(G_C - G_r) — the ratio of two underflowed γ's would give 0/0
    Ktail = k * jnp.exp(G_C[:, None, :] - G)

    # ---- Cross-chunk recurrence: the ONLY sequential part ---------------------
    # (Sec. 2.1 / Eq. 3 structure.) S is the raw chunk-entry state S_[n] (== S_0,
    # NOT decay-normalized); each step is three [C, d] x [d, d] matmuls.
    def chunk_step(
        S: jax.Array,
        inp: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n, delta_n = inp

        # diag(exp(c)) S_0: re-attaches the centering shift so that
        # Y (δ⊙S) == (A Ē) S_0 and Qg (δ⊙S) == Q_γ S_0 exactly
        S_c = delta_n[:, None] * S

        # Eq. 35:     R = U − Y S_0  (stacked residual rows ρ_r, Eq. 37)
        R = U_n - Y_n @ S_c

        # Eq. 24/44:  O = Q_γ S_0 + A_qk (U − Y S_0)
        o = Qg_n @ S_c + Aqk_n @ R

        # Eq. 23/40:  S_[n+1] = Diag(γ_C) S_0 + K_tailᵀ R
        # Diag(γ_C) S_0: broadcast over key-channel rows (decay lives on key axis)
        S_new = gamma_C_n[:, None] * S + Ktail_n.T @ R

        return S_new, o

    S_final, o = lax.scan(chunk_step, S0, (Y, U, Aqk, Qg, Ktail, gamma_C, delta))
    return o.reshape(L, dv), S_final


def _recurrent_single(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token reference (Eq. 9 / 29). Same signature as the chunkwise core.

    Three-line factored form of Eq. 9, algebraically equal to the
    (I - k_t e_t^T) Diag(α_t) form of Eq. 10/29. O(L·dk·dv), no triangular solve —
    a trustworthy ground truth for verifying the chunkwise path.
    """
    q = q.astype(D_TYPE)
    k = k.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    b = b.astype(D_TYPE)
    w = w.astype(D_TYPE)
    S0 = S0.astype(D_TYPE)

    alpha = jnp.exp(g)  # Eq. 12/30:  α_r = exp(g_r)
    e = b * k  # Eq. 8:      e_r = b_r ⊙ k_r
    z = w * v  # Eq. 8:      z_r = w_r ⊙ v_r

    def step(S, inp):
        qt, kt, at, et, zt = inp

        # Reshape inputs to column vectors for matrix multiplication
        qt = qt[:, None]
        kt = kt[:, None]
        at = at[:, None]
        et = et[:, None]
        zt = zt[:, None]

        # Eq. 9:  S̄_t = D_t S_{t-1} = Diag(α_t) S_{t-1} (scale key-channel rows)
        S_bar = at * S

        # Eq. 9:  r_t = S̄_tᵀ e_t  (read old content along erase direction)
        r_t = S_bar.T @ et

        # Eq. 9/15:  S_t = S̄_t + k_t (z_t − r_t)ᵀ  (rank-one delta write)
        S_new = S_bar + kt * (zt - r_t).T

        # Eq. 1:  o_t = S_tᵀ q_t
        o_t = S_new.T @ qt

        return S_new, o_t

    S_final, o = lax.scan(step, S0, (q, k, alpha, e, z))
    return o.squeeze(-1), S_final

# --------------------------------------------------------------------------- #
#  Batched public entry points: inputs are [B, H, L, d].  (No equations here —
#  pure plumbing: vmap the per-head algorithm over heads, then over batch.)
# --------------------------------------------------------------------------- #
def _batchify(fn):
    # vmap over heads (axis 1) then batch (axis 0); S0 has no L axis.
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
    """Parallel chunkwise forward.

    q, k, g, b : [B, H, L, dk]      v, w : [B, H, L, dv]      S0 : [B, H, dk, dv]
    returns (O : [B, H, L, dv], S_final : [B, H, dk, dv]).
    """
    f = lambda *a: _chunkwise_single(*a, chunk_size=chunk_size)
    return _batchify(f)(q, k, v, g, b, w, S0)


def recurrent_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Token-by-token reference forward, same I/O as the chunkwise version."""
    return _batchify(_recurrent_single)(q, k, v, g, b, w, S0)
