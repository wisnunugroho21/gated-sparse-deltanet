"""Numerical verification of sdm_rule.py against the paper's equations
(Cabannes et al., "Sparse Delta Memory", arXiv:2607.07386, Sec. 3.1).

Run with:  python3 test_sdm_rule.py
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from jax import lax

from rule import recurrent_gated_delta_rule_2
from sdm_rule import (
    chunkwise_sdm,
    chunkwise_sdm_2,
    recurrent_sdm,
    recurrent_sdm_2,
)

rng = np.random.default_rng(0)


def rand_val(B, H, L, k):
    """Softmax-normalized positive values, like the layer produces."""
    s = rng.standard_normal((B, H, L, k)).astype(np.float32)
    e = np.exp(s - s.max(-1, keepdims=True))
    return jnp.asarray(e / e.sum(-1, keepdims=True))


def make_inputs(B=2, H=3, L=64, N=64, W=8, R=8, dv=16, decay_scale=0.3,
                write_pool=None, read_pool=None):
    """write_pool / read_pool restrict indices to slots [0, pool) — small
    pools force heavy slot collisions across tokens (deep segmented-decay
    chains and dense index-match interactions in the chunkwise core)."""
    pool = write_pool or N
    iw = jnp.asarray(
        np.stack([rng.choice(pool, size=W, replace=False)
                  for _ in range(B * H * L)]
                 ).reshape(B, H, L, W).astype(np.int32))
    kw = rand_val(B, H, L, W)
    rpool = read_pool or N
    ir = jnp.asarray(
        np.stack([rng.choice(rpool, size=R, replace=False)
                  for _ in range(B * H * L)]
                 ).reshape(B, H, L, R).astype(np.int32))
    qr = rand_val(B, H, L, R)
    v = jnp.asarray(rng.standard_normal((B, H, L, dv)).astype(np.float32))
    g = jnp.asarray(
        -decay_scale * np.abs(rng.standard_normal((B, H, L))).astype(np.float32))
    beta = jnp.asarray(
        (1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L))))).astype(np.float32))
    M0 = jnp.asarray(rng.standard_normal((B, H, N, dv)).astype(np.float32))
    return iw, kw, ir, qr, v, g, beta, M0


def report(name, a, b):
    err = float(jnp.max(jnp.abs(a - b)))
    rel = err / (float(jnp.max(jnp.abs(b))) + 1e-30)
    print(f"{name:55s} max_abs={err:.3e}  rel={rel:.3e}")
    return err


# ---------------------------------------------------------------- #
# 0. Independent naive reference, written directly from Eqs. 3-5
#    (with the Fig. 2 retrieval ṽ = M̃ᵀk), numpy float64 —
#    independent of sdm_rule.py internals.
# ---------------------------------------------------------------- #
def naive_np(iw, kw, ir, qr, v, g, beta, M0):
    iw, ir = np.asarray(iw), np.asarray(ir)
    kw, qr, v, g, beta, M0 = (
        np.asarray(x, dtype=np.float64) for x in (kw, qr, v, g, beta, M0))
    B, H, L, _ = kw.shape
    N, dv = M0.shape[-2:]
    Y = np.zeros((B, H, L, dv))
    Mf = np.zeros((B, H, N, dv))
    for bi in range(B):
        for h in range(H):
            M = M0[bi, h].copy()
            for t in range(L):
                a = np.exp(g[bi, h, t])
                Iw = iw[bi, h, t]
                Msel = a * M[Iw]                         # Eq. 3 (selected only)
                v_ret = kw[bi, h, t] @ Msel              # ṽ (Fig. 2)
                delta = beta[bi, h, t] * (v[bi, h, t] - v_ret)
                M[Iw] = Msel + np.outer(kw[bi, h, t], delta)  # Eq. 4
                Y[bi, h, t] = qr[bi, h, t] @ M[ir[bi, h, t]]  # Eq. 5, post-write
            Mf[bi, h] = M
    return Y, Mf


print("=== Test 1: recurrent_sdm vs independent float64 naive (Eqs. 3-5) ===")
inp = make_inputs()
y_sdm, M_sdm = recurrent_sdm(*inp)
y_naive, M_naive = naive_np(*inp)
e1 = report("recurrent_sdm vs naive: y", y_sdm, jnp.asarray(y_naive, jnp.float32))
e2 = report("recurrent_sdm vs naive: M_final", M_sdm, jnp.asarray(M_naive, jnp.float32))
assert e1 < 1e-4 and e2 < 1e-4

print("\n=== Test 2: dense limit recovers Gated DeltaNet"
      " (paper 'Connection to GDN') ===")
# N = dk, W = R = N (every slot selected every token), dense key values:
# SDM must equal rule.py's GDR-2 with tied gates b = w = β and a scalar
# per-token decay broadcast over key channels:
#   S = Diag(α)S − β k kᵀ Diag(α) S + β k vᵀ   on both sides.
B, H, L, N, dv = 2, 3, 32, 16, 8
all_idx = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32), (B, H, L, N))
# L2-normalized dense keys/queries (as GDN's block design does) — keeps the
# delta-rule state bounded so absolute-error thresholds are meaningful.
kd = rng.standard_normal((B, H, L, N)).astype(np.float32)
qd = rng.standard_normal((B, H, L, N)).astype(np.float32)
kd = jnp.asarray(kd / np.linalg.norm(kd, axis=-1, keepdims=True))
qd = jnp.asarray(qd / np.linalg.norm(qd, axis=-1, keepdims=True))
vd = jnp.asarray(rng.standard_normal((B, H, L, dv)).astype(np.float32))
gd = jnp.asarray(-0.3 * np.abs(rng.standard_normal((B, H, L))).astype(np.float32))
bd = jnp.asarray(
    (1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L))))).astype(np.float32))
S0 = jnp.asarray(rng.standard_normal((B, H, N, dv)).astype(np.float32))

y_dense, M_dense = recurrent_sdm(all_idx, kd, all_idx, qd, vd, gd, bd, S0)
O_gdr2, S_gdr2 = recurrent_gated_delta_rule_2(
    qd, kd, vd,
    jnp.broadcast_to(gd[..., None], (B, H, L, N)),   # scalar decay -> per channel
    jnp.broadcast_to(bd[..., None], (B, H, L, N)),   # b = β
    jnp.broadcast_to(bd[..., None], (B, H, L, dv)),  # w = β
    S0)
e1 = report("dense-limit SDM vs GDR-2 (tied gates): y", y_dense, O_gdr2)
e2 = report("dense-limit SDM vs GDR-2 (tied gates): M_final", M_dense, S_gdr2)
assert e1 < 1e-4 and e2 < 1e-4

print("\n=== Test 3: unselected slots are EXACTLY untouched (no decay leak) ===")
# Writes restricted to slots [0, N/2); the upper half must come out
# bit-identical to M0 (Sec. 3.1: unselected slots remain unchanged —
# including no decay).
inp3 = make_inputs(N=64, write_pool=32)
_, M3 = recurrent_sdm(*inp3)
d = float(jnp.max(jnp.abs(M3[:, :, 32:] - inp3[-1][:, :, 32:])))
print(f"untouched half of the table: max|dM| = {d:.3e}")
assert d == 0.0, "unwritten slots must pass through bit-exactly"

print("\n=== Test 4: state carry across segments equals one full pass ===")
iw, kw, ir, qr, v, g, beta, M0 = inp
cut = 37
y_a, M_a = recurrent_sdm(iw[:, :, :cut], kw[:, :, :cut], ir[:, :, :cut],
                         qr[:, :, :cut], v[:, :, :cut], g[:, :, :cut],
                         beta[:, :, :cut], M0)
y_b, M_b = recurrent_sdm(iw[:, :, cut:], kw[:, :, cut:], ir[:, :, cut:],
                         qr[:, :, cut:], v[:, :, cut:], g[:, :, cut:],
                         beta[:, :, cut:], M_a)
e1 = report("segmented (37+27) vs full: y", jnp.concatenate([y_a, y_b], 2), y_sdm)
e2 = report("segmented (37+27) vs full: M_final", M_b, M_sdm)
assert e1 < 1e-5 and e2 < 1e-5

print("\n=== Test 5: gradients vs a differentiable dense one-hot reference ===")
# Same equations with the sparsity realized as dense one-hot scatters —
# an independent, fully differentiable formulation. Gradients through the
# core's gather/scatter path must match it on every float input.
def dense_ref(iw, ir, kw, qr, v, g, beta, M0):
    Nn = M0.shape[-2]
    oh_w = jax.nn.one_hot(iw, Nn, dtype=jnp.float32)   # [B,H,L,W,N]
    oh_r = jax.nn.one_hot(ir, Nn, dtype=jnp.float32)
    kd = jnp.sum(oh_w * kw[..., None], axis=-2)        # dense keys   [B,H,L,N]
    qd = jnp.sum(oh_r * qr[..., None], axis=-2)
    m = jnp.sum(oh_w, axis=-2)                         # write mask (0/1)
    alpha = jnp.exp(g)
    xs = tuple(jnp.moveaxis(t, 2, 0) for t in (kd, qd, m, alpha, beta, v))

    def step(M, inp):
        kd_t, qd_t, m_t, a_t, b_t, v_t = inp
        dec = jnp.where(m_t > 0.5, a_t[..., None], 1.0)      # lazy decay
        Md = M * dec[..., None]
        v_ret = jnp.einsum("bhn,bhnd->bhd", kd_t, Md)
        delta = b_t[..., None] * (v_t - v_ret)
        M = Md + kd_t[..., None] * delta[..., None, :]
        y_t = jnp.einsum("bhn,bhnd->bhd", qd_t, M)
        return M, y_t

    Mf, ys = lax.scan(step, M0, xs)
    return jnp.moveaxis(ys, 0, 2), Mf


iw, kw, ir, qr, v, g, beta, M0 = make_inputs(B=1, H=2, L=32, N=32, W=6, R=6, dv=8)
y_a, M_a = recurrent_sdm(iw, kw, ir, qr, v, g, beta, M0)
y_b, M_b = dense_ref(iw, ir, kw, qr, v, g, beta, M0)
report("dense one-hot ref forward: y", y_a, y_b)
report("dense one-hot ref forward: M_final", M_a, M_b)


def loss_sparse(floats):
    kw_, qr_, v_, g_, beta_, M0_ = floats
    y, Mf = recurrent_sdm(iw, kw_, ir, qr_, v_, g_, beta_, M0_)
    return jnp.sum(y**2) + jnp.sum(Mf**2)


def loss_dense(floats):
    kw_, qr_, v_, g_, beta_, M0_ = floats
    y, Mf = dense_ref(iw, ir, kw_, qr_, v_, g_, beta_, M0_)
    return jnp.sum(y**2) + jnp.sum(Mf**2)


floats = (kw, qr, v, g, beta, M0)
g_sp = jax.grad(loss_sparse)(floats)
g_de = jax.grad(loss_dense)(floats)
for n, a_, b_ in zip(["dkw", "dqr", "dv", "dg", "dbeta", "dM0"], g_sp, g_de):
    e = report(f"grad {n}", a_, b_)
    assert bool(jnp.all(jnp.isfinite(a_))), f"non-finite grad {n}"
    assert e < 1e-3, f"grad mismatch {n}"

print("\n=== Test 6: chunkwise vs recurrent, several chunk sizes ===")
for C in (1, 8, 16, 32, 64):
    y_ch, M_ch = chunkwise_sdm(*inp, chunk_size=C)
    e1 = report(f"C={C}: y", y_ch, y_sdm)
    e2 = report(f"C={C}: M_final", M_ch, M_sdm)
    assert e1 < 1e-3 and e2 < 1e-3, f"chunkwise mismatch at C={C}"

print("\n=== Test 7: heavy slot collisions (small pools) + zero init state ===")
# write_pool = read_pool = 8 with W = R = 8: EVERY token rewrites and
# rereads the same 8 slots — maximal index-match density, the deepest
# segmented-decay chains, and reads that always hit in-chunk writes.
inp7 = make_inputs(B=1, H=2, L=64, N=64, W=8, R=8,
                   write_pool=8, read_pool=8)
inp7 = inp7[:-1] + (jnp.zeros_like(inp7[-1]),)  # also cover M0 = 0
y_r7, M_r7 = recurrent_sdm(*inp7)
for C in (8, 16, 64):
    y_c7, M_c7 = chunkwise_sdm(*inp7, chunk_size=C)
    e1 = report(f"collisions C={C}: y", y_c7, y_r7)
    e2 = report(f"collisions C={C}: M_final", M_c7, M_r7)
    assert e1 < 1e-3 and e2 < 1e-3, f"collision mismatch at C={C}"

print("\n=== Test 8: strong decay stress — masked-pairwise is overflow-proof ===")
# Hot slots + strong decay: per-slot within-chunk |Λ| far beyond fp32's
# exp range for the factored e^{+λ}·e^{-λ} split (rule.py's ~88/~176
# limits). The masked-before-exp pairwise form must stay finite and match
# the recurrent oracle (the paper observes near-zero forget gates, App. B
# — this regime is real).
for scale, C in [(2.0, 16), (5.0, 32), (10.0, 64)]:
    inp8 = make_inputs(B=1, H=1, L=64, N=32, W=8, R=8,
                       decay_scale=scale, write_pool=8, read_pool=8)
    y_r8, M_r8 = recurrent_sdm(*inp8)
    y_c8, M_c8 = chunkwise_sdm(*inp8, chunk_size=C)
    # deepest per-slot within-chunk log-decay of the last chunk (report only)
    g_np, iw_np = np.asarray(inp8[5])[0, 0], np.asarray(inp8[0])[0, 0]
    lam = np.zeros(32)
    for t in range(64 - C, 64):
        np.add.at(lam, iw_np[t], g_np[t])
    deep = lam.min()
    finite = bool(jnp.all(jnp.isfinite(y_c8)) and jnp.all(jnp.isfinite(M_c8)))
    err = float(jnp.max(jnp.abs(y_c8 - y_r8)))
    rel = err / (float(jnp.max(jnp.abs(y_r8))) + 1e-30)
    print(f"decay_scale={scale:5.1f}, C={C:2d}: min per-slot Λ ≲ {deep:8.1f}  "
          f"finite={finite}  max_err={err:.3e}  rel={rel:.3e}")
    assert finite, f"chunkwise non-finite at scale={scale}"
    assert rel < 1e-4, f"chunkwise inaccurate at scale={scale}: rel={rel}"

print("\n=== Test 9: gradients — chunkwise vs recurrent ===")
iw9, kw9, ir9, qr9, v9, g9, beta9, M09 = make_inputs(
    B=1, H=2, L=32, N=32, W=6, R=6, write_pool=12, read_pool=12)


def loss9(fn):
    def f(floats):
        kw_, qr_, v_, g_, beta_, M0_ = floats
        y, Mf = fn(iw9, kw_, ir9, qr_, v_, g_, beta_, M0_)
        return jnp.sum(y**2) + jnp.sum(Mf**2)
    return f


floats9 = (kw9, qr9, v9, g9, beta9, M09)
g_ch = jax.grad(loss9(lambda *a: chunkwise_sdm(*a, chunk_size=8)))(floats9)
g_re = jax.grad(loss9(recurrent_sdm))(floats9)
for n, a_, b_ in zip(["dkw", "dqr", "dv", "dg", "dbeta", "dM0"], g_ch, g_re):
    e = report(f"grad {n}", a_, b_)
    assert bool(jnp.all(jnp.isfinite(a_))), f"non-finite chunkwise grad {n}"
    assert e < 1e-3, f"chunkwise grad mismatch {n}"

print("\n=== Test 10: SDM-2 (decoupled b, w) — dense limit recovers GDR-2 ===")
# Independent scalar erase b and per-channel write w. Dense limit must
# equal rule.py's Gated Delta Rule-2 with scalar decay/erase broadcast
# over key channels and w passed through UNCHANGED — the exact "-2"
# decoupling, now on the slot axis.  (Reuses Test 2's dense tensors.)
b_s = jnp.asarray(
    (1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L))))).astype(np.float32))
w_c = jnp.asarray(
    (1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L, dv))))).astype(np.float32))
y2d, M2d = recurrent_sdm_2(all_idx, kd, all_idx, qd, vd, gd, b_s, w_c, S0)
O2d, S2d = recurrent_gated_delta_rule_2(
    qd, kd, vd,
    jnp.broadcast_to(gd[..., None], (B, H, L, N)),
    jnp.broadcast_to(b_s[..., None], (B, H, L, N)),  # scalar erase -> per channel
    w_c,                                             # per-channel write, as-is
    S0)
e1 = report("dense-limit SDM-2 vs GDR-2: y", y2d, O2d)
e2 = report("dense-limit SDM-2 vs GDR-2: M_final", M2d, S2d)
assert e1 < 1e-4 and e2 < 1e-4

print("\n=== Test 11: SDM-2 sparse — naive oracle, chunkwise, gradients ===")
def naive2_np(iw, kw, ir, qr, v, g, b, w, M0):
    """Decoupled-gate float64 naive: Δ = w ⊙ v − b·ṽ."""
    iw, ir = np.asarray(iw), np.asarray(ir)
    kw, qr, v, g, b, w, M0 = (
        np.asarray(x, dtype=np.float64) for x in (kw, qr, v, g, b, w, M0))
    Bb, Hh, Ll, _ = kw.shape
    Y = np.zeros((Bb, Hh, Ll, v.shape[-1]))
    Mf = np.zeros_like(M0)
    for bi in range(Bb):
        for h in range(Hh):
            M = M0[bi, h].copy()
            for t in range(Ll):
                Iw = iw[bi, h, t]
                Msel = np.exp(g[bi, h, t]) * M[Iw]
                v_ret = kw[bi, h, t] @ Msel
                delta = w[bi, h, t] * v[bi, h, t] - b[bi, h, t] * v_ret
                M[Iw] = Msel + np.outer(kw[bi, h, t], delta)
                Y[bi, h, t] = qr[bi, h, t] @ M[ir[bi, h, t]]
            Mf[bi, h] = M
    return Y, Mf


iw11, kw11, ir11, qr11, v11, g11, b11, M011 = make_inputs(
    B=1, H=2, L=64, N=64, W=8, R=8, write_pool=16, read_pool=16)
w11 = jnp.asarray(
    (1.0 / (1.0 + np.exp(-rng.standard_normal((1, 2, 64, 16))))).astype(np.float32))
inp11 = (iw11, kw11, ir11, qr11, v11, g11, b11, w11, M011)

y_r11, M_r11 = recurrent_sdm_2(*inp11)
y_n11, M_n11 = naive2_np(*inp11)
e1 = report("recurrent SDM-2 vs float64 naive: y", y_r11,
            jnp.asarray(y_n11, jnp.float32))
e2 = report("recurrent SDM-2 vs float64 naive: M_final", M_r11,
            jnp.asarray(M_n11, jnp.float32))
assert e1 < 1e-4 and e2 < 1e-4

for C in (8, 32, 64):
    y_c11, M_c11 = chunkwise_sdm_2(*inp11, chunk_size=C)
    e1 = report(f"SDM-2 chunkwise C={C}: y", y_c11, y_r11)
    e2 = report(f"SDM-2 chunkwise C={C}: M_final", M_c11, M_r11)
    assert e1 < 1e-3 and e2 < 1e-3, f"SDM-2 chunkwise mismatch at C={C}"


def loss11(fn):
    def f(floats):
        kw_, qr_, v_, g_, b_, w_, M0_ = floats
        y, Mf = fn(iw11, kw_, ir11, qr_, v_, g_, b_, w_, M0_)
        return jnp.sum(y**2) + jnp.sum(Mf**2)
    return f


floats11 = (kw11, qr11, v11, g11, b11, w11, M011)
g_ch = jax.grad(loss11(lambda *a: chunkwise_sdm_2(*a, chunk_size=16)))(floats11)
g_re = jax.grad(loss11(recurrent_sdm_2))(floats11)
for n, a_, b_ in zip(["dkw", "dqr", "dv", "dg", "db", "dw", "dM0"], g_ch, g_re):
    e = report(f"grad {n}", a_, b_)
    assert bool(jnp.all(jnp.isfinite(a_))), f"non-finite SDM-2 grad {n}"
    assert e < 1e-3, f"SDM-2 grad mismatch {n}"

print("\nAll tests done.")
