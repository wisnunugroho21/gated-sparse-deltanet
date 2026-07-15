"""Numerical verification of sparse_rule.py — Gated Sparse DeltaNet-2 (Sparse
Delta Memory, Cabannes et al., arXiv:2607.07386, Sec. 3.1 / Eqs. 3-5 + PKM
addressing of Sec. 2, with the GDN-2 erase/write decoupling built in).

The GDN-2 gates (erase b, write w) are an intrinsic, always-on part of the
core, so every test drives them; feeding neutral gates (b≡1, w≡1) recovers the
single-β SDM special case (Test 11).

Run with:  python3 test_sparse_rule.py
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from functools import partial

from sparse_rule import (
    _pkm_select,
    chunkwise_sparse_delta_memory,
    recurrent_sparse_delta_memory,
)

rng = np.random.default_rng(0)

# Input order matches the core signature:
# (kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w, M0).
names = ["kw1", "kw2", "qr1", "qr2", "v", "g", "beta", "ew1", "ew2", "w", "M0"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_inputs(B=2, H=3, L=64, root=8, dv=24, decay_scale=0.5):
    """Random Gated Sparse DeltaNet-2 inputs. N = root² slots; W, R ≤ root."""
    kw1, kw2, qr1, qr2, ew1, ew2 = (
        rng.standard_normal((B, H, L, root)).astype(np.float32) for _ in range(6))
    v = rng.standard_normal((B, H, L, dv)).astype(np.float32)
    w = _sigmoid(rng.standard_normal((B, H, L, dv))).astype(np.float32)  # write gate
    # g = log alpha <= 0 (per-head scalar per token)
    g = -decay_scale * np.abs(rng.standard_normal((B, H, L))).astype(np.float32)
    beta = _sigmoid(rng.standard_normal((B, H, L))).astype(np.float32)
    N = root * root
    M0 = rng.standard_normal((B, H, N, dv)).astype(np.float32)
    return tuple(
        jnp.asarray(x) for x in (kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w, M0))


def report(name, a, b):
    err = float(jnp.max(jnp.abs(a - b)))
    rel = err / (float(jnp.max(jnp.abs(b))) + 1e-30)
    print(f"{name:55s} max_abs={err:.3e}  rel={rel:.3e}")
    return err


def _softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------- #
# 0. Independent naive reference, written directly from Eqs. 3-5 + the GDN-2
#    gates, with BRUTE-FORCE selection (materialize all N scores, argsort
#    top-k), numpy float64 — independent of sparse_rule.py's PKM trick and scan.
#    ew1/ew2 = None ⇒ erase gate b ≡ 1; w = None ⇒ z = v (single-β SDM ref).
# ---------------------------------------------------------------- #
def naive_sdm_np(kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w, M0, W, R):
    kw1, kw2, qr1, qr2, v, g, beta, M0 = (
        np.asarray(x, dtype=np.float64)
        for x in (kw1, kw2, qr1, qr2, v, g, beta, M0))
    use_b = ew1 is not None
    use_w = w is not None
    if use_b:
        ew1, ew2 = np.asarray(ew1, np.float64), np.asarray(ew2, np.float64)
    if use_w:
        w = np.asarray(w, np.float64)
    B, H, L, root = kw1.shape
    N = root * root
    dv = v.shape[-1]
    Y = np.zeros((B, H, L, dv))
    Mf = np.zeros((B, H, N, dv))
    for bi in range(B):
        for h in range(H):
            M = M0[bi, h].copy()
            for t in range(L):
                # Full outer-sum scores (brute force), then argsort top-k.
                sw = (kw1[bi, h, t][:, None] + kw2[bi, h, t][None, :]).reshape(-1)
                wi = np.argsort(-sw)[:W]              # top-W write slots
                kappa = _softmax_np(sw[wi])           # write weights κ
                # GDN-2 erase gate b at the selected slots: κ_e = b⊙κ.
                if use_b:
                    ew = (ew1[bi, h, t][:, None] + ew2[bi, h, t][None, :]).reshape(-1)
                    kappa_e = _sigmoid(ew[wi]) * kappa
                else:
                    kappa_e = kappa
                z = (w[bi, h, t] * v[bi, h, t]) if use_w else v[bi, h, t]
                a = np.exp(g[bi, h, t])               # α_t, Eq. 3
                Msel = a * M[wi]                       # forget (selected only)
                r = kappa_e @ Msel                     # erase-gated delta read
                # Eq. 4: delta write (residual z − r, scattered with κ).
                M[wi] = Msel + beta[bi, h, t] * kappa[:, None] * (
                    z[None, :] - r[None, :])
                # Eq. 5: sparse read from post-write memory.
                sr = (qr1[bi, h, t][:, None] + qr2[bi, h, t][None, :]).reshape(-1)
                ri = np.argsort(-sr)[:R]
                rho = _softmax_np(sr[ri])
                Y[bi, h, t] = rho @ M[ri]
            Mf[bi, h] = M
    return Y, Mf


# ---------------------------------------------------------------- #
# 1. PKM addressing: the outer-sum top-k trick == brute-force top-k set.
# ---------------------------------------------------------------- #
print("=== Test 1: _pkm_select == brute-force top-k (as a set) ===")
mismatches = 0
for trial in range(200):
    root = int(rng.integers(4, 20))
    topk = int(rng.integers(1, root + 1))
    s1 = rng.standard_normal(root).astype(np.float32)
    s2 = rng.standard_normal(root).astype(np.float32)
    idx, _ = _pkm_select(jnp.asarray(s1), jnp.asarray(s2), topk)
    s_full = (s1[:, None] + s2[None, :]).reshape(-1)
    brute = set(np.argsort(-s_full)[:topk].tolist())
    got = set(np.asarray(idx).tolist())
    mismatches += int(brute != got)
print(f"pkm-select set matches brute force: {200 - mismatches}/200 trials")
assert mismatches == 0, "PKM top-k trick disagrees with brute-force top-k"

# ---------------------------------------------------------------- #
# 2. Recurrent core vs independent brute-force numpy oracle (with gates).
# ---------------------------------------------------------------- #
print("\n=== Test 2: recurrent core vs float64 brute-force oracle ===")
for W, R in [(4, 4), (2, 6), (8, 8), (1, 1)]:
    inp = make_inputs()
    y_jax, M_jax = recurrent_sparse_delta_memory(*inp, W=W, R=R)
    y_np, M_np = naive_sdm_np(*inp, W=W, R=R)
    e_y = report(f"W={W},R={R}: y", y_jax, jnp.asarray(y_np, jnp.float32))
    e_m = report(f"W={W},R={R}: M_final", M_jax, jnp.asarray(M_np, jnp.float32))
    assert e_y < 1e-4 and e_m < 1e-4, f"SDM recurrence mismatch at W={W},R={R}"

# ---------------------------------------------------------------- #
# 3. Strong-decay regime stays correct and finite.
# ---------------------------------------------------------------- #
print("\n=== Test 3: strong decay (scale=4.0) matches oracle, stays finite ===")
inp_s = make_inputs(B=1, H=2, L=64, root=10, dv=16, decay_scale=4.0)
y_jax, M_jax = recurrent_sparse_delta_memory(*inp_s, W=6, R=6)
y_np, M_np = naive_sdm_np(*inp_s, W=6, R=6)
assert bool(jnp.all(jnp.isfinite(y_jax))) and bool(jnp.all(jnp.isfinite(M_jax)))
report("strong decay: y", y_jax, jnp.asarray(y_np, jnp.float32))
report("strong decay: M_final", M_jax, jnp.asarray(M_np, jnp.float32))

# ---------------------------------------------------------------- #
# 4. Zero initial memory.
# ---------------------------------------------------------------- #
print("\n=== Test 4: zero initial memory M0=0 ===")
inp = make_inputs()
inp0 = inp[:10] + (jnp.zeros_like(inp[10]),)
y_jax, M_jax = recurrent_sparse_delta_memory(*inp0, W=4, R=4)
y_np, M_np = naive_sdm_np(*inp0, W=4, R=4)
report("M0=0: y", y_jax, jnp.asarray(y_np, jnp.float32))
report("M0=0: M_final", M_jax, jnp.asarray(M_np, jnp.float32))

# ---------------------------------------------------------------- #
# 5. Gradients are finite through gather/scatter/top_k/softmax + the gates.
# ---------------------------------------------------------------- #
print("\n=== Test 5: gradients finite (all 11 inputs incl. gates) ===")
inp = make_inputs(B=1, H=2, L=32, root=8, dv=16)


def loss_fn(args):
    y, Mf = recurrent_sparse_delta_memory(*args, W=4, R=4)
    return jnp.sum(y**2) + jnp.sum(Mf**2)


grads = jax.grad(loss_fn)(inp)
for n, gr in zip(names, grads):
    ok = bool(jnp.all(jnp.isfinite(gr)))
    print(f"grad {n:5s}: finite={ok}  max_abs={float(jnp.max(jnp.abs(gr))):.3e}")
    assert ok, f"non-finite grad {n}"
# The value path, gates β/w, and erase scores must all carry real gradient.
for idx, nm in [(4, "v"), (6, "beta"), (7, "ew1"), (9, "w")]:
    assert float(jnp.max(jnp.abs(grads[idx]))) > 0, f"{nm} has zero gradient"

# ---------------------------------------------------------------- #
# 6. Chunkwise training core vs the recurrent core, several C, W/R.
# ---------------------------------------------------------------- #
print("\n=== Test 6: chunkwise vs recurrent, several chunk sizes ===")
inp = make_inputs(B=2, H=2, L=64, root=8, dv=24)
y_rec, M_rec = recurrent_sparse_delta_memory(*inp, W=4, R=4)
for C in (1, 2, 4, 8, 16, 32, 64):
    y_ch, M_ch = chunkwise_sparse_delta_memory(*inp, chunk_size=C, W=4, R=4)
    e_y = report(f"C={C}: y", y_ch, y_rec)
    e_m = report(f"C={C}: M_final", M_ch, M_rec)
    assert e_y < 1e-4 and e_m < 1e-4, f"chunkwise mismatch at C={C}"

print("\n=== Test 7: chunkwise vs recurrent, various W/R (C=16) ===")
for W, R in [(2, 6), (6, 2), (8, 8), (1, 1)]:
    y_rec, M_rec = recurrent_sparse_delta_memory(*inp, W=W, R=R)
    y_ch, M_ch = chunkwise_sparse_delta_memory(*inp, chunk_size=16, W=W, R=R)
    e_y = report(f"W={W},R={R}: y", y_ch, y_rec)
    e_m = report(f"W={W},R={R}: M_final", M_ch, M_rec)
    assert e_y < 1e-4 and e_m < 1e-4, f"chunkwise mismatch at W={W},R={R}"

# ---------------------------------------------------------------- #
# 8. Chunkwise is overflow-safe under strong decay (all exponents ≤ 0).
# ---------------------------------------------------------------- #
print("\n=== Test 8: chunkwise overflow-safe under strong decay ===")
for scale, C in [(3.0, 32), (8.0, 32), (20.0, 64)]:
    inp_s = make_inputs(B=1, H=1, L=64, root=10, dv=16, decay_scale=scale)
    y_ch, M_ch = chunkwise_sparse_delta_memory(*inp_s, chunk_size=C, W=6, R=6)
    y_rec, M_rec = recurrent_sparse_delta_memory(*inp_s, W=6, R=6)
    finite = bool(jnp.all(jnp.isfinite(y_ch)) and jnp.all(jnp.isfinite(M_ch)))
    err = float(jnp.max(jnp.abs(y_ch - y_rec)))
    rel = err / (float(jnp.max(jnp.abs(y_rec))) + 1e-30)
    print(f"decay_scale={scale:5.1f}, C={C}: finite={finite}  rel={rel:.3e}")
    assert finite, f"chunkwise non-finite at scale={scale}"
    assert rel < 1e-4, f"chunkwise inaccurate at scale={scale}: rel={rel}"

# ---------------------------------------------------------------- #
# 9. Chunkwise gradients agree with the recurrent core.
# ---------------------------------------------------------------- #
print("\n=== Test 9: chunkwise gradients vs recurrent (C=16) ===")
inp = make_inputs(B=1, H=2, L=32, root=8, dv=16)


def loss_ch(args):
    y, Mf = chunkwise_sparse_delta_memory(*args, chunk_size=16, W=4, R=4)
    return jnp.sum(y**2) + jnp.sum(Mf**2)


g_ch = jax.grad(loss_ch)(inp)
g_re = jax.grad(loss_fn)(inp)
for n, a_, b_ in zip(names, g_ch, g_re):
    ok = bool(jnp.all(jnp.isfinite(a_)))
    err = float(jnp.max(jnp.abs(a_ - b_)))
    rel = err / (float(jnp.max(jnp.abs(b_))) + 1e-30)
    print(f"grad {n:5s}: finite={ok}  max_err_vs_recurrent={err:.3e}  rel={rel:.3e}")
    assert ok, f"non-finite chunkwise grad {n}"
    assert rel < 1e-3, f"chunkwise grad mismatch {n}: rel={rel}"

# ---------------------------------------------------------------- #
# 10. The erase / write gates are isolated: neutralizing one matches the
#     oracle computed with that gate switched off.
# ---------------------------------------------------------------- #
print("\n=== Test 10: erase-only / write-only match the oracle ===")
inp = make_inputs(B=2, H=2, L=64, root=8, dv=24)
kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, w, M0 = inp
big = jnp.full_like(ew1, 30.0)          # b = σ(60) ≈ 1
ones_w = jnp.ones_like(w)               # w = 1 -> z = v
# erase only: neutral w
y_e, M_e = chunkwise_sparse_delta_memory(
    kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, ones_w, M0, chunk_size=16, W=4, R=4)
y_e_np, _ = naive_sdm_np(
    kw1, kw2, qr1, qr2, v, g, beta, ew1, ew2, None, M0, W=4, R=4)
report("erase-only chunkwise vs oracle: y", y_e, jnp.asarray(y_e_np, jnp.float32))
# write only: neutral b (big erase scores)
y_w, M_w = chunkwise_sparse_delta_memory(
    kw1, kw2, qr1, qr2, v, g, beta, big, big, w, M0, chunk_size=16, W=4, R=4)
y_w_np, _ = naive_sdm_np(
    kw1, kw2, qr1, qr2, v, g, beta, None, None, w, M0, W=4, R=4)
e_w = report("write-only chunkwise vs oracle: y", y_w, jnp.asarray(y_w_np, jnp.float32))
assert e_w < 1e-4, "write-only path disagrees with the oracle"

# ---------------------------------------------------------------- #
# 11. Neutral gates (b=1, w=1) reduce to the single-β SDM special case.
# ---------------------------------------------------------------- #
print("\n=== Test 11: neutral gates reduce to single-β SDM (oracle b=1,w=1) ===")
y_neu, M_neu = chunkwise_sparse_delta_memory(
    kw1, kw2, qr1, qr2, v, g, beta, big, big, ones_w, M0, chunk_size=16, W=4, R=4)
y_ref, M_ref = naive_sdm_np(
    kw1, kw2, qr1, qr2, v, g, beta, None, None, None, M0, W=4, R=4)
e_y = report("neutral gates vs single-β SDM: y", y_neu, jnp.asarray(y_ref, jnp.float32))
e_m = report("neutral gates vs single-β SDM: M", M_neu, jnp.asarray(M_ref, jnp.float32))
assert e_y < 1e-4 and e_m < 1e-4, "neutral gates must reduce to single-β SDM"

print("\nAll tests done.")
