"""Numerical verification of sparse_rule.py against the Sparse Delta Memory
equations (Cabannes et al., "Sparse Delta Memory", arXiv:2607.07386),
Sec. 3.1 (Eqs. 3-5) + the PKM addressing of Sec. 2.

Phase 1 scope: the recurrent reference core + Product-Key addressing.

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


def make_inputs(B=2, H=3, L=64, root=8, dv=24, decay_scale=0.5):
    """Random SDM inputs. N = root² slots; W, R chosen by the caller (≤ root)."""
    kw1 = rng.standard_normal((B, H, L, root)).astype(np.float32)
    kw2 = rng.standard_normal((B, H, L, root)).astype(np.float32)
    qr1 = rng.standard_normal((B, H, L, root)).astype(np.float32)
    qr2 = rng.standard_normal((B, H, L, root)).astype(np.float32)
    v = rng.standard_normal((B, H, L, dv)).astype(np.float32)
    # g = log alpha <= 0 (per-head scalar per token)
    g = -decay_scale * np.abs(rng.standard_normal((B, H, L))).astype(np.float32)
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L)))).astype(np.float32)
    N = root * root
    M0 = rng.standard_normal((B, H, N, dv)).astype(np.float32)
    return tuple(
        jnp.asarray(x) for x in (kw1, kw2, qr1, qr2, v, g, beta, M0)
    )


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
# 0. Independent naive reference, written directly from Eqs. 3-5 with
#    BRUTE-FORCE selection (materialize all N scores, argsort top-k),
#    numpy float64 — independent of sparse_rule.py's PKM trick and scan.
# ---------------------------------------------------------------- #
def naive_sdm_np(kw1, kw2, qr1, qr2, v, g, beta, M0, W, R,
                 ew1=None, ew2=None, w=None):
    """Brute-force SDM oracle. Passing (ew1, ew2) adds the GDN-2 per-slot erase
    gate b (κ_e = b⊙κ on the recall); passing w adds the value-side write gate
    (z = w⊙v). All None ⇒ faithful SDM."""
    kw1, kw2, qr1, qr2, v, g, beta, M0 = (
        np.asarray(x, dtype=np.float64)
        for x in (kw1, kw2, qr1, qr2, v, g, beta, M0)
    )
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
                    b = 1.0 / (1.0 + np.exp(-ew[wi]))
                    kappa_e = b * kappa
                else:
                    kappa_e = kappa
                z = (w[bi, h, t] * v[bi, h, t]) if use_w else v[bi, h, t]
                a = np.exp(g[bi, h, t])               # α_t, Eq. 3
                Msel = a * M[wi]                       # forget (selected only)
                r = kappa_e @ Msel                     # erase-gated delta read
                # Eq. 4: delta write (residual z − r, scattered with κ).
                M[wi] = Msel + beta[bi, h, t] * kappa[:, None] * (
                    z[None, :] - r[None, :]
                )
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

# Selected scores must equal the true outer-sum scores at those slots.
root, topk = 12, 5
s1 = rng.standard_normal(root).astype(np.float32)
s2 = rng.standard_normal(root).astype(np.float32)
idx, sc = _pkm_select(jnp.asarray(s1), jnp.asarray(s2), topk)
s_full = (s1[:, None] + s2[None, :]).reshape(-1)
report("pkm-select scores vs s_full[idx]", sc, jnp.asarray(s_full[np.asarray(idx)]))

# ---------------------------------------------------------------- #
# 2. Recurrent core vs independent brute-force numpy oracle.
# ---------------------------------------------------------------- #
print("\n=== Test 2: recurrent_sparse_delta_memory vs float64 brute-force oracle ===")
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
inp0 = inp[:7] + (jnp.zeros_like(inp[7]),)
y_jax, M_jax = recurrent_sparse_delta_memory(*inp0, W=4, R=4)
y_np, M_np = naive_sdm_np(*inp0, W=4, R=4)
report("M0=0: y", y_jax, jnp.asarray(y_np, jnp.float32))
report("M0=0: M_final", M_jax, jnp.asarray(M_np, jnp.float32))

# ---------------------------------------------------------------- #
# 5. Gradients are finite (jax.grad through gather/scatter/top_k/softmax).
# ---------------------------------------------------------------- #
print("\n=== Test 5: gradients finite ===")
inp = make_inputs(B=1, H=2, L=32, root=8, dv=16)


def loss_fn(args):
    y, Mf = recurrent_sparse_delta_memory(*args, W=4, R=4)
    return jnp.sum(y**2) + jnp.sum(Mf**2)


grads = jax.grad(loss_fn)(inp)
names = ["kw1", "kw2", "qr1", "qr2", "v", "g", "beta", "M0"]
for n, gr in zip(names, grads):
    ok = bool(jnp.all(jnp.isfinite(gr)))
    print(f"grad {n:5s}: finite={ok}  max_abs={float(jnp.max(jnp.abs(gr))):.3e}")
    assert ok, f"non-finite grad {n}"

# The read queries only affect selection (top-k -> hard, piecewise-constant)
# and the softmax read weights; the write path likewise. A finite, non-trivial
# gradient w.r.t. the value path and gates is the meaningful signal for training.
assert float(jnp.max(jnp.abs(grads[4]))) > 0, "value path has zero gradient"
assert float(jnp.max(jnp.abs(grads[6]))) > 0, "beta gate has zero gradient"

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
# 10. GDN-2 decoupling: erase gate b and/or write gate w.
# ---------------------------------------------------------------- #
print("\n=== Test 10: gated (GDN-2 decoupled) cores vs brute-force oracle ===")


def make_gates(B, H, L, root, dv, seed=1):
    r = np.random.default_rng(seed)
    ew1 = r.standard_normal((B, H, L, root)).astype(np.float32)
    ew2 = r.standard_normal((B, H, L, root)).astype(np.float32)
    w = 1.0 / (1.0 + np.exp(-r.standard_normal((B, H, L, dv)))).astype(np.float32)
    return jnp.asarray(ew1), jnp.asarray(ew2), jnp.asarray(w)


B_, H_, L_, root_, dv_ = 2, 2, 64, 8, 24
inp = make_inputs(B=B_, H=H_, L=L_, root=root_, dv=dv_)
ew1, ew2, w = make_gates(B_, H_, L_, root_, dv_)
for tag, kw_gates, np_gates in [
    ("erase only", dict(ew1=ew1, ew2=ew2), dict(ew1=ew1, ew2=ew2)),
    ("write only", dict(w=w), dict(w=w)),
    ("erase+write", dict(ew1=ew1, ew2=ew2, w=w), dict(ew1=ew1, ew2=ew2, w=w)),
]:
    y_rec, M_rec = recurrent_sparse_delta_memory(*inp, W=4, R=4, **kw_gates)
    y_ch, M_ch = chunkwise_sparse_delta_memory(*inp, chunk_size=16, W=4, R=4, **kw_gates)
    y_np, M_np = naive_sdm_np(*inp, W=4, R=4, **np_gates)
    e_rn = report(f"[{tag}] recurrent vs oracle: y", y_rec, jnp.asarray(y_np, jnp.float32))
    report(f"[{tag}] recurrent vs oracle: M", M_rec, jnp.asarray(M_np, jnp.float32))
    e_cr = report(f"[{tag}] chunkwise vs recurrent: y", y_ch, y_rec)
    report(f"[{tag}] chunkwise vs recurrent: M", M_ch, M_rec)
    assert e_rn < 1e-4 and e_cr < 1e-4, f"gated mismatch [{tag}]"

print("\n=== Test 11: gates reduce to faithful SDM at b=1 (ew=+inf), w=1 ===")
big = jnp.full_like(ew1, 30.0)          # sigmoid(30+30) ≈ 1  -> b ≈ 1
ones_w = jnp.ones_like(w)               # w = 1  -> z = v
y_gate, M_gate = chunkwise_sparse_delta_memory(
    *inp, chunk_size=16, W=4, R=4, ew1=big, ew2=big, w=ones_w)
y_faith, M_faith = chunkwise_sparse_delta_memory(*inp, chunk_size=16, W=4, R=4)
e_y = report("neutral gates vs faithful: y", y_gate, y_faith)
e_m = report("neutral gates vs faithful: M", M_gate, M_faith)
assert e_y < 1e-4 and e_m < 1e-4, "neutral gates must reduce to faithful SDM"

print("\n=== Test 12: gated gradients finite (erase+write, C=16) ===")
inp_g = make_inputs(B=1, H=2, L=32, root=8, dv=16)
ew1g, ew2g, wg = make_gates(1, 2, 32, 8, 16, seed=3)


def loss_gated(args):
    y, Mf = chunkwise_sparse_delta_memory(
        *args, chunk_size=16, W=4, R=4, ew1=ew1g, ew2=ew2g, w=wg)
    return jnp.sum(y**2) + jnp.sum(Mf**2)


grads = jax.grad(loss_gated)(inp_g)
for n, gr in zip(names, grads):
    ok = bool(jnp.all(jnp.isfinite(gr)))
    print(f"grad {n:5s}: finite={ok}  max_abs={float(jnp.max(jnp.abs(gr))):.3e}")
    assert ok, f"non-finite gated grad {n}"
# Gradient w.r.t. the erase scores and write gate themselves.
gew1, gw = jax.grad(
    lambda e, ww: jnp.sum(chunkwise_sparse_delta_memory(
        *inp_g, chunk_size=16, W=4, R=4, ew1=e, ew2=ew2g, w=ww)[0] ** 2),
    argnums=(0, 1))(ew1g, wg)
print(f"grad ew1: max_abs={float(jnp.max(jnp.abs(gew1))):.3e}  "
      f"grad w: max_abs={float(jnp.max(jnp.abs(gw))):.3e}")
assert float(jnp.max(jnp.abs(gew1))) > 0 and float(jnp.max(jnp.abs(gw))) > 0

print("\nAll tests done.")
