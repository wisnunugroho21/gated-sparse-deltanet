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
def naive_sdm_np(kw1, kw2, qr1, qr2, v, g, beta, M0, W, R):
    kw1, kw2, qr1, qr2, v, g, beta, M0 = (
        np.asarray(x, dtype=np.float64)
        for x in (kw1, kw2, qr1, qr2, v, g, beta, M0)
    )
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
                kappa = _softmax_np(sw[wi])           # Eq. write weights
                a = np.exp(g[bi, h, t])               # α_t, Eq. 3
                Msel = a * M[wi]                       # forget (selected only)
                r = kappa @ Msel                       # aggregate delta read
                # Eq. 4: delta write.
                M[wi] = Msel + beta[bi, h, t] * kappa[:, None] * (
                    v[bi, h, t][None, :] - r[None, :]
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

print("\nAll tests done.")
