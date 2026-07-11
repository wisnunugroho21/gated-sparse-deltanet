"""Numerical verification of rule.py against the paper's equations
(Hatamizadeh, Choi, Kautz, "Gated DeltaNet-2", arXiv:2605.22791).

Run with:  python3 test_rule.py
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from functools import partial

from rule import (
    _batchify,
    _chunkwise_single_pairwise,
    chunkwise_gated_delta_rule_2,
    recurrent_gated_delta_rule_2,
)


def pairwise_gated_delta_rule_2(*args, chunk_size=64):
    """Batched wrapper around the pairwise log-space core (same I/O contract
    as chunkwise_gated_delta_rule_2)."""
    f = partial(_chunkwise_single_pairwise, chunk_size=chunk_size)
    return _batchify(f)(*args)

rng = np.random.default_rng(0)

def make_inputs(B=2, H=3, L=128, dk=16, dv=24, decay_scale=0.5):
    q = rng.standard_normal((B, H, L, dk)).astype(np.float32)
    k = rng.standard_normal((B, H, L, dk)).astype(np.float32)
    # L2-normalize q, k as the paper's block design does
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.standard_normal((B, H, L, dv)).astype(np.float32)
    # g = log alpha <= 0
    g = -decay_scale * np.abs(rng.standard_normal((B, H, L, dk))).astype(np.float32)
    b = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L, dk)))).astype(np.float32)
    w = 1.0 / (1.0 + np.exp(-rng.standard_normal((B, H, L, dv)))).astype(np.float32)
    S0 = rng.standard_normal((B, H, dk, dv)).astype(np.float32)
    return tuple(jnp.asarray(x) for x in (q, k, v, g, b, w, S0))


def report(name, a, b):
    err = float(jnp.max(jnp.abs(a - b)))
    rel = err / (float(jnp.max(jnp.abs(b))) + 1e-30)
    print(f"{name:55s} max_abs={err:.3e}  rel={rel:.3e}")
    return err


# ---------------------------------------------------------------- #
# 0. Independent naive reference, written directly from Eq. 9/29,
#    numpy float64 — independent of rule.py internals.
# ---------------------------------------------------------------- #
def naive_np(q, k, v, g, b, w, S0):
    q, k, v, g, b, w, S0 = (np.asarray(x, dtype=np.float64) for x in (q, k, v, g, b, w, S0))
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    O = np.zeros((B, H, L, dv))
    Sf = np.zeros((B, H, dk, dv))
    for bi in range(B):
        for h in range(H):
            S = S0[bi, h].copy()
            for t in range(L):
                alpha = np.exp(g[bi, h, t])          # Eq. 30
                Sbar = alpha[:, None] * S            # Diag(alpha) S
                e = b[bi, h, t] * k[bi, h, t]        # Eq. 8
                z = w[bi, h, t] * v[bi, h, t]        # Eq. 8
                r = Sbar.T @ e                       # Eq. 9
                S = Sbar + np.outer(k[bi, h, t], z - r)  # Eq. 9/15
                O[bi, h, t] = S.T @ q[bi, h, t]      # o_t = S_t^T q_t
            Sf[bi, h] = S
    return O, Sf


print("=== Test 1: recurrent_gated_delta_rule_2 vs independent float64 naive (Eq. 9) ===")
inp = make_inputs()
O_rec, S_rec = recurrent_gated_delta_rule_2(*inp)
O_naive, S_naive = naive_np(*inp)
e1 = report("recurrent vs naive: O", O_rec, jnp.asarray(O_naive, jnp.float32))
e2 = report("recurrent vs naive: S_final", S_rec, jnp.asarray(S_naive, jnp.float32))

print("\n=== Test 2: chunkwise vs recurrent, several chunk sizes ===")
for C in (8, 16, 32, 64, 128):
    O_ch, S_ch = chunkwise_gated_delta_rule_2(*inp, chunk_size=C)
    report(f"C={C}: O", O_ch, O_rec)
    report(f"C={C}: S_final", S_ch, S_rec)

print("\n=== Test 3: zero initial state ===")
inp0 = inp[:6] + (jnp.zeros_like(inp[6]),)
O_ch, S_ch = chunkwise_gated_delta_rule_2(*inp0, chunk_size=16)
O_rec0, S_rec0 = recurrent_gated_delta_rule_2(*inp0)
report("S0=0: O", O_ch, O_rec0)
report("S0=0: S_final", S_ch, S_rec0)

print("\n=== Test 4: tied gates b=beta*1, w=beta*1 must reduce to KDA (App. A.5) ===")
# KDA reference: S_t = (I - beta k k^T) D S_{t-1} + beta k v^T
def kda_naive(q, k, v, g, beta, S0):
    q, k, v, g, beta, S0 = (np.asarray(x, dtype=np.float64) for x in (q, k, v, g, beta, S0))
    B, H, L, dk = q.shape
    dv = v.shape[-1]
    O = np.zeros((B, H, L, dv))
    for bi in range(B):
        for h in range(H):
            S = S0[bi, h].copy()
            for t in range(L):
                D = np.exp(g[bi, h, t])
                Sd = D[:, None] * S
                kk = k[bi, h, t]
                S = Sd - beta[bi, h, t] * np.outer(kk, Sd.T @ kk) + beta[bi, h, t] * np.outer(kk, v[bi, h, t])
                O[bi, h, t] = S.T @ q[bi, h, t]
    return O

q, k, v, g, b, w, S0 = inp
beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((2, 3, 128)))).astype(np.float32)
beta_j = jnp.asarray(beta)
b_tied = jnp.broadcast_to(beta_j[..., None], b.shape)
w_tied = jnp.broadcast_to(beta_j[..., None], w.shape)
O_tied, _ = chunkwise_gated_delta_rule_2(q, k, v, g, b_tied, w_tied, S0, chunk_size=16)
O_kda = kda_naive(q, k, v, g, beta, S0)
report("tied-gate chunkwise vs KDA naive: O", O_tied, jnp.asarray(O_kda, jnp.float32))

print("\n=== Test 5: further tie decay -> Gated DeltaNet reduction ===")
g_scalar = -0.3 * np.abs(rng.standard_normal((2, 3, 128, 1))).astype(np.float32)
g_tied = jnp.broadcast_to(jnp.asarray(g_scalar), g.shape)
O_gdn2, _ = chunkwise_gated_delta_rule_2(q, k, v, g_tied, b_tied, w_tied, S0, chunk_size=16)
O_gdn = kda_naive(q, k, v, np.broadcast_to(g_scalar, g.shape), beta, S0)
report("tied decay+gates vs Gated DeltaNet naive: O", O_gdn2, jnp.asarray(O_gdn, jnp.float32))

print("\n=== Test 6: negative-eigenvalue variant, b in [0,2] ===")
b2 = 2.0 * jnp.asarray(1.0 / (1.0 + np.exp(-rng.standard_normal(b.shape))), jnp.float32)
inp_neg = (q, k, v, g, b2, w, S0)
O_ch, S_ch = chunkwise_gated_delta_rule_2(*inp_neg, chunk_size=16)
O_rec2, S_rec2 = recurrent_gated_delta_rule_2(*inp_neg)
report("b in [0,2]: chunkwise vs recurrent O", O_ch, O_rec2)
report("b in [0,2]: S_final", S_ch, S_rec2)

print("\n=== Test 7: gradients (chunkwise vs recurrent) ===")
def loss_fn(fn):
    def f(args):
        O, Sf = fn(*args)
        return jnp.sum(O**2) + jnp.sum(Sf**2)
    return f

g_ch = jax.grad(loss_fn(lambda *a: chunkwise_gated_delta_rule_2(*a, chunk_size=16)))(inp)
g_re = jax.grad(loss_fn(recurrent_gated_delta_rule_2))(inp)
names = ["dq", "dk", "dv", "dg", "db", "dw", "dS0"]
for n, a_, b_ in zip(names, g_ch, g_re):
    report(f"grad {n}", a_, b_)
    assert bool(jnp.all(jnp.isfinite(a_))), f"non-finite grad {n}"

print("\n=== Test 8: strong decay stress (fp32 range with exponent centering) ===")
print("(safe if within-chunk |G_C| < ~176; previously the limit was ~88)")
for scale, C in [(1.5, 64), (2.5, 64), (3.0, 64), (5.0, 64), (3.0, 32), (5.0, 16)]:
    inp_s = make_inputs(B=1, H=1, L=256, dk=16, dv=16, decay_scale=scale)
    O_ch, S_ch = chunkwise_gated_delta_rule_2(*inp_s, chunk_size=C)
    O_rec_s, S_rec_s = recurrent_gated_delta_rule_2(*inp_s)
    finite = bool(jnp.all(jnp.isfinite(O_ch)))
    err = float(jnp.max(jnp.abs(O_ch - O_rec_s)))
    rel = err / (float(jnp.max(jnp.abs(O_rec_s))) + 1e-30)
    g_np = np.asarray(inp_s[3])[0, 0]
    Gcum = np.concatenate([np.cumsum(g_np[i:i + C], axis=0) for i in range(0, 256, C)])
    print(f"decay_scale={scale}, C={C:3d}: min within-chunk G={Gcum.min():8.1f}  "
          f"finite={finite}  max_err={err:.3e}  rel={rel:.3e}")

print("\n=== Test 9: gradients finite under strong decay (scale=1.5, C=64) ===")
inp_s = make_inputs(B=1, H=2, L=256, dk=16, dv=16, decay_scale=1.5)
g_s = jax.grad(loss_fn(lambda *a: chunkwise_gated_delta_rule_2(*a, chunk_size=64)))(inp_s)
g_r = jax.grad(loss_fn(recurrent_gated_delta_rule_2))(inp_s)
for n, a_, b_ in zip(names, g_s, g_r):
    ok = bool(jnp.all(jnp.isfinite(a_)))
    err = float(jnp.max(jnp.abs(a_ - b_)))
    rel = err / (float(jnp.max(jnp.abs(b_))) + 1e-30)
    print(f"grad {n:4s}: finite={ok}  max_err_vs_recurrent={err:.3e}  rel={rel:.3e}")
    assert ok, f"non-finite grad {n}"

print("\n=== Test 10: pairwise core vs recurrent, several chunk sizes ===")
for C in (8, 16, 32, 64, 128):
    O_pw, S_pw = pairwise_gated_delta_rule_2(*inp, chunk_size=C)
    e_o = report(f"pairwise C={C}: O", O_pw, O_rec)
    e_s = report(f"pairwise C={C}: S_final", S_pw, S_rec)
    assert e_o < 1e-4 and e_s < 1e-4, f"pairwise mismatch at C={C}"

print("\n=== Test 11: pairwise core is overflow-proof (beyond centered's ~176 limit) ===")
for scale, C in [(3.0, 64), (5.0, 64), (10.0, 64)]:
    inp_s = make_inputs(B=1, H=1, L=256, dk=16, dv=16, decay_scale=scale)
    O_pw, S_pw = pairwise_gated_delta_rule_2(*inp_s, chunk_size=C)
    O_ct, _ = chunkwise_gated_delta_rule_2(*inp_s, chunk_size=C)
    O_rec_s, S_rec_s = recurrent_gated_delta_rule_2(*inp_s)
    finite_pw = bool(jnp.all(jnp.isfinite(O_pw)) and jnp.all(jnp.isfinite(S_pw)))
    finite_ct = bool(jnp.all(jnp.isfinite(O_ct)))
    err = float(jnp.max(jnp.abs(O_pw - O_rec_s)))
    rel = err / (float(jnp.max(jnp.abs(O_rec_s))) + 1e-30)
    g_np = np.asarray(inp_s[3])[0, 0]
    Gcum = np.concatenate([np.cumsum(g_np[i:i + C], axis=0) for i in range(0, 256, C)])
    print(f"decay_scale={scale:5.1f}, C={C}: min within-chunk G={Gcum.min():8.1f}  "
          f"pairwise finite={finite_pw} rel={rel:.3e}  (centered finite={finite_ct})")
    assert finite_pw, f"pairwise non-finite at scale={scale}"
    assert rel < 1e-4, f"pairwise inaccurate at scale={scale}: rel={rel}"

print("\n=== Test 12: pairwise gradients under extreme decay (scale=5, C=64) ===")
inp_s = make_inputs(B=1, H=2, L=256, dk=16, dv=16, decay_scale=5.0)
g_pw = jax.grad(loss_fn(lambda *a: pairwise_gated_delta_rule_2(*a, chunk_size=64)))(inp_s)
g_r = jax.grad(loss_fn(recurrent_gated_delta_rule_2))(inp_s)
for n, a_, b_ in zip(names, g_pw, g_r):
    ok = bool(jnp.all(jnp.isfinite(a_)))
    err = float(jnp.max(jnp.abs(a_ - b_)))
    rel = err / (float(jnp.max(jnp.abs(b_))) + 1e-30)
    print(f"grad {n:4s}: finite={ok}  max_err_vs_recurrent={err:.3e}  rel={rel:.3e}")
    assert ok, f"non-finite pairwise grad {n}"
    assert rel < 1e-3, f"pairwise grad mismatch {n}: rel={rel}"

print("\n=== Test 13: pairwise vs centered agree exactly in the safe regime ===")
O_pw, S_pw = pairwise_gated_delta_rule_2(*inp, chunk_size=16)
O_ct, S_ct = chunkwise_gated_delta_rule_2(*inp, chunk_size=16)
e_o = report("pairwise vs centered: O", O_pw, O_ct)
e_s = report("pairwise vs centered: S_final", S_pw, S_ct)
assert e_o < 1e-4 and e_s < 1e-4, "pairwise and centered cores disagree"

print("\nAll tests done.")
