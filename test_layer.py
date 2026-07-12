"""Numerical verification of layer.py — the Gated DeltaNet-2 token mixer
(Hatamizadeh, Choi, Kautz, arXiv:2605.22791, Sec. 3.5 / App. C-D) — and of
its streaming contract (step / init_cache).

Run with:  python3 test_layer.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

jax.config.update("jax_enable_x64", False)

from layer import GatedDeltaNet2

D, H, dk, dv, Cs = 64, 4, 16, 16, 64
B, L = 2, 128

layer = GatedDeltaNet2(d_model=D, num_heads=H, head_k_dim=dk, head_v_dim=dv,
                       chunk_size=Cs, rngs=nnx.Rngs(0))
x = jax.random.normal(jax.random.PRNGKey(1), (B, L, D), jnp.float32)


print("=== Test 1: training forward — shape and finiteness ===")
y = layer(x)
print(f"__call__: {y.shape}  finite={bool(jnp.all(jnp.isfinite(y)))}")
assert y.shape == (B, L, D)
assert bool(jnp.all(jnp.isfinite(y)))

print("\n=== Test 2: step() with a fresh cache equals __call__ ===")
y2, cache_full = layer.step(x, layer.init_cache(B))
d = float(jnp.max(jnp.abs(y2 - y)))
print(f"step(full) vs __call__: max|dy| = {d:.3e}")
assert d < 1e-5

print("\n=== Test 3: segmented streaming (ragged splits) equals full pass ===")
# 64 (chunk-aligned) + 37 + 27: exercises the chunkwise prefix, the recurrent
# tail, AND the conv-cache handoff at non-aligned boundaries.
segs = [x[:, :64], x[:, 64:101], x[:, 101:]]
cache = layer.init_cache(B)
outs = []
for s in segs:
    o, cache = layer.step(s, cache)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y)))
dS = float(jnp.max(jnp.abs(cache.recurrent_state - cache_full.recurrent_state)))
print(f"segmented (64+37+27) vs full: max|dy| = {d:.3e}  max|dS| = {dS:.3e}")
assert d < 1e-4 and dS < 1e-4

print("\n=== Test 4: token-by-token decode equals full pass ===")
cache = layer.init_cache(B)
outs = []
for t in range(24):
    o, cache = layer.step(x[:, t:t + 1], cache)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y[:, :24])))
print(f"decode (L=1 steps) vs full: max|dy| = {d:.3e}")
assert d < 1e-4

print("\n=== Test 5: grouped value heads (Hv = 2H, Sec. 3.5 / App. C.1) ===")
gqa = GatedDeltaNet2(d_model=D, num_heads=H, head_k_dim=dk, head_v_dim=8,
                     num_v_heads=2 * H, chunk_size=Cs, rngs=nnx.Rngs(2))
yg = gqa(x)
yg2, cg = gqa.step(x, gqa.init_cache(B))
d = float(jnp.max(jnp.abs(yg2 - yg)))
print(f"GQA out {yg.shape}  finite={bool(jnp.all(jnp.isfinite(yg)))}  "
      f"stream vs full max|dy| = {d:.3e}  S {cg.recurrent_state.shape}")
assert yg.shape == (B, L, D) and d < 1e-4
assert cg.recurrent_state.shape == (B, 2 * H, dk, 8)

print("\n=== Test 6: negative-eigenvalue variant (erase gate in [0,2]) ===")
neg = GatedDeltaNet2(d_model=D, num_heads=H, head_k_dim=dk, head_v_dim=dv,
                     chunk_size=Cs, expanded_erase=True, rngs=nnx.Rngs(3))
yn = neg(x)
print(f"expanded_erase: finite={bool(jnp.all(jnp.isfinite(yn)))}")
assert bool(jnp.all(jnp.isfinite(yn)))

print("\n=== Test 7: decay parameterization matches App. C.1 / the GDN init ===")
a = np.asarray(layer.A_log[...])
dt_bias = np.asarray(layer.dt_bias[...])
dt = np.log1p(np.exp(dt_bias))  # softplus(δ) = decay magnitude at Proj_f x ≈ 0
print(f"A_log shape {a.shape} (per key head);  exp(a) in "
      f"[{np.exp(a).min():.2f}, {np.exp(a).max():.2f}] (paper family: [1, 16])")
print(f"dt_bias shape {dt_bias.shape} (per key channel);  softplus(δ) in "
      f"[{dt.min():.2e}, {dt.max():.2e}] (paper family: [1e-3, 1e-1])")
assert a.shape == (H,), "a must be stored per key head (App. C.1)"
assert dt_bias.shape == (H * dk,), "δ must be stored per key channel (App. C.1)"
assert 1.0 <= np.exp(a).min() and np.exp(a).max() <= 16.0
assert 0.9e-3 <= dt.min() and dt.max() <= 1.1e-1

print("\n=== Test 8: gradients reach every parameter ===")
def loss(model, x):
    return jnp.sum(model(x) ** 2)

grads = nnx.grad(loss)(layer, x)
flat, _ = jax.tree_util.tree_flatten(grads)
bad = [g for g in flat if not bool(jnp.all(jnp.isfinite(g)))]
zero = [g for g in flat if float(jnp.max(jnp.abs(g))) == 0.0]
print(f"{len(flat)} parameter tensors — non-finite: {len(bad)}, all-zero: {len(zero)}")
assert not bad and not zero

print("\n=== Test 9: __call__ validates L % chunk_size ===")
try:
    layer(x[:, :100])
    raise AssertionError("expected ValueError for L=100, C=64")
except ValueError as e:
    print(f"__call__(L=100) raises: {e}")

print("\nAll tests done.")
