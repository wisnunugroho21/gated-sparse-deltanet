"""Numerical verification of sparse_layer.py — the Gated Sparse DeltaNet-2
token mixer (Sparse Delta Memory, Cabannes et al., arXiv:2607.07386, Sec. 3.1,
with the intrinsic GDN-2 erase/write decoupling) — and of its streaming
contract (step / init_cache).

Run with:  python3 test_sparse_layer.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

jax.config.update("jax_enable_x64", False)

from sparse_layer import SparseDeltaMemory

# root >= W,R. N = root² = 64 slots per head; small so the dense chunkwise
# interaction tensors stay tiny.
D, H, dv, root, W, R, Cs = 64, 2, 16, 8, 4, 4, 16
B, L = 2, 64

layer = SparseDeltaMemory(
    d_model=D, num_heads=H, head_v_dim=dv, n_slots_root=root,
    num_write=W, num_read=R, chunk_size=Cs, rngs=nnx.Rngs(0))
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
assert cache_full.recurrent_state.shape == (B, H, root * root, dv)

print("\n=== Test 3: segmented streaming (ragged splits) equals full pass ===")
# 32 (chunk-aligned) + 19 + 13: exercises the chunkwise prefix and the
# recurrent tail across non-aligned boundaries. With no short conv the carry
# is EXACT, so this must match to fp32 round-off.
segs = [x[:, :32], x[:, 32:51], x[:, 51:]]
cache = layer.init_cache(B)
outs = []
for s in segs:
    o, cache = layer.step(s, cache)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y)))
dM = float(jnp.max(jnp.abs(cache.recurrent_state - cache_full.recurrent_state)))
print(f"segmented (32+19+13) vs full: max|dy| = {d:.3e}  max|dM| = {dM:.3e}")
assert d < 1e-4 and dM < 1e-4

print("\n=== Test 4: token-by-token decode equals full pass ===")
cache = layer.init_cache(B)
outs = []
for t in range(20):
    o, cache = layer.step(x[:, t:t + 1], cache)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y[:, :20])))
print(f"decode (L=1 steps) vs full: max|dy| = {d:.3e}")
assert d < 1e-4

print("\n=== Test 5: return_state carry is EXACT (no conv context to lose) ===")
# Full pass == first-half pass, then second half warm-started from its memory.
y5, M5 = layer(x, return_state=True)
_, M_half = layer(x[:, :32], return_state=True)
y_cont = layer(x[:, 32:], initial_state=M_half)
y_join = jnp.concatenate([layer(x[:, :32]), y_cont], axis=1)
d_join = float(jnp.max(jnp.abs(y_join - y)))
d_state = float(jnp.max(jnp.abs(M5 - cache_full.recurrent_state)))
print(f"warm-started join vs full: max|dy| = {d_join:.3e}   "
      f"return_state vs step cache: max|dM| = {d_state:.3e}")
assert M5.shape == (B, H, root * root, dv)
assert d_join < 1e-4 and d_state < 1e-5

print("\n=== Test 6: learnable M0 (parametric memory) participates & matters ===")
# The learned-M0 default and the null-initialized baseline differ, and M0 is a
# real parameter that receives gradient.
null_layer = SparseDeltaMemory(
    d_model=D, num_heads=H, head_v_dim=dv, n_slots_root=root, num_write=W,
    num_read=R, chunk_size=Cs, learn_initial_state=False, rngs=nnx.Rngs(0))
assert not hasattr(null_layer, "M0")
# Seed the learned M0 with something non-zero so the two configs actually differ.
layer.M0[...] = 0.1 * jax.random.normal(jax.random.PRNGKey(7), layer.M0[...].shape)
y_learn = layer(x)
y_null = null_layer(x)
d_m0 = float(jnp.max(jnp.abs(y_learn - y_null)))
print(f"learned-M0 vs null-M0 output differ by max|dy| = {d_m0:.3e}")
assert d_m0 > 1e-4, "a non-zero learned M0 must change the output"

print("\n=== Test 7: gradients reach every parameter (incl. M0) ===")
def loss(model, x):
    return jnp.sum(model(x) ** 2)

grads = nnx.grad(loss)(layer, x)
flat, _ = jax.tree_util.tree_flatten(grads)
bad = [gg for gg in flat if not bool(jnp.all(jnp.isfinite(gg)))]
zero = [gg for gg in flat if float(jnp.max(jnp.abs(gg))) == 0.0]
print(f"{len(flat)} parameter tensors — non-finite: {len(bad)}, all-zero: {len(zero)}")
assert not bad and not zero
# M0 specifically must get a non-trivial gradient.
gM0 = nnx.grad(loss)(layer, x)["M0"][...]
print(f"grad M0: max_abs = {float(jnp.max(jnp.abs(gM0))):.3e}")
assert float(jnp.max(jnp.abs(gM0))) > 0

print("\n=== Test 8: __call__ validates L % chunk_size ===")
try:
    layer(x[:, :50])
    raise AssertionError("expected ValueError for L=50, C=16")
except ValueError as e:
    print(f"__call__(L=50) raises: {e}")

print("\n=== Test 9: decay init matches the GDN/FLA family (per head) ===")
a = np.asarray(layer.A_log[...])
dt_bias = np.asarray(layer.dt_bias[...])
dt = np.log1p(np.exp(dt_bias))
print(f"A_log {a.shape}: exp(a) in [{np.exp(a).min():.2f}, {np.exp(a).max():.2f}]"
      f" (family [1,16]);  softplus(dt_bias) in [{dt.min():.2e}, {dt.max():.2e}]"
      f" (family [1e-3,1e-1])")
assert a.shape == (H,) and dt_bias.shape == (H,)
assert 1.0 <= np.exp(a).min() and np.exp(a).max() <= 16.0
assert 0.9e-3 <= dt.min() and dt.max() <= 1.1e-1

print("\n=== Test 10: GDN-2 gates are intrinsic (always on) & carry gradient ===")
# The layer always builds the erase/write projections — GDN-2 decoupling is
# not optional. Their gradients are already covered by Test 7's all-param
# check; here we confirm they exist and receive a non-trivial gradient.
assert hasattr(layer, "e_proj") and hasattr(layer, "w_proj")
gd = nnx.grad(loss)(layer, x)
g_e = gd["e_proj"]["kernel"][...]
g_w = gd["w_proj"]["kernel"][...]
print(f"grad e_proj: max_abs={float(jnp.max(jnp.abs(g_e))):.3e}  "
      f"grad w_proj: max_abs={float(jnp.max(jnp.abs(g_w))):.3e}")
assert float(jnp.max(jnp.abs(g_e))) > 0, "erase projection has zero gradient"
assert float(jnp.max(jnp.abs(g_w))) > 0, "write projection has zero gradient"

print("\nAll tests done.")
