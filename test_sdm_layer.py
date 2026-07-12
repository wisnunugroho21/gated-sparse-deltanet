"""Numerical verification of sdm_layer.py — the Sparse Delta Memory token
mixer (Cabannes et al., arXiv:2607.07386, Sec. 3.1-3.3 / Fig. 2) — and of
its streaming contract (step / init_cache).

Run with:  python3 test_sdm_layer.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx

jax.config.update("jax_enable_x64", False)

from sdm_layer import SparseDeltaMemory

D, H, SN, W, R, dv = 64, 2, 16, 12, 12, 16   # N = SN² = 256 slots/head
B, L = 2, 96

layer = SparseDeltaMemory(d_model=D, num_heads=H, sqrt_n=SN,
                          num_writes=W, num_reads=R, head_v_dim=dv,
                          rngs=nnx.Rngs(0))
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
# 37 + 27 + 32: arbitrary split points — no chunk-size or conv-context
# constraint exists for SDM (the table IS the whole state).
segs = [x[:, :37], x[:, 37:64], x[:, 64:]]
cache = layer.init_cache(B)
outs = []
for s in segs:
    o, cache = layer.step(s, cache)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y)))
dM = float(jnp.max(jnp.abs(cache.memory - cache_full.memory)))
print(f"segmented (37+27+32) vs full: max|dy| = {d:.3e}  max|dM| = {dM:.3e}")
assert d < 1e-4 and dM < 1e-4

print("\n=== Test 4: token-by-token decode equals full pass ===")
cache = layer.init_cache(B)
outs = []
for t in range(16):
    o, cache = layer.step(x[:, t:t + 1], cache)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y[:, :16])))
print(f"decode (L=1 steps) vs full: max|dy| = {d:.3e}")
assert d < 1e-4

print("\n=== Test 5: product-key selection contract (Sec. 3.1 step 1) ===")
idx, val = layer._select(layer.k_proj(x), layer.W)
idx_np, val_np = np.asarray(idx), np.asarray(val)
print(f"idx {idx_np.shape} in [{idx_np.min()}, {idx_np.max()}]  (N = {layer.N})")
assert idx.shape == (B, H, L, W) and val.shape == (B, H, L, W)
assert idx_np.min() >= 0 and idx_np.max() < layer.N
# distinct slots per token — the core's scatter relies on this
uniq = min(len(np.unique(idx_np[b, h, t]))
           for b in range(B) for h in range(H) for t in range(0, L, 7))
print(f"min #distinct slot ids per token: {uniq} (must be {W})")
assert uniq == W
# softmax over the selected scores: positive, sums to 1, descending
s = np.abs(val_np.sum(-1) - 1.0).max()
mono = bool(np.all(np.diff(val_np, axis=-1) <= 1e-7))
print(f"weights: max|sum-1| = {s:.2e}  positive={bool((val_np > 0).all())}  "
      f"descending={mono}")
assert s < 1e-5 and (val_np > 0).all() and mono

print("\n=== Test 6: paper-default sizing — sqrt_n = d/(4H) (Sec. 3.2/3.3) ===")
auto = SparseDeltaMemory(d_model=256, num_heads=1, num_writes=16,
                         num_reads=16, head_v_dim=16, rngs=nnx.Rngs(2))
print(f"d=256, H=1: sqrt_n={auto.sqrt_n}  N={auto.N}  "
      f"table/head = {auto.N * auto.dv * 4 / 1024:.0f} KB fp32")
assert auto.sqrt_n == 64 and auto.N == 4096
ya = auto(jax.random.normal(jax.random.PRNGKey(3), (1, 8, 256), jnp.float32))
assert ya.shape == (1, 8, 256) and bool(jnp.all(jnp.isfinite(ya)))

print("\n=== Test 7: decay parameterization matches the GDN/FLA recipe (Sec. 4) ===")
a = np.asarray(layer.A_log[...])
dt_bias = np.asarray(layer.dt_bias[...])
dt = np.log1p(np.exp(dt_bias))  # softplus(b_dt) = decay magnitude at W_a x ≈ 0
print(f"A_log shape {a.shape} (per head);  exp(A_log) in "
      f"[{np.exp(a).min():.2f}, {np.exp(a).max():.2f}] (family: [1, 16])")
print(f"dt_bias shape {dt_bias.shape} (per head);  softplus(b_dt) in "
      f"[{dt.min():.2e}, {dt.max():.2e}] (family: [1e-3, 1e-1])")
assert a.shape == (H,), "A must be a per-head scalar parameter (Sec. 3.1)"
assert dt_bias.shape == (H,), "b_dt must be a per-head scalar (Sec. 3.1)"
assert 1.0 <= np.exp(a).min() and np.exp(a).max() <= 16.0
assert 0.9e-3 <= dt.min() and dt.max() <= 1.1e-1

print("\n=== Test 8: learned initial state M0 (Sec. 3.1) ===")
assert layer.M0 is not None and layer.M0[...].shape == (H, layer.N, dv)
def loss(model, xx):
    return jnp.sum(model(xx) ** 2)
grads = nnx.grad(loss)(layer, x)
gM0 = grads["M0"][...]
print(f"M0 {layer.M0[...].shape}: grad max|g| = {float(jnp.max(jnp.abs(gM0))):.3e} "
      f"(nonzero -> the table IS trained as parametric memory)")
assert float(jnp.max(jnp.abs(gM0))) > 0.0
# null-M0 ablation variant: no parameter, zero-initialized cache
null = SparseDeltaMemory(d_model=D, num_heads=H, sqrt_n=SN, num_writes=W,
                         num_reads=R, head_v_dim=dv, learned_init=False,
                         rngs=nnx.Rngs(4))
assert null.M0 is None
assert float(jnp.max(jnp.abs(null.init_cache(B).memory))) == 0.0
yn = null(x)
print(f"learned_init=False: out {yn.shape}  finite={bool(jnp.all(jnp.isfinite(yn)))}")
assert bool(jnp.all(jnp.isfinite(yn)))

print("\n=== Test 9: gradients reach every parameter ===")
flat, _ = jax.tree_util.tree_flatten(grads)
bad = [g_ for g_ in flat if not bool(jnp.all(jnp.isfinite(g_)))]
zero = [g_ for g_ in flat if float(jnp.max(jnp.abs(g_))) == 0.0]
print(f"{len(flat)} parameter tensors — non-finite: {len(bad)}, all-zero: {len(zero)}")
assert not bad and not zero

print("\n=== Test 10: return_state / initial_state — EXACT state carry ===")
# Unlike the GDN-2 layer (conv left-context caveat), the SDM table is the
# ENTIRE layer state, so __call__-level continuation must be exact.
y10, M10 = layer(x, return_state=True)
d_y = float(jnp.max(jnp.abs(y10 - y)))
d_M = float(jnp.max(jnp.abs(M10 - cache_full.memory)))
print(f"out vs plain __call__: max|dy| = {d_y:.3e}   "
      f"M vs step cache: max|dM| = {d_M:.3e}")
assert M10.shape == (B, H, layer.N, dv)
assert d_y == 0.0 and d_M < 1e-5

_, M_half = layer(x[:, :48], return_state=True)
y_cont = layer(x[:, 48:], initial_state=M_half)
d = float(jnp.max(jnp.abs(y_cont - y[:, 48:])))
print(f"warm-started second half vs full pass: max|dy| = {d:.3e}")
assert d < 1e-5, "SDM continuation must be exact (no conv context to lose)"

print("\n=== Test 11: core='chunkwise' (App. A) matches the recurrent core ===")
# Same seed -> identical parameters; only the execution core differs.
chunk_layer = SparseDeltaMemory(d_model=D, num_heads=H, sqrt_n=SN,
                                num_writes=W, num_reads=R, head_v_dim=dv,
                                core="chunkwise", chunk_size=32,
                                rngs=nnx.Rngs(0))
y_ch, M_ch = chunk_layer(x, return_state=True)   # L=96 = 3 chunks of 32
d_y = float(jnp.max(jnp.abs(y_ch - y)))
d_M = float(jnp.max(jnp.abs(M_ch - cache_full.memory)))
print(f"chunkwise vs recurrent layer: max|dy| = {d_y:.3e}  max|dM| = {d_M:.3e}")
assert d_y < 1e-4 and d_M < 1e-4

print("\n=== Test 12: chunkwise __call__ validates L % chunk_size ===")
try:
    chunk_layer(x[:, :50])
    raise AssertionError("expected ValueError for L=50, C=32")
except ValueError as e:
    print(f"__call__(L=50) raises: {e}")

print("\n=== Test 13: chunkwise step — prefill split + decode ===")
# 70 = 2 aligned chunks (64) through the chunkwise core + 6-token
# recurrent tail; then 26 more (tail-only). Must equal the recurrent
# full pass.
cache = chunk_layer.init_cache(B)
o1, cache = chunk_layer.step(x[:, :70], cache)
o2, cache = chunk_layer.step(x[:, 70:], cache)
d_y = float(jnp.max(jnp.abs(jnp.concatenate([o1, o2], 1) - y)))
d_M = float(jnp.max(jnp.abs(cache.memory - cache_full.memory)))
print(f"chunkwise step (70+26) vs recurrent full: max|dy| = {d_y:.3e}  "
      f"max|dM| = {d_M:.3e}")
assert d_y < 1e-4 and d_M < 1e-4
# token-by-token decode on top of the chunkwise-prefilled cache
cache2 = chunk_layer.init_cache(B)
o_pre, cache2 = chunk_layer.step(x[:, :64], cache2)
outs = [o_pre]
for t in range(64, 72):
    o, cache2 = chunk_layer.step(x[:, t:t + 1], cache2)
    outs.append(o)
d = float(jnp.max(jnp.abs(jnp.concatenate(outs, 1) - y[:, :72])))
print(f"prefill(64) + decode(8×L=1) vs full: max|dy| = {d:.3e}")
assert d < 1e-4
# gradients reach every parameter through the chunkwise core too
grads_ch = nnx.grad(loss)(chunk_layer, x[:, :64])
flat_ch, _ = jax.tree_util.tree_flatten(grads_ch)
bad = [g_ for g_ in flat_ch if not bool(jnp.all(jnp.isfinite(g_)))]
zero = [g_ for g_ in flat_ch if float(jnp.max(jnp.abs(g_))) == 0.0]
print(f"chunkwise grads: {len(flat_ch)} tensors — non-finite: {len(bad)}, "
      f"all-zero: {len(zero)}")
assert not bad and not zero

print("\nAll tests done.")
