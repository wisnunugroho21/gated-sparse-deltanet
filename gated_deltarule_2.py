import jax
import jax.numpy as jnp
from jax import lax

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
    q = q.astype(D_TYPE)
    k = k.astype(D_TYPE)
    v = v.astype(D_TYPE)
    g = g.astype(D_TYPE)
    b = b.astype(D_TYPE)
    w = w.astype(D_TYPE)
    S0 = S0.astype(D_TYPE)

    alpha = jnp.exp(g)
    e = b * k
    z = w * v

    def step(S, inp):
        qt, kt, at, et, zt = inp

        qt = qt[:, None]
        kt = kt[:, None]
        at = at[:, None]
        et = et[:, None]
        zt = zt[:, None]

        S_bar = at * S

        r_t = S_bar.T @ et

        S_new = S_bar + kt * (zt - r_t).T

        o_t = S_new.T @ qt

        return S_new, o_t

    S_final, o = lax.scan(step, S0, (q, k, alpha, e, z))
    return o.squeeze(-1), S_final


def _batchify(fn):
    over_heads = jax.vmap(fn, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))
    return jax.vmap(over_heads, in_axes=(0, 0, 0, 0, 0, 0, 0), out_axes=(0, 0))


def recurrent_gated_delta_rule_2(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _batchify(_recurrent_single)(q, k, v, g, b, w, S0)
