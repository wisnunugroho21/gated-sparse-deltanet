import jax
import jax.numpy as jnp
from jax import lax

D_TYPE = jnp.float32


def _chunkwise_single_faithful(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    b: jax.Array,
    w: jax.Array,
    S0: jax.Array,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    L, dk = k.shape
    dv = v.shape[-1]
    C = chunk_size
    if L % C:
        raise ValueError(
            f"sequence length L={L} must be divisible by chunk_size={C}")
    N = L // C  # number of chunks

    def to_chunks(x):
        return x.reshape(N, C, x.shape[-1]).astype(D_TYPE)

    q, k, v = to_chunks(q), to_chunks(k), to_chunks(v)
    g, b, w = to_chunks(g), to_chunks(b), to_chunks(w)

    eye = jnp.eye(C, dtype=D_TYPE)
    S0 = S0.astype(D_TYPE)

    G = jnp.cumsum(g, axis=1)

    gamma = jnp.exp(G)

    gamma_C = gamma[:, -1]

    Kbar = k * jnp.exp(-G)

    Ebar = gamma * (b * k)

    Z = w * v

    Qg = gamma * q

    T = jnp.tril(Ebar @ Kbar.swapaxes(-1, -2), k=-1)

    A = jax.scipy.linalg.solve_triangular(
        eye + T, jnp.broadcast_to(eye, T.shape), lower=True, unit_diagonal=True
    )

    Y = A @ Ebar

    U = A @ Z

    Aqk = jnp.tril(Qg @ Kbar.swapaxes(-1, -2))

    Ktail = k * (gamma_C[:, None, :] / gamma)

    def chunk_step(
        S: jax.Array,
        inp: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        Y_n, U_n, Aqk_n, Qg_n, Ktail_n, gamma_C_n = inp

        R = U_n - Y_n @ S

        o = Qg_n @ S + Aqk_n @ R

        S_new = gamma_C_n[:, None] * S + Ktail_n.T @ R

        return S_new, o

    S_final, o = lax.scan(chunk_step, S0, (Y, U, Aqk, Qg, Ktail, gamma_C))
    return o.reshape(L, dv), S_final

def _batchify(fn):
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
    chunk_size: int = 64
) -> tuple[jax.Array, jax.Array]:
    def fun(
        Q: jax.Array,
        K: jax.Array,
        V: jax.Array,
        G: jax.Array,
        B: jax.Array,
        W: jax.Array,
        So: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _chunkwise_single_faithful(
            Q, K, V, G, B, W, So, chunk_size=chunk_size)

    return _batchify(fun)(q, k, v, g, b, w, S0)
