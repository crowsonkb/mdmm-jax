#!/usr/bin/env python3

"""Finds the nearest permutation matrix to a signal matrix with constrained
optimization."""

import jax
import jax.numpy as jnp
import optax
from tqdm import trange, tqdm

import mdmm_jax


def total_infeasibility(tree):
    return jax.tree_util.tree_reduce(lambda x, y: x + jnp.sum(jnp.abs(y)), tree, jnp.array(0.))


def main():
    # Create target grid
    m = 10
    ramp = jnp.linspace(0, 0.9, m)
    grid = jnp.meshgrid(ramp, ramp)[0] + jnp.eye(m) * 0.1
    key = jax.random.PRNGKey(63596)
    grid = grid + jax.random.uniform(key, grid.shape) * 0.1
    print('The target signal grid:')
    print(grid)

    def loss_fn(v):
        return -jnp.sum(v * grid)

    # Create the constraints
    damping = 0.1
    constraints = mdmm_jax.combine(
        # Entries are either 0 or 1
        mdmm_jax.eq(lambda v: v * (1 - v), damping=damping),
        # Column sums are 1 (so there is only one nonzero element per column)
        mdmm_jax.eq(lambda v: jnp.sum(v, axis=0) - 1, damping=damping),
        # Row sums are 1 (so there is only one nonzero element per row)
        mdmm_jax.eq(lambda v: jnp.sum(v, axis=1) - 1, damping=damping),
    )

    # Create the random init
    key = jax.random.PRNGKey(63496)
    v = jax.random.uniform(key, [m, m])

    # Create the MDMM trainable parameters and optimizer state
    mdmm_params = constraints.init(v)
    params = v, mdmm_params
    opt = optax.chain(
        optax.sgd(1e-2),
        mdmm_jax.optax_prepare_update(),
    )
    opt_state = opt.init(params)

    # Define the "loss" value for the augmented Lagrangian system optimized by MDMM
    def system(params):
        main_params, mdmm_params = params
        loss = loss_fn(main_params)
        mdmm_loss, inf = constraints.loss(mdmm_params, main_params)
        return loss + mdmm_loss, (loss, inf)

    # Do the optimization
    @jax.jit
    def update(params, opt_state):
        grad, info = jax.grad(system, has_aux=True)(params)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, info

    for i in trange(150000):
        params, opt_state, info = update(params, opt_state)
        if i % 5000 == 0:
            tqdm.write(f'i: {i}, loss: {info[0]:g}, infeasibility: {total_infeasibility(info[1]):g}')

    # Print final matrix
    print('Final permutation matrix closest to signal grid:')
    print(params[0])

if __name__ == '__main__':
    main()
