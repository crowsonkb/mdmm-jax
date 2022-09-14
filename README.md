# mdmm-jax

Gradient-based constrained optimization for JAX (implementation by Katherine Crowson).

The Modified Differential Multiplier Method was proposed in Platt and Barr (1988), "[Constrained Differential Optimization](https://papers.nips.cc/paper/1987/file/a87ff679a2f3e71d9181a67b7542122c-Paper.pdf)".

MDMM minimizes a main objective f(x) subject to equality (g(x) = 0) and inequality (h(x) ≥ 0) constraints, where the constraints can be arbitrary differentiable functions of your parameters and data.

## Quick usage

Creating an equality constraint and its trainable parameters:

```python
import mdmm_jax

constraint = mdmm_jax.eq(my_function)
# Internally calls my_function(main_params, x)
mdmm_params = constraint.init(main_params, x)
```

Constructing the loss function for the augmented Lagrangian system incorporating the constraint loss (the loss will become far less interpretable so you should return the original loss as part of an auxiliary return value):

```python
def system(params, x):
    main_params, mdmm_params = params
    loss = loss_fn(main_params, x)
    mdmm_loss, inf = constraint.loss(mdmm_params, main_params, x)
    return loss + mdmm_loss, (loss, inf)
```

Turning an [Optax](https://optax.readthedocs.io/en/latest/) base optimizer into an MDMM constrained optimizer:

```python
optimizer = optax.chain(
    optax.sgd(1e-3),
    mdmm_jax.optax_prepare_update(),
)
params = main_params, mdmm_params
opt_state = optimizer.init(params)
```
