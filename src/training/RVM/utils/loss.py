import jax
import jax.numpy as jnp
import optax


@jax.jit
def update_model(
    model,
    params,
    opt_state,
    optimizer,
    sources,
    targets,
    target_deltas,
    rng_key,
):
    if target_deltas is None:
        target_deltas = jnp.zeros(targets.shape[:2], dtype=jnp.int32)

    def loss_fn(params):
        output = model.apply(
            {"params": params},
            sources,
            targets,
            target_deltas,
            method=model.reconstruct,
            rngs=rng_key,
        )

        reconstructed = output["reconstructed"]
        mask = output["mask"]

        mse_loss = jnp.mean(mask * (reconstructed - targets) ** 2)

        return mse_loss

    loss_value, grads = jax.value_and_grad(loss_fn)(params)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    metrics = {
        "loss": loss_value,
    }

    return params, opt_state, metrics
