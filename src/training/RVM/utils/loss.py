import jax
import jax.numpy as jnp
import optax


def update_model(
    model,
    model_params,
    opt_state,
    optimizer,
    sources,
    targets,
    target_deltas,
    rng_key,
):
    def loss_fn(params):
        output = model.apply(
            {"params": params},
            sources,
            targets,
            target_deltas,
            rngs=rng_key,
        )

        reconstructed = output["reconstructed"]
        mask = output["mask"]

        mask = jax.image.resize(
            mask, targets.shape[:-1] + (1,), method='nearest'
        )

        mse_loss = jnp.mean(mask * (reconstructed - targets) ** 2)

        return mse_loss

    loss_value, grads = jax.value_and_grad(loss_fn)(model_params)

    updates, opt_state = optimizer.update(grads, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)

    metrics = {
        "training/loss": loss_value,
    }

    return model_params, opt_state, metrics
