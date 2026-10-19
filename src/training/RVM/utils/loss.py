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
    channel_inds,
    target_deltas,
    rng_key,
):
    def loss_fn(params):
        output = model.apply(
            {"params": params},
            sources,
            targets,
            channel_inds,
            target_deltas,
            rngs={"default": rng_key},
        )

        assert output["mask"].shape[-1] == channel_inds.shape[-1]

        reconstructed = jnp.take_along_axis(output["reconstructed"], channel_inds[:, None, None, None, :], axis=-1)
        targets_ = jnp.take_along_axis(targets, channel_inds[:, None, None, None, :], axis=-1)

        assert reconstructed.shape[-3] % output["mask"].shape[-3] == 0, "Target height must be divisible by mask height"
        assert reconstructed.shape[-2] % output["mask"].shape[-2] == 0, "Target width must be divisible by mask width"

        patch_scaling_factor_h = targets_.shape[-3] // output["mask"].shape[-3]
        patch_scaling_factor_w = targets_.shape[-2] // output["mask"].shape[-2]

        mask = output["mask"].repeat(patch_scaling_factor_h, axis=-3)
        mask = mask.repeat(patch_scaling_factor_w, axis=-2)

        error = (reconstructed - targets_) ** 2
        mse_loss = jnp.sum(mask * error) / (jnp.sum(mask) + 1e-8)

        return mse_loss

    loss_value, grads = jax.value_and_grad(loss_fn)(model_params)

    updates, opt_state = optimizer.update(grads, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)

    metrics = {
        "training/loss": loss_value,
    }

    return model_params, opt_state, metrics
