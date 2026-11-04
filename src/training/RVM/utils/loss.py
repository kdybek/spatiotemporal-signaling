import jax
import jax.numpy as jnp
import optax


def masked_rank(x, mask, axis):
    """
    Compute ranks along axis while ignoring masked-out values.
    """
    x_masked = jnp.where(mask > 0, x, jnp.inf)

    ranks = jnp.argsort(
        jnp.argsort(x_masked, axis=axis),
        axis=axis
    ).astype(x.dtype)

    return ranks * mask


def spearman_correlation(x, y, mask, axis, eps=1e-8):
    """
    Compute Spearman correlation along given axes and average over remaining axes.

    Args:
        x: input tensor
        y: target tensor
        mask: binary mask with same shape as x/y
        axis: axes over which to compute correlation.
              If None, flatten everything.
        eps: numerical stability constant

    Returns:
        scalar mean Spearman correlation
    """
    if axis is None:
        axis = tuple(range(x.ndim))

    # Rank the values
    x_ranked = masked_rank(x, mask, axis=axis)
    y_ranked = masked_rank(y, mask, axis=axis)

    # Compute Pearson correlation on the ranked values
    return masked_correlation(x_ranked, y_ranked, mask, axis=axis, eps=eps)


def masked_correlation(x, y, mask, axis, eps=1e-8):
    """
    Compute masked Pearson correlation along given axes and average over remaining axes.

    Args:
        x: input tensor
        y: target tensor
        mask: binary mask with same shape as x/y
        axis: axes over which to compute correlation.
              If None, flatten everything.
        eps: numerical stability constant

    Returns:
        scalar mean masked Pearson correlation
    """
    if axis is None:
        axis = tuple(range(x.ndim))

    # Mean over correlation axes
    mask_sum = jnp.sum(mask, axis=axis, keepdims=True) + eps

    x_mean = jnp.sum(mask * x, axis=axis, keepdims=True) / mask_sum
    y_mean = jnp.sum(mask * y, axis=axis, keepdims=True) / mask_sum

    x_centered = x - x_mean
    y_centered = y - y_mean

    covariance = (
        jnp.sum(mask * x_centered * y_centered, axis=axis)
        / mask_sum.squeeze(axis)
    )

    x_var = (
        jnp.sum(mask * x_centered**2, axis=axis)
        / mask_sum.squeeze(axis)
    )

    y_var = (
        jnp.sum(mask * y_centered**2, axis=axis)
        / mask_sum.squeeze(axis)
    )

    corr = covariance / (jnp.sqrt(x_var * y_var) + eps)

    # Mean over all remaining dimensions
    return jnp.mean(corr)


def reconstruction_metrics(reconstructed, targets, mask, var_scaled_loss):
    error = (reconstructed - targets) ** 2

    per_sample_loss = (
        jnp.sum(mask * error, axis=(2, 3))
        / (jnp.sum(mask, axis=(2, 3)) + 1e-8)
    )

    if var_scaled_loss:
        var = jnp.var(targets, axis=(2, 3))
        weights = 1.0 / (var + 1e-8)
        weights = weights / jnp.mean(weights)

        mse_loss = jnp.mean(weights * per_sample_loss)
    else:
        mse_loss = jnp.mean(per_sample_loss)

    # Compute correlations
    targets_flat = targets.reshape(targets.shape[0], targets.shape[1], -1, targets.shape[-1])
    reconstructed_flat = reconstructed.reshape(reconstructed.shape[0], reconstructed.shape[1], -1, reconstructed.shape[-1])
    mask_flat = mask.reshape(mask.shape[0], mask.shape[1], -1, mask.shape[-1])
    spearman_corr = spearman_correlation(reconstructed_flat, targets_flat, mask_flat, axis=2)
    pearson_corr = masked_correlation(reconstructed_flat, targets_flat, mask_flat, axis=2)
    
    return mse_loss, spearman_corr, pearson_corr      


def update_model(
    model,
    model_params,
    opt_state,
    optimizer,
    sources,
    targets,
    channel_inds,
    target_deltas,
    var_scaled_loss,
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

        reconstructed = output["reconstructed"]
        mask = output["mask"]

        assert mask.shape[-1] == channel_inds.shape[-1]
            
        reconstructed = jnp.take_along_axis(reconstructed, channel_inds[:, None, None, None, :], axis=-1)
        targets_ = jnp.take_along_axis(targets, channel_inds[:, None, None, None, :], axis=-1)
            
        assert reconstructed.shape[-3] % mask.shape[-3] == 0, "Target height must be divisible by mask height"
        assert reconstructed.shape[-2] % mask.shape[-2] == 0, "Target width must be divisible by mask width"
            
        patch_scaling_factor_h = targets_.shape[-3] // mask.shape[-3]
        patch_scaling_factor_w = targets_.shape[-2] // mask.shape[-2]
            
        mask = mask.repeat(patch_scaling_factor_h, axis=-3)
        mask = mask.repeat(patch_scaling_factor_w, axis=-2)

        mse_loss, spearman_corr, pearson_corr = reconstruction_metrics(
            reconstructed, targets_, channel_inds, mask, var_scaled_loss
        )

        return mse_loss, {
            "spearman_correlation": spearman_corr,
            "pearson_correlation": pearson_corr,
        }

    (loss_value, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_params)

    updates, opt_state = optimizer.update(grads, opt_state, model_params)
    model_params = optax.apply_updates(model_params, updates)

    metrics = {
        "training/loss": loss_value,
        "training/spearman_correlation": aux["spearman_correlation"],
        "training/pearson_correlation": aux["pearson_correlation"],
    }

    return model_params, opt_state, metrics
