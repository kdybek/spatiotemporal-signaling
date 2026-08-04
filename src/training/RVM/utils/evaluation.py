from tqdm import tqdm
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

from utils.dataloader import batch_iterator, prepare_src_tgt_pairs


EVAL_SUBFOLDER = "evaluation"
RECONSTRUCTION_SUBFOLDER = "reconstruction"
DIM_RED_SUBFOLDER = "dimensionality_reduction"


def compute_outputs(
    model,
    test_dataset,
    params,
    src_frames,
    tgt_frames,
    src_sample_prefix,
    min_offset,
    max_offset,
    batch_size,
    rng_key
):
    @jax.jit
    def forward(sources, targets, target_deltas, rng):
        output = model.apply(
            {"params": params},
            sources,
            targets,
            target_deltas,
            rngs={"default": rng},
        )

        mask = jax.image.resize(
            output["mask"],
            targets.shape[:-1] + (1,),
            method="nearest",
        )
        # Repeat mask for each channel
        mask = jnp.repeat(mask, targets.shape[-1], axis=-1)

        # Average over spatial and temporal dimensions
        features = jnp.mean(output["features"], axis=(1, 2))

        output["mask"] = mask
        output["features"] = features

        return output

    reconstruced = []
    masks = []
    features = []
    targets = []
    all_exp_names = []
    loader = batch_iterator(test_dataset, batch_size=batch_size, exp_name=True)
    for clips, exp_names in tqdm(loader, desc='Evaluation'):
        src, tgt, offsets = prepare_src_tgt_pairs(
            clips, src_frames, tgt_frames, src_sample_prefix, min_offset, max_offset
        )
        all_exp_names.extend(exp_names)

        eval_key, rng_key = jax.random.split(rng_key)
        output = forward(src, tgt, offsets, rng=eval_key)

        reconstruced.extend(np.array(output["reconstructed"]))
        masks.extend(np.array(output["mask"]))
        features.extend(np.array(output["features"]))
        targets.extend(np.array(tgt))

    reconstruced = np.array(reconstruced)
    masks = np.array(masks)
    features = np.array(features)
    targets = np.array(targets)

    return {
        "reconstructed": reconstruced,
        "masks": masks,
        "features": features,
        "targets": targets,
        "exp_names": all_exp_names,
    }


def visualize_reconstructions(reconstructed, targets, masks, max_samples=8):
    reconstructed = np.clip(reconstructed, 0, 1)
    masked_view = targets * (1 - masks) + 0.5 * masks
    combined = targets * (1 - masks) + reconstructed * masks

    masked_view = (masked_view * 255).astype(np.uint8)
    combined = (combined * 255).astype(np.uint8)
    target = (targets * 255).astype(np.uint8)

    metrics = {}

    C = target.shape[-1]
    for c in range(C):
        for i in range(min(max_samples, target.shape[0])):
            metrics[f"{RECONSTRUCTION_SUBFOLDER}/channel_{c}/image_set_{i}"] = [
                wandb.Image(target[i, 0, ..., c], caption="Target"),
                wandb.Image(masked_view[i, 0, ..., c], caption="Masked View"),
                wandb.Image(combined[i, 0, ..., c], caption="Reconstructed"),
            ]

    return metrics


def visualize_features(features, labels):
    features = jnp.mean(features, axis=1)  # Average over the spatial dimension
    assert len(labels) == features.shape[0], \
        "Number of labels must match the number of feature vectors."

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    if len(features) < 2:
        return {}

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features)),
        random_state=42,
    )
    tsne_features = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        tsne_features[:, 0],
        tsne_features[:, 1],
        c=y,
        s=10,
    )

    ax.set_title("t-SNE Embeddings")

    return {f"{DIM_RED_SUBFOLDER}/tsne": wandb.Image(fig)}


def evaluate_probing(features, labels, cv=5, random_state=42):
    assert len(labels) == features.shape[0], \
        "Number of labels must match the number of feature vectors."

    if len(features) < cv:
        return {}

    features = jnp.mean(features, axis=1)  # Average over the spatial dimension

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    cv_split = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_val_score(
        clf,
        features,
        labels,
        cv=cv_split,
        scoring="accuracy"
    )

    return {
        f"{EVAL_SUBFOLDER}/probing_mean_acc": np.mean(scores),
        f"{EVAL_SUBFOLDER}/probing_std_acc": np.std(scores),
    }


def evaluate_loss(reconstructed, targets, masks):
    error = (reconstructed - targets) ** 2
    mse_loss = np.sum(masks * error) / (np.sum(masks) + 1e-8)

    return {
        f"{EVAL_SUBFOLDER}/loss": mse_loss,
    }


def full_evaluation(
        model,
        test_dataset,
        params,
        src_frames,
        tgt_frames,
        src_sample_prefix,
        min_offset,
        max_offset,
        batch_size,
        rng_key
):
    outputs = compute_outputs(
        model,
        test_dataset,
        params,
        src_frames,
        tgt_frames,
        src_sample_prefix,
        min_offset,
        max_offset,
        batch_size,
        rng_key
    )

    reconstructed = outputs["reconstructed"]
    masks = outputs["masks"]
    features = outputs["features"]
    targets = outputs["targets"]
    exp_names = outputs["exp_names"]

    metrics = {}
    metrics.update(evaluate_loss(reconstructed, targets, masks))
    metrics.update(visualize_reconstructions(reconstructed, targets, masks))
    metrics.update(visualize_features(features, exp_names))
    metrics.update(evaluate_probing(features, exp_names))

    return metrics
