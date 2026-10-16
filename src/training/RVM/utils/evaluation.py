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

from utils.dataloader import batch_iterator, prepare_rvm_src_tgt_pairs


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
    def forward(sources, targets, channel_inds, target_deltas, rng):
        output = model.apply(
            {"params": params},
            sources,
            targets,
            channel_inds,
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

        output["mask"] = mask
        output["reconstructed"] = jnp.take_along_axis(
            output["reconstructed"], channel_inds[:, None, None, None, :], axis=-1
        )
        output["targets"] = jnp.take_along_axis(
            targets, channel_inds[:, None, None, None, :], axis=-1
        )

        return output

    reconstruced = []
    masks = []
    targets = []
    channel_inds = []
    loader = batch_iterator(test_dataset, batch_size=batch_size)
    for clips, channel_inds, _ in tqdm(loader, desc='Evaluation'):
        src, tgt, offsets = prepare_rvm_src_tgt_pairs(
            clips, src_frames, tgt_frames, src_sample_prefix, min_offset, max_offset
        )

        eval_key, rng_key = jax.random.split(rng_key)
        output = forward(src, tgt, channel_inds, offsets, rng=eval_key)

        reconstruced.extend(np.array(output["reconstructed"]))
        masks.extend(np.array(output["mask"]))
        targets.extend(np.array(output["targets"]))
        channel_inds.extend(np.array(channel_inds))

    reconstruced = np.array(reconstruced)
    masks = np.array(masks)
    targets = np.array(targets)
    channel_inds = np.array(channel_inds)

    return {
        "reconstructed": reconstruced,
        "masks": masks,
        "targets": targets,
        "channel_inds": channel_inds,
    }




def visualize_reconstructions(reconstructed, targets, masks, channel_inds, channel_names, max_samples=8):
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
            channel_ind = channel_inds[i, c]
            metrics[f"{RECONSTRUCTION_SUBFOLDER}/{channel_names[channel_ind]}/image_set_{i}"] = [
                wandb.Image(target[i, 0, ..., c], caption="Target"),
                wandb.Image(masked_view[i, 0, ..., c], caption="Masked View"),
                wandb.Image(combined[i, 0, ..., c], caption="Reconstructed"),
            ]

    return metrics


def visualize_features(features, labels):
    assert len(labels) == features.shape[0], \
        "Number of labels must match the number of feature vectors."

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    if len(features) < 2:
        return {}

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features)),
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


def evaluate_probing(features, labels, cv=5):
    assert len(labels) == features.shape[0], \
        "Number of labels must match the number of feature vectors."

    if len(features) < cv:
        return {}

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    cv_split = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
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
        channel_names_list,
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
    targets = outputs["targets"]
    channel_inds = outputs["channel_inds"]

    metrics = {}
    metrics.update(evaluate_loss(reconstructed, targets, masks))
    metrics.update(visualize_reconstructions(reconstructed, targets, channel_inds, channel_names_list, masks))

    return metrics
