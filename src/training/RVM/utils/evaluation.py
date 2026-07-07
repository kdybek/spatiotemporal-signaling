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


@jax.jit
def forward(model, params, sources, targets, target_deltas, rng_key):
    output = model.apply(
        {"params": params},
        sources,
        targets,
        target_deltas,
        method=model.reconstruct,
        rngs=rng_key,
    )

    mask = jax.image.resize(
        output["mask"], targets.shape[:-1] + (1,), method='nearest'
    )
    output["mask"] = mask

    return output


def compute_outputs(
    model,
    test_dataset, 
    params, 
    src_frames, 
    tgt_frames, 
    batch_size, 
    rng_key
):
    reconstruced = []
    masks = []
    features = []
    targets = []
    all_exp_names = []
    for clips, exp_names in tqdm(batch_iterator(test_dataset, batch_size=batch_size, exp_name=True)):
        src, tgt, offsets = prepare_rvm_src_tgt_pairs(
            clips, src_frames, tgt_frames
        )
        all_exp_names.extend(exp_names)

        eval_key, rng_key = jax.random.split(rng_key)
        output = forward(model, params, src, tgt, offsets, eval_key)

        reconstruced.extend(np.array(output["reconstructed"]))
        masks.extend(np.array(output["mask"]))
        features.extend(np.array(output["features"]))
        targets.extend(np.array(tgt))

    reconstruced = np.array(reconstruced)
    masks = np.array(masks)
    features = np.array(features)
    targets = np.array(targets)

    return reconstruced, masks, features, targets, all_exp_names


def visualize_reconstruction(reconstructed, target, mask):
    reconstructed = np.clip(reconstructed, 0, 1)
    masked_view = target * (1 - mask) + 0.5 * mask
    combined = target * (1 - mask) + reconstructed * mask

    masked_view = (masked_view * 255).astype(np.uint8)
    combined = (combined * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)

    metrics = {}

    C = target.shape[-1]
    for c in range(C):
        metrics[f"evaluation/channel_{c}/masked_view"] = wandb.Video(masked_view[..., c])
        metrics[f"evaluation/channel_{c}/combined"] = wandb.Video(combined[..., c])
        metrics[f"evaluation/channel_{c}/target"] = wandb.Video(target[..., c])

    return metrics


def visualize_features(features, labels):
    print(f"Visualizing features with initial shape: {features.shape}")
    features = features.mean(axis=1)  # Average over patches

    assert len(labels) == features.shape[0], "Number of labels must match the number of feature vectors."

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
    )
    latent_trajs_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        latent_trajs_2d[:, 0],
        latent_trajs_2d[:, 1],
        c=y,
        s=10,
    )

    ax.set_title("t-SNE Embeddings")

    return {"evaluation/tsne": wandb.Image(fig)}


def evaluate_probing(features, labels, cv=5, random_state=42):
    features = features.mean(axis=1)  # Average over patches

    assert len(labels) == features.shape[0], \
        "Number of labels must match the number of feature vectors."

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
        "evaluation/probing/mean_accuracy": np.mean(scores),
        "evaluation/probing/std_accuracy": np.std(scores),
    }


def full_evaluation(model, test_dataset, params, src_frames, tgt_frames, batch_size, rng_key):
    reconstruced, masks, features, targets, exp_names = compute_outputs(
        model,
        test_dataset,
        params,
        src_frames,
        tgt_frames,
        batch_size,
        rng_key
    )

    mse_loss = np.mean(masks * (reconstruced - targets) ** 2)

    metrics = {
        "evaluation/loss": mse_loss,
    }

    metrics.update(visualize_reconstruction(reconstruced, targets, masks))
    metrics.update(visualize_features(features, exp_names))
    metrics.update(evaluate_probing(features, exp_names))
    
    return metrics