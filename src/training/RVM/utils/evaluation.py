from tqdm import tqdm
import jax
import optax
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
import re

from utils.dataloader import batch_iterator, prepare_rvm_src_tgt_pairs
from utils.model import get_optimizer, AttentionPoolingClassifier

EVAL_SUBFOLDER = "evaluation"
DOWNSTREAM_SUBFOLDER = "downstream"
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
    eval_steps,
    rng_key,
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

        assert output["mask"].shape[-1] == channel_inds.shape[-1]

        output["reconstructed"] = jnp.take_along_axis(
            output["reconstructed"], channel_inds[:, None, None, None, :], axis=-1
        )
        output["targets"] = jnp.take_along_axis(
            targets, channel_inds[:, None, None, None, :], axis=-1
        )

        assert (
            output["reconstructed"].shape[-3] % output["mask"].shape[-3] == 0
        ), "Target height must be divisible by mask height"
        assert (
            output["reconstructed"].shape[-2] % output["mask"].shape[-2] == 0
        ), "Target width must be divisible by mask width"

        patch_scaling_factor_h = (
            output["reconstructed"].shape[-3] // output["mask"].shape[-3]
        )
        patch_scaling_factor_w = (
            output["reconstructed"].shape[-2] // output["mask"].shape[-2]
        )

        mask = output["mask"].repeat(patch_scaling_factor_h, axis=-3)
        mask = mask.repeat(patch_scaling_factor_w, axis=-2)

        output["mask"] = mask

        return output

    reconstruced = []
    masks = []
    targets = []
    channel_inds = []
    step = 1
    while step <= eval_steps:
        loader = batch_iterator(test_dataset, batch_size=batch_size)
        for clips, channel_inds_, _ in tqdm(loader, desc="Evaluation"):
            src, tgt, offsets = prepare_rvm_src_tgt_pairs(
                clips, src_frames, tgt_frames, src_sample_prefix, min_offset, max_offset
            )

            eval_key, rng_key = jax.random.split(rng_key)
            output = forward(src, tgt, channel_inds_, offsets, rng=eval_key)

            reconstruced.extend(np.array(output["reconstructed"]))
            masks.extend(np.array(output["mask"]))
            targets.extend(np.array(output["targets"]))
            channel_inds.extend(np.array(channel_inds_))

            step += 1
            if step > eval_steps:
                break

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


def visualize_reconstructions(
    reconstructed, targets, masks, channel_inds, channel_names, max_samples=8
):
    reconstructed = reconstructed[:max_samples]
    targets = targets[:max_samples]
    masks = masks[:max_samples]
    
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
            metrics[
                f"{RECONSTRUCTION_SUBFOLDER}/{channel_names[channel_ind]}/image_set_{i}"
            ] = [
                wandb.Image(target[i, 0, ..., c], caption="Target"),
                wandb.Image(masked_view[i, 0, ..., c], caption="Masked View"),
                wandb.Image(combined[i, 0, ..., c], caption="Reconstructed"),
            ]

    return metrics


def visualize_features(features, labels):
    assert (
        len(labels) == features.shape[0]
    ), "Number of labels must match the number of feature vectors."

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
    assert (
        len(labels) == features.shape[0]
    ), "Number of labels must match the number of feature vectors."

    if len(features) < cv:
        return {}

    clf = Pipeline(
        [("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]
    )

    cv_split = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
    )

    scores = cross_val_score(clf, features, labels, cv=cv_split, scoring="accuracy")

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


def partial_evaluation(
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
    eval_steps,
    rng_key,
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
        eval_steps,
        rng_key,
    )

    reconstructed = outputs["reconstructed"]
    masks = outputs["masks"]
    targets = outputs["targets"]
    channel_inds = outputs["channel_inds"]

    metrics = {}
    metrics.update(
        evaluate_loss(reconstructed=reconstructed, targets=targets, masks=masks)
    )
    metrics.update(
        visualize_reconstructions(
            reconstructed=reconstructed,
            targets=targets,
            masks=masks,
            channel_inds=channel_inds,
            channel_names=channel_names_list,
        )
    )

    return metrics


def sanitize_cell_type(cell_type):
    s = re.sub(r"^MCF10A_", "", cell_type)
    s = re.sub(r"_Clone.*$", "", s)

    return s


def find_classes(dataset, target):
    classes = set()
    all_metadata = dataset.get_all_metadata()

    for metadata in all_metadata:
        if target == "cell_type":
            cell_type = metadata["Cell_type"]
            cell_type = sanitize_cell_type(cell_type)
            classes.add(cell_type)
        elif target == "treatment":
            treatment = metadata["Inhibitor"]
            classes.add(treatment)
        else:
            raise ValueError(f"Unknown target: {target}")

    return sorted(list(classes))


def downstream_evaluation(
    foundation_model,
    foundation_model_params,
    downstream_train_dataset,
    downstream_test_dataset,
    downstream_train_steps,
    downstream_eval_interval,
    downstream_eval_steps,
    batch_size,
    base_lr,
    grad_accumulation_steps,
    main_training_step,
    rng_key,
):
    metrics = {}
    for target in ["cell_type", "treatment"]:
        m = downstream_evaluation_(
            foundation_model,
            foundation_model_params,
            downstream_train_dataset,
            downstream_test_dataset,
            target,
            downstream_train_steps,
            downstream_eval_interval,
            downstream_eval_steps,
            batch_size,
            base_lr,
            grad_accumulation_steps,
            main_training_step,
            rng_key,
        )
        metrics.update(m)

    return metrics


def downstream_evaluation_(
    foundation_model,
    foundation_model_params,
    downstream_train_dataset,
    downstream_test_dataset,
    target,
    downstream_train_steps,
    downstream_eval_interval,
    downstream_eval_steps,
    batch_size,
    base_lr,
    grad_accumulation_steps,
    main_training_step,
    rng_key,
):
    classes_train = find_classes(downstream_train_dataset, target)
    classes_test = find_classes(downstream_test_dataset, target)

    assert set(classes_train) == set(classes_test), "Train and test classes must match."

    class_to_index = {cls: idx for idx, cls in enumerate(classes_train)}
    num_classes = len(classes_train)

    print(f"Downstream evaluation for target: {target}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {classes_train}")

    downstream_model = AttentionPoolingClassifier(num_classes)

    latent_emb_dim = foundation_model.latent_emb_dim
    dummy_input = jnp.ones((1, 1, latent_emb_dim))
    init_key, rng_key = jax.random.split(rng_key)
    downstream_params = downstream_model.init(init_key, dummy_input)["params"]

    optimizer = get_optimizer(
        base_lr=base_lr, 
        warmup_steps=int(0.05 * downstream_train_steps), 
        total_steps=downstream_train_steps, 
        grad_accumulation_steps=grad_accumulation_steps
    )
    opt_state = optimizer.init(downstream_params)

    @jax.jit
    def training_step(
        params,
        opt_state,
        src_frames,
        channel_inds,
        labels,
        rng_key
    ):
        features = foundation_model.apply(
            {"params": foundation_model_params}, 
            src_frames, 
            channel_inds, 
            method=foundation_model.encode,
            rngs={"default": rng_key}
        )

        features = features.reshape(features.shape[0], -1, features.shape[-1])
        
        def loss_fn(params, features, labels):
            output = downstream_model.apply({"params": params}, features)
            logits = output["logits"]
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params, features, labels)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    @jax.jit
    def predict(params, src_frames, channel_inds, rng_key):
        features = foundation_model.apply(
            {"params": foundation_model_params}, 
            src_frames, 
            channel_inds, 
            method=foundation_model.encode,
            rngs={"default": rng_key}
        )
        features = features.reshape(features.shape[0], -1, features.shape[-1])
        output = downstream_model.apply({"params": params}, features)
        logits = output["logits"]
        
        return logits

    wandb.define_metric(f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_fd_step_{main_training_step}_train_loss", step_metric=f"tgt_{target}_fd_step_{main_training_step}_step")
    wandb.define_metric(f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_fd_step_{main_training_step}_test_loss", step_metric=f"tgt_{target}_fd_step_{main_training_step}_step")
    wandb.define_metric(f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_fd_step_{main_training_step}_test_accuracy", step_metric=f"tgt_{target}_fd_step_{main_training_step}_step")

    def get_labels_from_metadata(metadata):
        labels = []
        for meta in metadata:
            if target == "cell_type":
                cell_type = meta["Cell_type"]
                cell_type = sanitize_cell_type(cell_type)
                label_idx = class_to_index[cell_type]
            elif target == "treatment":
                treatment = meta["Inhibitor"]
                label_idx = class_to_index[treatment]
            else:
                raise ValueError(f"Unknown target: {target}")
            
            labels.append(label_idx)

        return np.array(labels)

    def eval(ds_params, rng_key, train_step):
        all_predictions = []
        all_logits = []
        all_labels = []
        eval_step = 1
        while eval_step <= downstream_eval_steps:
            loader = batch_iterator(downstream_test_dataset, batch_size=batch_size)
            for clips, channel_inds, metadata in tqdm(loader, desc="Downstream testing"):

                clips = np.transpose(clips, (0, 1, 3, 4, 2))

                labels = get_labels_from_metadata(metadata)

                logits = predict(ds_params, clips, channel_inds, rng_key)
                predictions = jnp.argmax(logits, axis=-1)
                all_predictions.extend(np.array(predictions))
                all_logits.extend(np.array(logits))
                all_labels.extend(labels)

                eval_step += 1
                if eval_step > downstream_eval_steps:
                    break

        all_predictions = np.array(all_predictions)
        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        accuracy = np.mean(all_predictions == all_labels)
        loss = optax.softmax_cross_entropy_with_integer_labels(all_logits, all_labels).mean()

        wandb.log({
            f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_fd_step_{main_training_step}_test_loss": loss,
            f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_fd_step_{main_training_step}_test_accuracy": accuracy,
            f"tgt_{target}_fd_step_{main_training_step}_step": train_step
        })

        return {
            f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_test_accuracy": accuracy,
        }

    eval_key, rng_key = jax.random.split(rng_key)
    metrics = eval(downstream_params, eval_key, train_step=0)

    train_step = 1
    while train_step <= downstream_train_steps:
        loader = batch_iterator(downstream_train_dataset, batch_size=batch_size)
        for clips, channel_inds, metadata in tqdm(loader, desc=f"Downstream training"):

            clips = np.transpose(clips, (0, 1, 3, 4, 2))

            labels = get_labels_from_metadata(metadata)

            step_key, rng_key = jax.random.split(rng_key)
            downstream_params, opt_state, loss = training_step(
                downstream_params,
                opt_state,
                clips,
                channel_inds,
                labels,
                step_key
            )

            if train_step % downstream_eval_interval == 0:
                eval_key, rng_key = jax.random.split(rng_key)
                metrics = eval(downstream_params, eval_key, train_step)  # Override metrics with the latest evaluation results

            wandb.log({
                f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_fd_step_{main_training_step}_train_loss": loss,
                f"tgt_{target}_fd_step_{main_training_step}_step": train_step
            })

            train_step += 1
            if train_step > downstream_train_steps:
                break

    return metrics
