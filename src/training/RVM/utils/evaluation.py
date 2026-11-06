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
from collections import defaultdict

from utils.dataloader import batch_iterator, prepare_rvm_src_tgt_pairs
from utils.model import get_optimizer, AttentionPoolingClassifier, AttentionPoolingOrderClassifier
from utils.loss import reconstruction_metrics

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


def evaluate_reconstruction(reconstructed, targets, masks, var_scaled_loss):
    mse_loss, spearman_corr, pearson_corr = reconstruction_metrics(
        reconstructed, targets, masks, var_scaled_loss
    )

    return {
        f"{EVAL_SUBFOLDER}/loss": mse_loss,
        f"{EVAL_SUBFOLDER}/spearman_correlation": spearman_corr,
        f"{EVAL_SUBFOLDER}/pearson_correlation": pearson_corr,
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
    var_scaled_loss,
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
        evaluate_reconstruction(reconstructed=reconstructed, targets=targets, masks=masks, var_scaled_loss=var_scaled_loss)
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
        elif target == "cell_type_treatment":
            cell_type = metadata["Cell_type"]
            cell_type = sanitize_cell_type(cell_type)
            treatment = metadata["Inhibitor"]
            classes.add(f"{cell_type}_{treatment}")
        elif target in ["order_same_loc", "order_diff_loc"]:
            classes = set([0, 1])
            break
        else:
            raise ValueError(f"Unknown target: {target}")

    return sorted(list(classes))


def get_labels(metadata, target, class_to_index, t1, t2):
    labels = []
    for meta in metadata:
        if target == "cell_type":
            cell_type = meta["Cell_type"]
            cell_type = sanitize_cell_type(cell_type)
            label_idx = class_to_index[cell_type]
        elif target == "treatment":
            treatment = meta["Inhibitor"]
            label_idx = class_to_index[treatment]
        elif target == "cell_type_treatment":
            cell_type = meta["Cell_type"]
            cell_type = sanitize_cell_type(cell_type)
            treatment = meta["Inhibitor"]
            label_idx = class_to_index[f"{cell_type}_{treatment}"]
        elif target in ["order_same_loc", "order_diff_loc"]:
            labels = [0 if t1_ < t2_ else 1 for t1_, t2_ in zip(t1, t2)]
            break
        else:
            raise ValueError(f"Unknown target: {target}")

        labels.append(label_idx)

    return np.array(labels)


def entropy(p, axis=-1, eps=1e-12):
    p = jnp.clip(p, eps, 1.0)
    return -jnp.sum(p * jnp.log(p), axis=axis)


def downstream_evaluation(
    foundation_model,
    foundation_model_params,
    downstream_datasets,
    downstream_train_steps,
    downstream_eval_interval,
    downstream_eval_steps,
    batch_size,
    base_lr,
    grad_accumulation_steps,
    main_training_step,
    full_eval,
    rng_key,
):
    if not full_eval:
        NUM_FRAMES = [4]
    else:
        NUM_FRAMES = [1, 2, 4, 8]

    metrics = {}
    results_csv = []
    for num_frames in NUM_FRAMES:
        if f"train_{num_frames}frames" not in downstream_datasets:
            raise ValueError(f"Missing train dataset for {num_frames} frames.")
        if f"test_{num_frames}frames" not in downstream_datasets:
            raise ValueError(f"Missing test dataset for {num_frames} frames.")

        m, r = downstream_evaluation_(
            foundation_model,
            foundation_model_params,
            downstream_datasets[f"train_{num_frames}frames"],
            downstream_datasets[f"test_{num_frames}frames"],
            downstream_train_steps,
            downstream_eval_interval,
            downstream_eval_steps,
            batch_size,
            base_lr,
            grad_accumulation_steps,
            main_training_step,
            full_eval,
            rng_key,
        )
        metrics.update(m)
        for row in r:
            row.update({"num_frames": num_frames})
        results_csv.extend(r)

    return metrics, results_csv


def downstream_evaluation_(
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
    full_eval,
    rng_key,
):
    if not full_eval:
        TARGETS = ["cell_type", "treatment", "order_diff_loc"]
        FEATURE_EXTRACTION = ["all"]
    else:
        TARGETS = ["cell_type", "treatment", "cell_type_treatment", "order_same_loc", "order_diff_loc"]
        FEATURE_EXTRACTION = ["all", "rand", "1ch"]

    TASKS = [(target, feat) for target in TARGETS for feat in FEATURE_EXTRACTION]

    downstream_models_dict = defaultdict(dict)
    downstream_params_dict = defaultdict(dict)
    optimizers_dict = defaultdict(dict)
    opt_states_dict = defaultdict(dict)
    class_to_index_dict = defaultdict(dict)
    num_classes_dict = defaultdict(dict)
    for target, feat in TASKS:
        classes_train = find_classes(downstream_train_dataset, target)
        classes_test = find_classes(downstream_test_dataset, target)

        if set(classes_train) != set(classes_test):
            print(f"Warning: Classes in train and test datasets do not match for target '{target}'.")
            print(f"Train classes: {classes_train}")
            print(f"Test classes: {classes_test}")

        class_to_index = {cls: idx for idx, cls in enumerate(classes_train)}
        num_classes = len(classes_train)

        print(f"Downstream evaluation for target: {target}")
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {classes_train}")

        downstream_model = AttentionPoolingClassifier(num_classes) if target not in ["order_same_loc", "order_diff_loc"] else AttentionPoolingOrderClassifier()

        latent_emb_dim = foundation_model.latent_emb_dim
        dummy_input1 = jnp.ones((1, 1, latent_emb_dim))
        init_key, rng_key = jax.random.split(rng_key)
        if target not in ["order_same_loc", "order_diff_loc"]:
            params = downstream_model.init(init_key, dummy_input1)["params"]
        else:
            dummy_input2 = jnp.ones((1, 1, latent_emb_dim))
            params = downstream_model.init(init_key, dummy_input1, dummy_input2)["params"]

        optimizer = get_optimizer(
            base_lr=base_lr, 
            warmup_steps=int(0.05 * downstream_train_steps), 
            total_steps=downstream_train_steps, 
            grad_accumulation_steps=grad_accumulation_steps
        )
        opt_state = optimizer.init(params)

        downstream_models_dict[target][feat] = downstream_model
        downstream_params_dict[target][feat] = params
        optimizers_dict[target][feat] = optimizer
        opt_states_dict[target][feat] = opt_state
        class_to_index_dict[target][feat] = class_to_index
        num_classes_dict[target][feat] = num_classes

    @jax.jit
    def extract_features(src_frames, channel_inds, rng_key):
        model_key1, model_key2, token_sample_key, channel_sample_key = jax.random.split(rng_key, 4)
        features = foundation_model.apply(
            {"params": foundation_model_params}, 
            src_frames, 
            channel_inds, 
            method=foundation_model.encode,
            rngs={"default": model_key1}
        )
        features_all = features.reshape(features.shape[0], -1, features.shape[-1])
        B, N, D = features_all.shape
        rand_inds_token = jax.random.choice(token_sample_key, N, shape=(B,), replace=True)
        features_rand = features_all[jnp.arange(B), rand_inds_token, :]
        features_rand = features_rand.reshape(B, 1, D)

        rand_inds_ch = jax.random.choice(channel_sample_key, channel_inds.shape[-1], shape=(B,), replace=True)
        channel_inds_1ch = channel_inds[jnp.arange(B), rand_inds_ch]
        channel_inds_1ch = channel_inds_1ch.reshape(B, 1)
        features_1ch = foundation_model.apply(
            {"params": foundation_model_params}, 
            src_frames, 
            channel_inds_1ch, 
            method=foundation_model.encode,
            rngs={"default": model_key2}
        )
        features_1ch = features_1ch.reshape(features_1ch.shape[0], -1, features_1ch.shape[-1])

        return {
            "all": features_all,
            "rand": features_rand,
            "1ch": features_1ch,
        }

    def extract_features_from_batch(batch, rng_key):
        clips = batch["clip"]
        clips_same_loc_t2 = batch["clip_same_loc_t2"]
        clips_diff_loc_t2 = batch["clip_diff_loc_t2"]
        channel_inds = batch["channel_inds"]

        clips = np.transpose(clips, (0, 1, 3, 4, 2))
        clips_same_loc_t2 = np.transpose(clips_same_loc_t2, (0, 1, 3, 4, 2))
        clips_diff_loc_t2 = np.transpose(clips_diff_loc_t2, (0, 1, 3, 4, 2))

        rng_key1, rng_key2, rng_key3 = jax.random.split(rng_key, 3)
        features_dict = {}
        features_dict["clip"] = extract_features(clips, channel_inds, rng_key1)
        features_dict["clip_same_loc_t2"] = extract_features(clips_same_loc_t2, channel_inds, rng_key2)
        features_dict["clip_diff_loc_t2"] = extract_features(clips_diff_loc_t2, channel_inds, rng_key3)

        return features_dict
    
    def make_training_step(target, feat):
        downstream_model = downstream_models_dict[target][feat]
        optimizer = optimizers_dict[target][feat]

        @jax.jit
        def training_step(
            params,
            opt_state,
            features,
            features_same_loc_t2,
            features_diff_loc_t2,
            labels,
        ):
            def loss_fn(params_):
                output = downstream_model.apply({"params": params_}, features)
                logits = output["logits"]
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                return loss

            def loss_fn_order_same_loc(params_):
                output = downstream_model.apply({"params": params_}, features,features_same_loc_t2)
                logits = output["logits"]
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                return loss

            def loss_fn_order_diff_loc(params_):
                output = downstream_model.apply({"params": params_}, features,features_diff_loc_t2)
                logits = output["logits"]
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
                return loss


            grad_fn = jax.value_and_grad(loss_fn)
            grad_fn_order_same_loc = jax.value_and_grad(loss_fn_order_same_loc)
            grad_fn_order_diff_loc = jax.value_and_grad(loss_fn_order_diff_loc)

            if target not in ["order_same_loc", "order_diff_loc"]:
                loss, grads = grad_fn(params)
            elif target == "order_same_loc":
                loss, grads = grad_fn_order_same_loc(params)
            elif target == "order_diff_loc":
                loss, grads = grad_fn_order_diff_loc(params)

            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)

            return new_params, new_opt_state, loss

        return training_step

    def make_predict(target, feat):
        downstream_model = downstream_models_dict[target][feat]

        @jax.jit
        def predict(
            params, 
            features,
            features_same_loc_t2,
            features_diff_loc_t2,
        ):
            if target not in ["order_same_loc", "order_diff_loc"]:
                output = downstream_model.apply({"params": params}, features)
                attention_weights = output["attention_weights"]
                attention_entropy = entropy(attention_weights, axis=-1)
            elif target == "order_same_loc":
                output = downstream_model.apply({"params": params}, features, features_same_loc_t2)
                attention_weights1 = output["attention_weights1"]
                attention_weights2 = output["attention_weights2"]
                attention_entropy = (entropy(attention_weights1, axis=-1) + entropy(attention_weights2, axis=-1)) / 2
            elif target == "order_diff_loc":
                output = downstream_model.apply({"params": params}, features, features_diff_loc_t2)
                attention_weights1 = output["attention_weights1"]
                attention_weights2 = output["attention_weights2"]
                attention_entropy = (entropy(attention_weights1, axis=-1) + entropy(attention_weights2, axis=-1)) / 2

            return output["logits"], attention_entropy

        return predict
    
    training_funcs = defaultdict(dict)
    predict_funcs = defaultdict(dict)
    for target, feat in TASKS:
        training_funcs[target][feat] = make_training_step(target, feat)
        predict_funcs[target][feat] = make_predict(target, feat)

        wandb.define_metric(
            f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_train_loss", 
            step_metric=f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_step"
        )
        wandb.define_metric(
            f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_test_loss", 
            step_metric=f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_step"
        )
        wandb.define_metric(
            f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_test_accuracy", 
            step_metric=f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_step"
        )

    def eval(rng_key, train_step):
        all_predictions = defaultdict(dict)
        all_logits = defaultdict(dict)
        all_labels = defaultdict(dict)
        all_attention_entropy = defaultdict(dict)

        for target, feat in TASKS:
            all_predictions[target][feat] = []
            all_logits[target][feat] = []
            all_labels[target][feat] = []
            all_attention_entropy[target][feat] = []

        eval_step = 1
        while eval_step <= downstream_eval_steps:
            loader = batch_iterator(downstream_test_dataset, batch_size=batch_size)
            for batch in tqdm(loader, desc="Downstream testing"):
                metadata = batch["metadata"]
                t1 = batch["t1"]
                t2 = batch["t2"]

                extract_key, rng_key = jax.random.split(rng_key)
                features_dict = extract_features_from_batch(batch, extract_key)

                for target, feat in TASKS:
                    predict = predict_funcs[target][feat]
                    params = downstream_params_dict[target][feat]

                    logits, attention_entropy = predict(
                        params,
                        features_dict["clip"][feat],
                        features_dict["clip_same_loc_t2"][feat],
                        features_dict["clip_diff_loc_t2"][feat],
                    )

                    labels = get_labels(metadata, target, class_to_index_dict[target][feat], t1, t2)

                    predictions = jnp.argmax(logits, axis=-1)
                    all_predictions[target][feat].extend(np.array(predictions))
                    all_logits[target][feat].extend(np.array(logits))
                    all_labels[target][feat].extend(labels)
                    all_attention_entropy[target][feat].extend(np.array(attention_entropy))
                eval_step += 1
                if eval_step > downstream_eval_steps:
                    break

        metrics = {}
        results_csv = []
        for target, feat in TASKS:
            all_predictions[target][feat] = np.array(all_predictions[target][feat])
            all_logits[target][feat] = np.array(all_logits[target][feat])
            all_labels[target][feat] = np.array(all_labels[target][feat])
            all_attention_entropy[target][feat] = np.array(all_attention_entropy[target][feat])
            accuracy = np.mean(all_predictions[target][feat] == all_labels[target][feat])
            loss = optax.softmax_cross_entropy_with_integer_labels(all_logits[target][feat], all_labels[target][feat]).mean()
            attention_entropy_mean = np.mean(all_attention_entropy[target][feat])

            wandb.log({
                f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_test_loss": loss,
                f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_test_accuracy": accuracy,
                f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_attention_entropy": attention_entropy_mean,
                f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_step": train_step
            })

            metrics[f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_test_accuracy"] = accuracy
            metrics[f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_attention_entropy"] = attention_entropy_mean

            results_csv.append({
                "target": target,
                "feature": feat,
                "accuracy": accuracy,
                "loss": loss,
                "attention_entropy": attention_entropy_mean,
            })

        return metrics, results_csv

    eval_key, rng_key = jax.random.split(rng_key)
    metrics = eval(eval_key, train_step=0)

    train_step = 1
    while train_step <= downstream_train_steps:
        loader = batch_iterator(downstream_train_dataset, batch_size=batch_size)
        for batch in tqdm(loader, desc=f"Downstream training"):
            metadata = batch["metadata"]
            t1 = batch["t1"]
            t2 = batch["t2"]

            extract_key, rng_key = jax.random.split(rng_key)
            features_dict = extract_features_from_batch(batch, extract_key)

            for target, feat in TASKS:
                training_step = training_funcs[target][feat]
                params = downstream_params_dict[target][feat]
                opt_state = opt_states_dict[target][feat]

                labels = get_labels(metadata, target, class_to_index_dict[target][feat], t1, t2)
                params, opt_state, loss = training_step(
                    params,
                    opt_state,
                    features_dict["clip"][feat],
                    features_dict["clip_same_loc_t2"][feat],
                    features_dict["clip_diff_loc_t2"][feat],
                    labels,
                )

                downstream_params_dict[target][feat] = params
                opt_states_dict[target][feat] = opt_state

                wandb.log({
                    f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_train_loss": loss,
                    f"{DOWNSTREAM_SUBFOLDER}/tgt_{target}_feat_{feat}_fd_step_{main_training_step}_step": train_step
                })

            if train_step % downstream_eval_interval == 0:
                eval_key, rng_key = jax.random.split(rng_key)
                metrics, results_csv = eval(eval_key, train_step)  # Override metrics with the latest evaluation results

            train_step += 1
            if train_step > downstream_train_steps:
                break

    return metrics, results_csv
