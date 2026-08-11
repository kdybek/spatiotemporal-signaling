import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import random
import os
import json
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from utils.model import get_rvm
from utils.dataloader import create_train_val_datasets, TransformPipeline, batch_iterator

SEED = 0
CONFIG_PATH = 'checkpoints/rvm_200K_2Ch/config.json'
CHECKPOINT_PATH = 'checkpoints/rvm_200K_2Ch/checkpoint_20000'
BATCH_SIZE = 16
NUM_SAMPLES = 10000
OUTPUT_DIR = 'latents/rvm_200K_2Ch'
CLIP_FRAMES = 16
DATASET_PATH = '/home/zppmimuw/myscratch/datasets/geminin_drugs_full_vid_3.zarr'

FEATURES_PATH = os.path.join(OUTPUT_DIR, 'features_.npy')
FEATURES_METADATA_PATH = os.path.join(OUTPUT_DIR, 'metadata_.pkl')


def set_seed(seed):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)


def load_checkpoint(checkpointer, checkpoint_path):
    state = checkpointer.restore(os.path.abspath(checkpoint_path))
    return state['params'], state['opt_state'], state['step']


def extract_latents(
    model,
    dataset,
    params,
    rng_key,
):
    @jax.jit
    def forward(sources):
        output = model.apply(
            {"params": params},
            sources,
            state=None,
            method=model.encode,
            rngs={"default": rng_key}
        )

        output = jnp.mean(output, axis=-2)  # Spatial averaging

        return output

    all_features = []
    metadata = []
    num_samples = 0
    while num_samples < NUM_SAMPLES:
        loader = batch_iterator(dataset, batch_size=BATCH_SIZE, aux=True)
        for batch in loader:
            clips = batch["clips"]
            meta = batch["metadata"]
            clips = np.transpose(clips, (0, 1, 3, 4, 2))

            features = forward(clips)

            all_features.extend(np.array(features))
            metadata.extend(meta)

            num_samples += len(clips)

            if num_samples >= NUM_SAMPLES:
                break

            print(f"Extracted {num_samples} samples...")

    features = np.array(all_features)

    return features, metadata


def probe(features, labels, label_name, time_horizon):
    X = features
    y = np.array(labels)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    print(f"Time horizon: {time_horizon}, label: {label_name}, accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


def plot_tsne(features, labels, label_name, time_horizon, random_state=42):
    X = np.asarray(features)

    # Encode string labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )

    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 7))

    for class_id, class_name in enumerate(le.classes_):
        mask = y == class_id

        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            label=class_name,
            alpha=0.7,
            s=3
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(f"t-SNE — {label_name}, first {time_horizon} positions")
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tsne_{label_name.replace(' ', '_')}_time_horizon_{time_horizon}.png", dpi=300)
    plt.close()


def collect_features():
    rng_key = jax.random.PRNGKey(SEED)

    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = get_rvm(
        num_channels=len(config['channel_names'].split()),
        masking_ratio=config['masking_ratio'],
        variant=config['rvm_variant'],
    )

    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    model_params, _, step = load_checkpoint(checkpointer, CHECKPOINT_PATH)

    transform_pipeline = TransformPipeline(
        transform_names_list=config['transforms'].split(),
        arcsinh_cofactor=config['arcsinh_cofactor'],
        butterworth_cutoff=config['butterworth_cutoff'],
        butterworth_order=config['butterworth_order'],
        per_frame_butterworth=config['per_frame_butterworth'],
    )

    _, val_dataset = create_train_val_datasets(
        DATASET_PATH,
        CLIP_FRAMES,
        config['clip_size'],
        config['acq_freq'],
        config['channel_names'].split(),
        transform_pipeline,
        random_crop_val=True
    )

    latent_rng_key, rng_key = jax.random.split(rng_key)
    features, metadata = extract_latents(
        model,
        val_dataset,
        model_params,
        latent_rng_key
    )

    np.save(FEATURES_PATH, features)
    with open(FEATURES_METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)


def main():
    if not os.path.exists(FEATURES_PATH) or not os.path.exists(FEATURES_METADATA_PATH):
        collect_features()
    else:
        features = np.load(FEATURES_PATH)
        with open(FEATURES_METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)

    cell_types = [m['Cell_type'] for m in metadata]
    inhibitors = [m['Inhibitor'] for m in metadata]

    type_inhibitor = [cell_types[i] + inhibitors[i] for i in range(len(cell_types))]

    for time_horizon in [1, 2, 4, 8, 16]:
        pref_pool_feat = features[:, :time_horizon, :].mean(axis=1)

        probe(pref_pool_feat, cell_types, "cell type", time_horizon)
        probe(pref_pool_feat, inhibitors, "inhibitor", time_horizon)
        probe(pref_pool_feat, type_inhibitor, "cell type + inhibitor", time_horizon)


if __name__ == "__main__":
    main()
