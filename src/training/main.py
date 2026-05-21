import torch
import numpy as np
from transformers import VideoMAEForPreTraining, VideoMAEConfig
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from absl import app, flags
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
import os
import wandb
import json

from utils.datasets import create_train_test_datasets, percentile_norm, butterworth_filter
from utils.logging import get_exp_name, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_integer('steps', 300_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 50_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 50_000, 'Saving interval.')
flags.DEFINE_integer('mask_curriculum_steps', 0,
                     'Number of steps over which to linearly increase the mask ratio from 0.2 to the final value.')
flags.DEFINE_string('dataset_path', 'toy_dataset.zarr',
                    'Path to the train dataset.')
flags.DEFINE_string('save_dir', 'checkpoints',
                    'Directory to save model checkpoints.')

flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for the optimizer.')
flags.DEFINE_integer(
    'batch_size', 16, 'Batch size for training and evaluation.')
flags.DEFINE_float('train_split', 0.8,
                   'Proportion of data to use for training (rest is for validation).')
flags.DEFINE_float('mask_ratio', 0.75,
                   'Ratio of patches to mask during training.')
flags.DEFINE_integer('tubelet_size', 2, 'Size of tubelets (temporal dimension).')
flags.DEFINE_integer('patch_size', 16, 'Size of spatial patches.')
flags.DEFINE_integer('clip_size', 224, 'Height and width of input images.')
flags.DEFINE_integer('clip_frames', 16, 'Number of frames in each video clip.')
flags.DEFINE_integer(
    'acq_freq', 30, 'Acquisition frequency (in minutes) for sampling video clips.')
flags.DEFINE_string('channel_names', 'Ch_ERK-KTR',
                    'Space-separated list of channel names to use from the videos.')
flags.DEFINE_string('transforms', 'arcsinh butterworth percentile_norm',
                    'Space-separated list of transforms to apply to the videos. Supported: percentile_norm, arcsinh, log1p, butterworth.')
flags.DEFINE_float('arcsinh_cofactor', 5.0,
                   'Cofactor for arcsinh transform.')
flags.DEFINE_float('butterworth_cutoff', 0.2,
                   'Cutoff frequency for Butterworth low-pass filter.')
flags.DEFINE_integer('butterworth_order', 2,
                     'Order of Butterworth low-pass filter.')
flags.DEFINE_boolean('per_frame_butterworth', False,
                     'Whether to apply Butterworth filter independently to each frame (instead of across time).')

flags.DEFINE_integer(
    'eval_traj_len', 8, 'Number of overlapping clips for the trajectory experimens.')
flags.DEFINE_integer(
    'eval_traj_stride', 4, 'Stride between overlapping clips for the trajectory experimens.')


def patchify(videos, tubelet_size, patch_size):
    B, T, C, H, W = videos.shape
    assert T % tubelet_size == 0, "Number of frames must be divisible by tubelet size."
    assert H % patch_size == 0 and W % patch_size == 0, "Height and width must be divisible by patch size."

    num_tubelets = T // tubelet_size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    videos = videos.view(B, num_tubelets, tubelet_size, C,
                         num_patches_h, patch_size, num_patches_w, patch_size)
    videos = videos.permute(0, 1, 4, 6, 2, 3, 5, 7).contiguous()
    videos = videos.view(B, num_tubelets * num_patches_h *
                         num_patches_w, tubelet_size * C * patch_size * patch_size)

    return videos


def unpatchify(patches, tubelet_size, patch_size, num_frames, image_size):
    B, N, D = patches.shape
    num_tubelets = num_frames // tubelet_size
    num_patches_h = image_size // patch_size
    num_patches_w = image_size // patch_size

    videos = patches.view(B, num_tubelets, num_patches_h,
                          num_patches_w, tubelet_size, -1, patch_size, patch_size)
    videos = videos.permute(0, 1, 4, 5, 2, 6, 3, 7).contiguous()
    videos = videos.view(B, num_frames, -1, image_size, image_size)

    return videos


def reconstruct_videos_from_patches(videos, reconstructed_patches, mask, tubelet_size, patch_size, num_frames, image_size):
    B, T, C, H, W = videos.shape
    patches = patchify(videos, tubelet_size, patch_size).clone()
    for i in range(B):
        patches[i][mask[i]] = reconstructed_patches[i]
    reconstructed_videos = unpatchify(
        patches, tubelet_size, patch_size, num_frames, image_size)

    return reconstructed_videos


def split_video_batch_into_overlapping_clips(videos, clip_frames, traj_len, traj_stride):
    B, T, C, H, W = videos.shape
    clips = []
    for start in range(0, T - clip_frames + 1, traj_stride):
        clip = videos[:, start:start+clip_frames, ...]
        clips.append(clip)
        if len(clips) == traj_len:
            break

    return clips


def evaluate_masked(model, dataloader, device, config, mask_ratio):
    model.eval()
    total_loss = 0.0
    first_visualization_done = False
    metrics = {}

    with torch.no_grad():
        for videos, _ in tqdm(dataloader):
            # Ensure videos have the expected number of frames
            videos = videos[:, :config.num_frames, ...]
            videos = videos.to(device)

            mask = get_random_mask(videos.size(
                0), get_seq_len(config), mask_ratio).to(device)
            outputs = model(pixel_values=videos, bool_masked_pos=mask)

            if not first_visualization_done:
                rec_videos = reconstruct_videos_from_patches(
                    videos.cpu(),
                    outputs.logits.cpu(),
                    mask.cpu(),
                    config.tubelet_size,
                    config.patch_size,
                    config.num_frames,
                    config.image_size
                ).numpy()

                original_video = videos[0].cpu().numpy()
                rec_video = rec_videos[0]
                original_video = np.clip(original_video, 0, 1)
                rec_video = np.clip(rec_video, 0, 1)
                original_video = (original_video.clip(0, 1) * 255).astype(np.uint8)
                rec_video = (rec_video * 255).astype(np.uint8)

                num_channels = original_video.shape[1]
                for ch in range(num_channels):
                    og_video_ch = np.repeat(original_video[:, ch:ch+1, ...], 3, axis=1)
                    rec_video_ch = np.repeat(rec_video[:, ch:ch+1, ...], 3, axis=1)
                    metrics[f'original_channel_{ch}'] = wandb.Video(
                        og_video_ch, fps=4, format="mp4")
                    metrics[f'reconstructed_channel_{ch}'] = wandb.Video(
                        rec_video_ch, fps=4, format="mp4")

                first_visualization_done = True

            total_loss += outputs.loss.item() * videos.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    metrics['eval_loss'] = avg_loss

    return metrics


def extract_videomae_latents(model, config, pixel_values):
    model.eval()

    with torch.no_grad():
        mask = torch.zeros(pixel_values.size(0), get_seq_len(config),
                           dtype=torch.bool, device=pixel_values.device)
        emb = model.videomae.embeddings(pixel_values, bool_masked_pos=mask)
        latents = model.videomae.encoder(emb)[0]  # (B, seq_len, D)

    return latents


def create_tsne_plot(latent_trajs, labels):
    B, traj_len, D = latent_trajs.shape

    assert len(labels) == B, "Number of labels must match the number of trajectories."

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
    )
    latent_trajs_2d = tsne.fit_transform(latent_trajs[:, 0, :].numpy())

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        latent_trajs_2d[:, 0],
        latent_trajs_2d[:, 1],
        c=y,
        s=10,
    )

    ax.set_title("t-SNE Embeddings")

    return wandb.Image(fig)


def get_traj_stats(latent_trajs):
    stats = {
        'dist_means': [],
        'dist_stds': [],
        'straightness': [],
    }

    for traj in latent_trajs:
        dists = torch.norm(traj[1:] - traj[:-1], dim=1)
        stats['dist_means'].append(dists.mean().item())
        stats['dist_stds'].append(dists.std().item())
        first_last_dist = torch.norm(traj[-1] - traj[0]).item()
        dist_sum = dists.sum().item()
        stats['straightness'].append(first_last_dist / (dist_sum + 1e-8))

    return stats


def create_traj_plots(latent_trajs, labels):
    stats = get_traj_stats(latent_trajs)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(stats['dist_means'], stats['dist_stds'], c=y)
    axes[0].set_xlabel('Mean Step Distance')
    axes[0].set_ylabel('Std of Step Distances')
    axes[0].set_title('Mean vs Std of Step Distances')

    axes[1].scatter(stats['dist_means'], stats['straightness'], c=y)
    axes[1].set_xlabel('Mean Step Distance')
    axes[1].set_ylabel('Straightness')
    axes[1].set_title('Mean Step Distance vs Straightness')

    axes[2].scatter(stats['dist_stds'], stats['straightness'], c=y)
    axes[2].set_xlabel('Std of Step Distances')
    axes[2].set_ylabel('Straightness')
    axes[2].set_title('Std of Step Distances vs Straightness')

    return wandb.Image(fig)


def evaluate_cluster(model, dataloader, device, config, traj_len, traj_stride):
    model.eval()
    latent_trajs = []
    labels = []

    MAX_TRAJ = 512  # To limit memory usage during t-SNE evaluation

    with torch.no_grad():
        for videos, exp_name in tqdm(dataloader):
            clips = split_video_batch_into_overlapping_clips(
                videos, config.num_frames, traj_len, traj_stride)

            traj = []
            for clip in clips:
                clip = clip.to(device)
                latent = extract_videomae_latents(
                    model, config, clip)  # (B, seq_len, D)
                latent = latent.mean(dim=1)  # (B, D)
                assert len(latent.shape) == 2, "Expected features to be of shape (B, D)"
                traj.append(latent.cpu().unsqueeze(1))  # (B, 1, D)

            traj = torch.cat(traj, dim=1)  # (B, traj_len, D)
            latent_trajs.append(traj)
            labels.extend(exp_name)

            if len(labels) >= MAX_TRAJ:
                break

    latent_trajs = torch.cat(latent_trajs, dim=0)  # (N, traj_len, D)

    tsne_plot = create_tsne_plot(latent_trajs, labels)
    traj_stats_plot = create_traj_plots(latent_trajs, labels)

    metrics = {
        'tsne_plot': tsne_plot,
        'traj_stats_plot': traj_stats_plot,
    }

    return metrics


def get_mask_ratio(step, curriculum_steps, target_mask_ratio):
    if step >= curriculum_steps:
        return target_mask_ratio
    else:
        return 0.2 + (target_mask_ratio - 0.2) * (step / curriculum_steps)


def set_seed(seed):
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_seq_len(config):
    return (config.num_frames // config.tubelet_size) * (config.image_size // config.patch_size) ** 2


def get_random_mask(batch_size, seq_len, mask_ratio):
    num_masked = int(seq_len * mask_ratio)

    noise = torch.rand(batch_size, seq_len)
    ids_shuffle = torch.argsort(noise, dim=1)

    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask.scatter_(1, ids_shuffle[:, :num_masked], True)
    mask.scatter_(1, ids_shuffle[:, num_masked:], False)

    return mask


class TransformPipeline:
    def __init__(self, transform_names_list, arcsinh_cofactor, butterworth_cutoff, butterworth_order, per_frame_butterworth):
        self.transform_names_list = transform_names_list
        self.arcsinh_cofactor = arcsinh_cofactor
        self.butterworth_cutoff = butterworth_cutoff
        self.butterworth_order = butterworth_order
        self.per_frame_butterworth = per_frame_butterworth

    def __call__(self, video):
        for transform_name in self.transform_names_list:
            if transform_name == 'percentile_norm':
                video = percentile_norm(video)
            elif transform_name == 'arcsinh':
                video = np.arcsinh(video / self.arcsinh_cofactor)
            elif transform_name == 'log1p':
                video = np.log1p(video)
            elif transform_name == 'butterworth':
                video = butterworth_filter(
                    video, cutoff=self.butterworth_cutoff, order=self.butterworth_order, per_frame=self.per_frame_butterworth)
            else:
                raise ValueError(f"Unsupported transform: {transform_name}")

        return video


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='Spaciotemporal Signaling',
                group=FLAGS.run_group, name=exp_name)
    set_seed(FLAGS.seed)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flags_dict = FLAGS.flag_values_dict()

    with open(os.path.join(FLAGS.save_dir, 'config.json'), 'w') as f:
        json.dump(flags_dict, f, indent=4)

    eval_clip_frames = FLAGS.clip_frames + \
        (FLAGS.eval_traj_len - 1) * FLAGS.eval_traj_stride

    transform_pipeline = TransformPipeline(
        transform_names_list=FLAGS.transforms.split(),
        arcsinh_cofactor=FLAGS.arcsinh_cofactor,
        butterworth_cutoff=FLAGS.butterworth_cutoff,
        butterworth_order=FLAGS.butterworth_order,
        per_frame_butterworth=FLAGS.per_frame_butterworth,
    )

    train_dataset, test_dataset = create_train_test_datasets(
        test_fraction=1 - FLAGS.train_split,
        zarr_path=FLAGS.dataset_path,
        clip_frames_train=FLAGS.clip_frames,
        clip_frames_test=eval_clip_frames,
        clip_size=FLAGS.clip_size,
        acq_freq=FLAGS.acq_freq,
        channel_names_list=FLAGS.channel_names.split(),
        transform_pipeline=transform_pipeline,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,  # For better class balance during eval
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T, C, H, W = train_dataset[0][0].shape
    assert H == W, "Only square images are supported."

    config = VideoMAEConfig(
        num_frames=T,
        image_size=H,
        num_channels=C,
        tubelet_size=FLAGS.tubelet_size,
        patch_size=FLAGS.patch_size,
    )
    seq_len = get_seq_len(config)

    model = VideoMAEForPreTraining(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=FLAGS.learning_rate)

    target_mask_ratio = FLAGS.mask_ratio

    eval_metrics = evaluate_masked(
        model, test_dataloader, device, config, target_mask_ratio)
    latent_metrics = evaluate_cluster(
        model, test_dataloader, device, config, FLAGS.eval_traj_len, FLAGS.eval_traj_stride)
    eval_metrics.update(latent_metrics)
    wandb.log(eval_metrics, step=0)

    step = 1
    metrics = {}
    while step < FLAGS.steps + 1:
        for videos, _ in tqdm(train_dataloader):

            videos = videos.to(device)  # (B,T,C,H,W)

            mask_ratio = get_mask_ratio(
                step, FLAGS.mask_curriculum_steps, target_mask_ratio)
            mask = get_random_mask(videos.size(
                0), seq_len, mask_ratio).to(device)

            model.train()
            outputs = model(pixel_values=videos, bool_masked_pos=mask)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics['train_loss'] = loss.item()

            if step % FLAGS.eval_interval == 0:
                eval_metrics = evaluate_masked(model, test_dataloader,
                                               device, config, target_mask_ratio)
                latent_metrics = evaluate_cluster(
                    model, test_dataloader, device, config, FLAGS.eval_traj_len, FLAGS.eval_traj_stride)

                metrics.update(eval_metrics)
                metrics.update(latent_metrics)

            if step % FLAGS.save_interval == 0:
                model_save_path = os.path.join(
                    FLAGS.save_dir, f'model_step_{step}.pt')
                torch.save(model.state_dict(), model_save_path)
                wandb.save(model_save_path)

            wandb.log(metrics, step=step)
            metrics = {}

            step += 1
            if step > FLAGS.steps:
                break


if __name__ == '__main__':
    app.run(main)
