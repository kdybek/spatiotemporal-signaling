import torch
import numpy as np
from transformers import VideoMAEForPreTraining, VideoMAEConfig
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from absl import app, flags
import random
import os
import wandb

from utils.datasets import ZarrVideoDataset
from utils.logging import get_exp_name, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_integer('steps', 600_000, 'Number of training steps.')
flags.DEFINE_integer('eval_interval', 100_000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 300_000, 'Saving interval.')
flags.DEFINE_string('train_dataset_path', 'train.zarr',
                    'Path to the train dataset.')
flags.DEFINE_string('test_dataset_path', 'test.zarr',
                    'Path to the test dataset.')
flags.DEFINE_string('save_dir', 'checkpoints',
                    'Directory to save model checkpoints.')

flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate for the optimizer.')
flags.DEFINE_integer(
    'batch_size', 4, 'Batch size for training and evaluation.')
flags.DEFINE_float('train_split', 0.8, 'Proportion of data to use for training (rest is for validation).')
flags.DEFINE_float('mask_ratio', 0.75,
                   'Ratio of patches to mask during training.')
flags.DEFINE_integer('tubelet_size', 2, 'Size of tubelets (temporal dimension).')
flags.DEFINE_integer('patch_size', 16, 'Size of spatial patches.')
flags.DEFINE_integer('clip_size', 224, 'Height and width of input images.')
flags.DEFINE_integer('clip_frames', 16, 'Number of frames in each video clip.')
flags.DEFINE_integer('acq_freq', 30, 'Acquisition frequency (in minutes) for sampling video clips.')
flags.DEFINE_string('channel_names', 'Ch_H2B Ch_ERK-KTR', 'Space-separated list of channel names to use from the videos.')
flags.DEFINE_string('transforms', 'arcsinh percentile_norm',
                    'Space-separated list of transforms to apply to the videos. Supported: percentile_norm, arcsinh, log1p.')
flags.DEFINE_float('arcsinh_cofactor', 5.0,
                   'Cofactor for arcsinh transform. Only used if "arcsinh" is in the transforms list.')
flags.DEFINE_bool('augment', True, 'Whether to apply data augmentation (random flips) during training.')


def evaluate(model, dataloader, device, config, mask_ratio):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for videos in tqdm(dataloader):
            videos = videos.to(device)

            mask = get_random_mask(videos.size(
                0), get_seq_len(config), mask_ratio).to(device)
            outputs = model(pixel_values=videos, bool_masked_pos=mask)

            total_loss += outputs.loss.item() * videos.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


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


def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='Spaciotemporal Signaling',
                group=FLAGS.run_group, name=exp_name)
    set_seed(FLAGS.seed)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    dataset = ZarrVideoDataset(
        zarr_path=FLAGS.train_dataset_path,
        clip_frames=FLAGS.clip_frames,
        clip_size=FLAGS.clip_size,
        acq_freq=FLAGS.acq_freq,
        transform_names=FLAGS.transforms,
        arcsinh_cofactor=FLAGS.arcsinh_cofactor,
        augment=FLAGS.augment,
    )

    train_size = int(len(dataset) * FLAGS.train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

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
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T, C, H, W = train_dataset[0].shape
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

    mask_ratio = FLAGS.mask_ratio

    eval_loss = evaluate(model, test_dataloader, device, config, mask_ratio)
    wandb.log({'eval_loss': eval_loss}, step=0)

    step = 1
    metrics = {}
    while step < FLAGS.steps:
        for videos in tqdm(train_dataloader):

            videos = videos.to(device)  # (B,T,C,H,W)

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
                eval_loss = evaluate(model, test_dataloader,
                                     device, config, mask_ratio)
                metrics['eval_loss'] = eval_loss

            if step % FLAGS.save_interval == 0:
                model_save_path = os.path.join(
                    FLAGS.save_dir, f'model_step_{step}.pt')
                torch.save(model.state_dict(), model_save_path)
                wandb.save(model_save_path)

            wandb.log(metrics, step=step)
            metrics = {}

            step += 1
            if step >= FLAGS.steps:
                break


if __name__ == '__main__':
    app.run(main)
