import torch
import numpy as np
from transformers import VideoMAEForPreTraining, VideoMAEConfig
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
from absl import app, flags
import random
import os
import wandb

from utils.datasets import ZarrVideoDataset
from utils.logging import get_exp_name, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_integer('epochs', 5, 'Number of training epochs.')
flags.DEFINE_integer('eval_interval', 1, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 5, 'Saving interval.')
flags.DEFINE_string('train_dataset_path', 'train.zarr', 'Path to the train dataset.')
flags.DEFINE_string('test_dataset_path', 'test.zarr', 'Path to the test dataset.')
flags.DEFINE_string('save_dir', 'checkpoints', 'Directory to save model checkpoints.')

flags.DEFINE_integer('batch_size', 4, 'Batch size for training and evaluation.')
flags.DEFINE_float('mask_ratio', 0.75, 'Ratio of patches to mask during training.')


def evaluate(model, dataloader, device, config, mask_ratio):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for videos in tqdm(dataloader):
            videos = videos.to(device)

            mask = get_random_mask(videos.size(0), get_seq_len(config), mask_ratio).to(device)
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
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = torch.randperm(seq_len)[:num_masked]
        mask[i, masked_indices] = True
    return mask


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='Spaciotemporal Signaling',
                group=FLAGS.run_group, name=exp_name)
    set_seed(FLAGS.seed)

    os.makedirs(FLAGS.save_dir, exist_ok=True)

    train_dataset = ZarrVideoDataset(
        zarr_path=FLAGS.train_dataset_path,
        dataset_key="Data",
    )

    test_dataset = ZarrVideoDataset(
        zarr_path=FLAGS.test_dataset_path,
        dataset_key="Data",
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
    )
    seq_len = get_seq_len(config)

    model = VideoMAEForPreTraining(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    epochs = FLAGS.epochs

    mask_ratio = FLAGS.mask_ratio

    eval_loss = evaluate(model, test_dataloader, device, config, mask_ratio)
    wandb.log({'eval_loss': eval_loss})

    for epoch in range(1, epochs + 1):
        model.train()
        for videos in tqdm(train_dataloader):

            videos = videos.to(device)  # (B,T,C,H,W)

            mask = get_random_mask(videos.size(0), seq_len, mask_ratio).to(device)
            outputs = model(pixel_values=videos, bool_masked_pos=mask)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})

        if epoch % FLAGS.eval_interval == 0:
            eval_loss = evaluate(model, test_dataloader, device, config, mask_ratio)
            wandb.log({'eval_loss': eval_loss})

        if epoch % FLAGS.save_interval == 0:
            model_save_path = os.path.join(FLAGS.save_dir, f'model_epoch_{epoch}.pt')
            torch.save(model.state_dict(), model_save_path)
            wandb.save(model_save_path)


if __name__ == '__main__':
    app.run(main)
