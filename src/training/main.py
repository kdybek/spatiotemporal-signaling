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

from utils.datasets import N5VideoDataset
from utils.logging import get_exp_name, setup_wandb


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_integer('epochs', 5, 'Number of training epochs.')
flags.DEFINE_integer('eval_interval', 1, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 5, 'Saving interval.')
flags.DEFINE_string('dataset_path', 'dataset.h5', 'Path to the dataset.')


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


def get_random_mask(batch_size, seq_len, mask_ratio=0.75):
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

    dataset = N5VideoDataset(
        h5_path=FLAGS.dataset_path,
        num_frames=16,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = VideoMAEConfig(
        num_frames=16,
        image_size=256,
        num_channels=1,
    )
    seq_len = get_seq_len(config)

    model = VideoMAEForPreTraining(config)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    epochs = FLAGS.epochs

    model.train()

    for epoch in range(epochs):
        for videos in tqdm(dataloader):
            videos = videos.to(device)  # (B,T,C,H,W)

            # HuggingFace expects (B,T,C,H,W)
            mask = get_random_mask(videos.size(0), seq_len).to(device)
            outputs = model(pixel_values=videos, bool_masked_pos=mask)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({'train_loss': loss.item()})


if __name__ == '__main__':
    app.run(main)
