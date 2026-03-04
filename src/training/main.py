import zarr
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from absl import app, flags

from utils.datasets import N5VideoDataset
from utils.logging import get_exp_name, setup_wandb


flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')

flags.DEFINE_integer('epochs', 5, 'Number of training epochs.')
flags.DEFINE_integer('eval_interval', 1, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 5, 'Saving interval.')


def set_seed(seed: int = 42):
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


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='Spaciotemporal Signaling', group=FLAGS.run_group, name=exp_name)
    set_seed(FLAGS.seed)

    processor = VideoMAEImageProcessor.from_pretrained(
        "MCG-NJU/videomae-base"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    dataset = N5VideoDataset(
        n5_path="/mnt/imaging.data/zppmimuw/dataset.n5",
        num_frames=16,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VideoMAEForPreTraining.from_pretrained(
        "MCG-NJU/videomae-base"
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    epochs = FLAGS.epochs

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for videos in tqdm(dataloader):
            videos = videos.to(device)  # (B,T,C,H,W)

            # HuggingFace expects (B,T,C,H,W)
            outputs = model(pixel_values=videos)

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


if __name__ == '__main__':
    app.run(main)
