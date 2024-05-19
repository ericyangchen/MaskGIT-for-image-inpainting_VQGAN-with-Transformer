import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from datetime import datetime
import pytz
from torch.utils.tensorboard import SummaryWriter

class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(
            device=args.device
        )
        self.optim, self.scheduler = self.configure_optimizers(args.learning_rate)

        if args.output_path is not None:
            self.checkpoint_path = os.path.join(args.output_path, "ckpt")

        self.writer = SummaryWriter(log_dir=f"{args.output_path}/logs")

    def configure_optimizers(self, learning_rate):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return optimizer, scheduler

    def train_one_epoch(self, epoch, train_loader):
        self.model.train()

        sum_loss = 0.0
        for image in train_loader:
            image = image.to(args.device)

            logits, z_indices = self.model(image)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            sum_loss += loss.item() * image.size(0)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

        epoch_loss = sum_loss / len(train_loader.dataset)
        self.writer.add_scalar("Loss/train", epoch_loss, epoch)

        self.scheduler.step()

    def eval_one_epoch(self, epoch, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for image in val_loader:
                image = image.to(args.device)

                logits, z_indices = self.model(image)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), z_indices.view(-1)
                )
                val_loss += loss.item() * image.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        self.writer.add_scalar("Loss/val", epoch_val_loss, epoch)

    def save_checkpoint(self, epoch):
        checkpoint_path = f"{self.checkpoint_path}/epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)


def create_output_dir(output_path):
    # create output(output_path) directory
    if output_path is not None:
        tz = pytz.timezone("Asia/Taipei")
        now = datetime.now(tz).strftime("%Y%m%d-%H%M")

        output_path = os.path.join(output_path, now)

        # create output directory
        os.makedirs(output_path, exist_ok=True)

        # create ckpt directory
        os.makedirs(f"{output_path}/ckpt", exist_ok=True)

        # create logs directory
        os.makedirs(f"{output_path}/logs", exist_ok=True)

        return output_path

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT")
    # fmt: off
    parser.add_argument("--output-path", type=str, default="outputs", help="Path to output.")
    parser.add_argument("--train_d_path", type=str, default="data/cat_face/train/", help="Training Dataset Path")
    parser.add_argument("--val_d_path", type=str, default="data/cat_face/val/", help="Validation Dataset Path")
    parser.add_argument("--device", type=str, default="cuda", help="Which device the training is on.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for training.")
    parser.add_argument("--partial", type=float, default=1.0, help="Number of epochs to train (default: 50)")
    
    # you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')
    # fmt: on

    args = parser.parse_args()

    args.output_path = create_output_dir(args.output_path)

    # save args
    with open(f"{args.output_path}/args.yml", "w") as f:
        yaml.safe_dump(vars(args), f)

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    for epoch in tqdm(range(args.start_from_epoch + 1, args.epochs + 1)):
        train_transformer.train_one_epoch(epoch, train_loader)
        train_transformer.eval_one_epoch(epoch, val_loader)

        if epoch % args.save_per_epoch == 0:
            train_transformer.save_checkpoint(epoch)
