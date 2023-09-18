import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataset import random_split
from unet import UNet, dice_loss, dice_coeff

import torch.nn.functional as F


class MyUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(MyUNet, self).__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.dconv_down1 = double_conv(in_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


from utils import TrainImageDataset
from torch.utils.data import DataLoader


checkpoint_dir = os.path.join("models")

writer = SummaryWriter("runs/unet_fine_tuning")


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

from tqdm import tqdm
from typing import Union


def evaluate(model, dataloader, device) -> Union[torch.Tensor, float]:
    # Start of validation
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            mask_pred = (F.sigmoid(outputs) > 0.5).float()

            val_loss += dice_coeff(mask_pred, masks, reduce_batch_first=False)

    model.train()
    return val_loss / len(dataloader)


def get_latest_checkpoint(fine_tuning: bool = False):
    checkpoints = os.listdir(checkpoint_dir)
    checkpoints = [c for c in checkpoints if c.endswith(".pth")]
    checkpoints_all = sorted(
        checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    if fine_tuning:
        checkpoints = list(filter(lambda x: "_fine_tuning" in x, checkpoints_all))
    else:
        checkpoints = list(filter(lambda x: "_fine_tuning" not in x, checkpoints_all))

    if len(checkpoints) == 0:
        print(f"Loading latest checkpoint from {checkpoints_all[-1]}")
        return os.path.join(checkpoint_dir, checkpoints_all[-1])
    else:
        print(f"Loading latest checkpoint from {checkpoints[-1]}")
        return os.path.join(checkpoint_dir, checkpoints[-1])


from torchviz import make_dot


def train(
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    val_percent=0.15,
    learning_rate=1e-5,
    epochs=45,
    batch_size=8,
    load_checkpoint=False,
    fine_tuning=False,
    plot_only=False,
):
    if fine_tuning:
        dataset = "data/tmp/flood"
    else:
        dataset = "data/tmp/water"

    dataset = TrainImageDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    num_val = int(val_percent * len(dataset))
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

    start_epoch = 0

    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )

    # Load checkpoint
    if load_checkpoint:
        checkpoint = torch.load(get_latest_checkpoint(fine_tuning))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss}")

    criterion = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    if plot_only:
        for images, _ in train_loader:
            images = images.to(device)
            g = make_dot(
                model(images).to(device),
                params=dict(model.named_parameters()),
                # show_saved=True,
            )
            # print(g)
            g.save("out.dot")
            return

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(
            enumerate(train_loader),
            total=int(len(dataset) / batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        for batch_idx, (images, masks) in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            # Forward
            outputs = model(images)
            loss = criterion(outputs, masks)
            # print(F.sigmoid(outputs.squeeze(1)).shape, masks.squeeze(1).shape)
            # exit()
            loss += dice_loss(
                F.sigmoid(outputs.squeeze(1)), masks.squeeze(1), multiclass=False
            )

            # Backward
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            total_loss += loss.item()

            # Log the scalar values
            writer.add_scalar(
                "training loss", loss.item(), epoch * len(dataloader) + batch_idx
            )

            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

            if batch_idx % 400 == 0:
                eval_res = evaluate(model, val_loader, device)
                scheduler.step(eval_res)
                writer.add_scalar(
                    "evaluation loss", eval_res, epoch * len(dataloader) + batch_idx
                )

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar("average training loss", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": total_loss / len(dataloader),
        }
        torch.save(
            checkpoint,
            os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch{'_fine_tuning' if fine_tuning else ''}_{epoch+1}.pth",
            ),
        )

    writer.close()


if __name__ == "__main__":
    train(load_checkpoint=True, fine_tuning=True, plot_only=True)
