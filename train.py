import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataset import random_split


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

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

learning_rate = 0.001
epochs = 10
batch_size = 16
checkpoint_dir = os.path.join("models")

writer = SummaryWriter("runs/unet_training")


if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

from tqdm import tqdm


def train():
    dataset = TrainImageDataset("data/tmp/water")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_percent = 0.2
    num_val = int(val_percent * len(dataset))
    num_train = len(dataset) - num_val
    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
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

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Log the scalar values
            writer.add_scalar(
                "training loss", loss.item(), epoch * len(dataloader) + batch_idx
            )

            progress_bar.set_postfix(loss=total_loss / (batch_idx + 1))

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
            checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        )

        # Start of validation
        model.eval()  # Set model to evaluation mode
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
            writer.add_scalar("average validation loss", val_loss, epoch)

    writer.close()


if __name__ == "__main__":
    train()
