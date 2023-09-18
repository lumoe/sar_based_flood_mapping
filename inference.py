import os

import torch
from torch.utils.data import DataLoader

from unet import UNet, dice_loss, dice_coeff
from utils import TrainImageDataset, TestImageDataset

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import numpy as np

import pickle


def visualise_overlap(image, tuw_result, result, reference, mask, chip_name=""):
    # Visualize the reference and Sentinel images side by side and on top of each other with 0.5 opacity
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    ax1, ax3, ax6, ax2, ax4, ax5 = axes.flatten()

    ax1.imshow(image, cmap="gray")
    ax1.set_title("Sentinel SAR Image")

    ax2.imshow(tuw_result)
    ax2.set_title("TUW Results")

    ax3.imshow(result, cmap="gray")
    ax3.set_title("Raw Model Output")

    ax4.imshow(result > 0.96)

    # ax3.imshow(result, cmap="gray")
    # ax3.imshow(reference_data, cmap="gray", alpha=0.5)
    # ax3.imshow(sentinel_data, alpha=0.5)
    ax4.set_title("Optimal Threshold (0.96)")

    ax5.imshow(reference)
    ax5.set_title("Reference Image")

    ax6.imshow(mask)
    ax6.set_title("Mask")

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axis("off")

    plt.suptitle(chip_name)
    plt.tight_layout()
    plt.show()


def get_latest_checkpoint(fine_tuning: bool = False):
    checkpoint_dir = os.path.join("models")

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)

    checkpoint = torch.load(get_latest_checkpoint(True))
    model.load_state_dict(checkpoint["model_state_dict"])
    start_epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Loaded checkpoint from epoch {start_epoch} with loss {loss}")

    model.eval()

    dataset = TestImageDataset("data/tmp/test")
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Initialize variables to store true positive rates, false positive rates, and AUC
    tpr_list = []
    fpr_list = []
    roc_auc_list = []

    # Initialize an array to store Dice coefficients for various thresholds
    avg_dice_coefficients = []

    # Define thresholds
    thresholds = np.linspace(0, 1, 50)
    thresholds = [0.9591836734693877]

    # Initialize variables to store averaged true positive rates and false positive rates
    avg_tpr_list = np.zeros_like(thresholds)
    avg_fpr_list = np.zeros_like(thresholds)

    for threshold in thresholds:
        dice_coefficients = []
        progress_bar = tqdm(
            enumerate(test_loader),
            total=int(len(dataset)),
        )
        progress_bar.set_description(f"Threshold: {threshold:.2f}")
        for idx, (image, reference, mask, tuw_results, filename) in progress_bar:
            res = model(image.to(device))

            visualise_overlap(
                image.squeeze().cpu().numpy(),
                tuw_results.squeeze().cpu().numpy(),
                res.squeeze().cpu().detach().numpy(),
                reference.squeeze().cpu().numpy(),
                mask.squeeze().cpu().numpy(),
                chip_name=filename,
            )

            dc = dice_coeff(
                tuw_results.cpu(), res.cpu() > threshold, reduce_batch_first=False
            )
            dice_coefficients.append(dc)

            # print(diff)

            # for threshold in thresholds:
            #     # Compute Dice coefficient for each threshold
            #     dc = dice_coeff(
            #         tuw_results.cpu(), res.cpu() > threshold, reduce_batch_first=False
            #     )
            #     dice_coefficients.append(dc)

            # print(f"Threshold: {thresh}, Dice Coefficient: {dice_coeff_val}")

            # if idx == 10:
            #     break

            # progress_bar.set_description(f"Dice Coefficient: {diff:.4f}")
        avg_dice_coeff = np.mean(dice_coefficients)
        avg_dice_coefficients.append(avg_dice_coeff)

    # Find the optimal threshold based on the maximum average Dice coefficient
    optimal_threshold = thresholds[np.argmax(avg_dice_coefficients)]
    print(f"Optimal Threshold: {optimal_threshold}")

    # Plot ROC curve using average Dice coefficients
    plt.figure()
    plt.plot(thresholds, avg_dice_coefficients, label="Average Dice Coefficient")
    # Add vertical line for optimal threshold with dashed line style and a label
    plt.axvline(
        optimal_threshold,
        linestyle="--",
        label=f"Optimal Threshold ({optimal_threshold:.2f})",
        color="orange",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Average Dice Coefficient")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    # Save the plot as PDF file to notebooks/
    plt.savefig("roc_curve.pdf")

    # Save the data as pickle files
    with open("roc_and_dice_data.pkl", "wb") as f:
        pickle.dump(
            {
                "avg_fpr_list": avg_fpr_list,
                "avg_tpr_list": avg_tpr_list,
                "roc_auc_list": roc_auc_list,
                "dice_coeffs": avg_dice_coefficients,
                "thresholds": thresholds,
            },
            f,
        )


if __name__ == "__main__":
    main()
