import torch
import torchvision
import random
import torch
import torch.nn as nn
from couette_solver import my2DCouetteSolver
from dataset import MyFlowDataset
from torch.utils.data import DataLoader


class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return self.mse(torch.log(pred + 1), torch.log(actual + 1))


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    batch_size,
    num_workers,
    pin_memory,
):
    # Consider that the couette solver now requires a desired_timesteps
    # parameter for improved reusabilty
    my_couette_data = my2DCouetteSolver(desired_timesteps=10000)
    my_images = my_couette_data[:-1]
    my_masks = my_couette_data[1:]
    my_zip = list(zip(my_images, my_masks))
    random.shuffle(my_zip)
    my_shuffled_images, my_shuffled_masks = zip(*my_zip)
    total_images = len(my_shuffled_images)
    # Implement a 90/10/10:train/dev/test split:
    # Consider that the couette solver yields 1100 - 1 timesteps
    number_train = int(0.9*total_images)
    number_dev = int(0.95*total_images)
    my_train_images = my_shuffled_images[:number_train]
    my_train_masks = my_shuffled_masks[:number_train]
    my_dev_images = my_shuffled_images[number_train:number_dev]
    my_dev_masks = my_shuffled_masks[number_train:number_dev]
    my_val_images = my_shuffled_images[number_dev:]
    my_val_masks = my_shuffled_masks[number_dev:]

    train_ds = MyFlowDataset(
        my_train_images,
        my_train_masks,
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # pin_memory=pin_memory,
    )

    dev_ds = MyFlowDataset(
            my_dev_images,
            my_dev_masks,
        )

    dev_loader = DataLoader(
            dataset=dev_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            # pin_memory=pin_memory,
        )

    val_ds = MyFlowDataset(
        my_val_images,
        my_val_masks,
    )

    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=pin_memory,
    )

    return train_loader, dev_loader, val_loader

# TODO: implement SSIM as a metric for comparing similarity between two images
# as proposed in the paper "Image Quality Assessment: From Error Visibility to
# Structural Similarity" by Zhou et al. (2004)


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    # TODO: Remove dice score, as it is only applicable to classification tasks
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
