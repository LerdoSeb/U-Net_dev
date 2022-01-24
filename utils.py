import torch
import torchvision
import random
from couette_solver import my2DCouetteSolver
from dataset import MyFlowDataset
from torch.utils.data import DataLoader


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
    my_couette_data = my2DCouetteSolver()
    my_images = my_couette_data[:-1]
    my_masks = my_couette_data[1:]
    my_zip = list(zip(my_images, my_masks))
    random.shuffle(my_zip)
    my_shuffled_images, my_shuffled_masks = zip(*my_zip)
    # Consider that the couette solver yields 1100 - 1 timesteps
    my_train_images = my_shuffled_images[:900]
    my_train_masks = my_shuffled_masks[:900]
    my_val_images = my_shuffled_images[900:]
    my_val_masks = my_shuffled_masks[900:]

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

    return train_loader, val_loader


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
