import torch
import torchvision
import random
import torch
import torch.nn as nn
import numpy as np
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
    couette_dim
):
    # Consider that the couette solver now requires a desired_timesteps
    # parameter for improved reusabilty
    sigma = 0.1
    my_couette_data = my2DCouetteSolver(
        desired_timesteps=10000, vertical_resolution=couette_dim, sigma=sigma)
    print(f'Noise level: {sigma}.')
    my_images = my_couette_data[:-1]
    my_masks = my_couette_data[1:]
    my_zip = list(zip(my_images, my_masks))
    random.shuffle(my_zip)
    my_shuffled_images, my_shuffled_masks = zip(*my_zip)
    total_images = len(my_shuffled_images)
    # Implement a 90/10/10:train/dev/test split:
    # Consider that the couette solver yields 1100 - 1 timesteps
    number_train = int(0.9*total_images)
    number_val = int(0.95*total_images)
    my_train_images = my_shuffled_images[:number_train]
    my_train_masks = my_shuffled_masks[:number_train]
    my_val_images = my_shuffled_images[number_train:number_val]
    my_val_masks = my_shuffled_masks[number_train:number_val]
    my_test_images = my_shuffled_images[number_val:]
    my_test_masks = my_shuffled_masks[number_val:]

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

    test_ds = MyFlowDataset(
        my_test_images,
        my_test_masks,
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

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


def save3DArray2File(input_array, prediction):
    # 1) Convert 3D array to 2D array
    input_reshaped = input_array.reshape(input_array.shape[0], -1)

    # 2) Save 2D array to file
    t, c, x, y = input_array.shape
    name = f'{prediction}_{t}_{x}_{y}'
    np.savetxt(f'{name}.csv', input_reshaped)


def load3DArrayFromFile(input_file, input_shape):
    # 3) load 2D array from file
    loaded_array = np.loadtxt(f'{input_file}')

    # 4) Revert 2D array to 3D array
    original_array = loaded_array.reshape(
        loaded_array.shape[0], loaded_array.shape[1] // input_shape[2], input_shape[2])
    return original_array


def checkSaveLoad(input_array, loaded_array):
    print("shape of input array: ", input_array.shape)
    print("shape of loaded array: ", loaded_array.shape)

    if (input_array == loaded_array).all():
        print("Yes, both the arrays are same")
    else:
        print("No, both the arrays are not same")
