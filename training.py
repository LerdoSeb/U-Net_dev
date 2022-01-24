import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import get_loaders, check_accuracy

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 10
NUM_WORKERS = 1
IMAGE_HEIGHT = 64  # 1280 originally
IMAGE_WIDTH = 64  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
# TRAIN_IMG_DIR = "data/train_images/"
# TRAIN_MASK_DIR = "data/train_masks/"
# VAL_IMG_DIR = "data/val_images/"
# VAL_MASK_DIR = "data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    # The train function will complete one epoch of the training cycle.
    loop = tqdm(loader)
    # print('tqdm')
    # The tqdm module allows to display a smart progress meter for iterables
    # using tqdm(iterable).

    for batch_idx, (data, targets) in enumerate(loop):
        # print('enum(loop)')
        # data = data.to(device=DEVICE)
        data = data.float().unsqueeze(1).to(device=DEVICE)
        # print('data success')
        # print(f'data.shape = {data.shape}')
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # print('targets success')
        # First consider the forward training path. This means calculate the
        # the predictions and determine the resultung error using the loss_fn.
        with torch.cuda.amp.autocast():
            # torch.cuda.amp and torch provide convenience methods for mixed
            # precision, where some operations use the torch.float32 (float)
            # datatype and other operations use torch.float16 (half). Some ops,
            # like linear layers and convolutions, are much faster in float16.
            # Other ops, like reductions, often require the dynamic range of
            # float32. Mixed precision tries to match each op to its appropriate
            # datatype.
            # print('autocast success')
            predictions = model(data)
            # print('model(data) success')
            loss = loss_fn(predictions.long(), targets.long())
            # print(f'The current mean average loss is: {loss}.')

        # Next consider the backward training path, especially the corresponding
        # scaler which is an object of the class GRADIENT SCALING:
        #
        # If the forward pass for a particular op has float16 inputs, the
        # backward pass for that op will produce float16 gradients. Gradient
        # values with small magnitudes may not be representable in float16.
        # These values will flush to zero (“underflow”), so the update for the
        # corresponding parameters will be lost.
        #
        # To prevent underflow, “gradient scaling” multiplies the network’s
        # loss(es) by a scale factor and invokes a backward pass on the scaled
        # loss(es). Gradients flowing backward through the network are then
        # scaled by the same factor. In other words, gradient values have a
        # larger magnitude, so they don’t flush to zero.
        #
        # Each parameter’s gradient (.grad attribute) should be unscaled before
        # the optimizer updates the parameters, so the scale factor does not
        # interfere with the learning rate.
        optimizer.zero_grad()
        # .zero_grad(): Sets the gradients of all optimized torch.Tensors to zero.
        scaler.scale(loss).backward()
        # .scale():
        # .backward(): Computes the gradient of current tensor w.r.t. graph
        # leaves. This function accumulates gradients in the leaves - you might
        # need to zero .grad attributes or set them to None before calling it.
        scaler.step(optimizer)
        # .step():
        scaler.update()
        # .update():

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        # postfix(): Specify additional stats to display at the end of the bar.


def main():
    # Our use-case does not require transforming the training data.
    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Rotate(limit=35, p=1.0),
    #         A.HorizontalFlip(p=0.5),
    #         A.VerticalFlip(p=0.1),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    # val_transforms = A.Compose(
    #     [
    #         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    #         A.Normalize(
    #             mean=[0.0, 0.0, 0.0],
    #             std=[1.0, 1.0, 1.0],
    #             max_pixel_value=255.0,
    #         ),
    #         ToTensorV2(),
    #     ],
    # )

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    # Instantiates the UNET neural network.
    loss_fn = nn.L1Loss()
    # Defines the loss function to be MAE (=Mean Average Error). Note that for
    # the initial semantic segmentation task the loss function was set to be
    # nn.BCEWithLogitsLoss (=Binary Cross Entropy).
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer":optimizer.state_dict(),
        # }
        # save_checkpoint(checkpoint)

        # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )


if __name__ == "__main__":
    main()
