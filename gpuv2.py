import torch
import torch.nn as nn
import albumentations as A
import albumentations as A

from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import itertools
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.utils as vutils
import shutil
import random
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from models import Discriminator, Generator
from splitDataset import create_empty_dirs, traverse_and_split
import os
from torch.utils.tensorboard import SummaryWriter
from datasetLoader import MapDataset

torch.backends.cudnn.benchmark = True

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)




def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler, L1_LAMBDA, DEVICE):
    loop = tqdm(loader, leave=True)

    total_l1_loss = 0.0
    total_d_real_loss = 0.0
    total_d_fake_loss = 0.0
    num_batches = 0

    for idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real_patch, D_real_pixel = disc(x, y)
            D_fake_patch, D_fake_pixel = disc(x, y_fake.detach())

            # Patch-level loss
            D_real_loss_patch = bce(D_real_patch, torch.ones_like(D_real_patch))
            D_fake_loss_patch = bce(D_fake_patch, torch.zeros_like(D_fake_patch))

            # Pixel-level loss
            D_real_loss_pixel = bce(D_real_pixel, torch.ones_like(D_real_pixel))
            D_fake_loss_pixel = bce(D_fake_pixel, torch.zeros_like(D_fake_pixel))

            # Combined loss
            D_real_loss = (D_real_loss_patch + D_real_loss_pixel) / 2
            D_fake_loss = (D_fake_loss_patch + D_fake_loss_pixel) / 2
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake_patch, D_fake_pixel = disc(x, y_fake)
            G_fake_loss_patch = bce(D_fake_patch, torch.ones_like(D_fake_patch))
            G_fake_loss_pixel = bce(D_fake_pixel, torch.ones_like(D_fake_pixel))
            G_fake_loss = (G_fake_loss_patch + G_fake_loss_pixel) / 2
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Update metrics
        total_l1_loss += L1.item() * x.size(0)
        total_d_real_loss += D_real_loss.item() * x.size(0)
        total_d_fake_loss += D_fake_loss.item() * x.size(0)
        num_batches += x.size(0)

        if idx % 10 == 0:
            loop.set_postfix(
                D_real_patch=torch.sigmoid(D_real_patch).mean().item(),
                D_fake_patch=torch.sigmoid(D_fake_patch).mean().item(),
                D_real_pixel=torch.sigmoid(D_real_pixel).mean().item(),
                D_fake_pixel=torch.sigmoid(D_fake_pixel).mean().item(),
            )

    # Compute average losses
    avg_l1_loss = total_l1_loss / num_batches
    avg_d_real_loss = total_d_real_loss / num_batches
    avg_d_fake_loss = total_d_fake_loss / num_batches

    return avg_l1_loss, avg_d_real_loss, avg_d_fake_loss

def evaluate_generator_fn(gen, L1_LOSS, val_loader, DEVICE):
    gen.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad(): 
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = gen(inputs)

            loss = L1_LOSS(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

    mean_l1_loss = total_loss / num_batches

    return mean_l1_loss

def log_losses_generator(epoch, gen_train_loss, gen_test_loss, writer):
    #writer.add_scalar('Loss/Train', gen_train_loss, epoch)
     writer.add_scalars('Loss',
                    {'train': gen_train_loss, 'test': gen_test_loss},
                    epoch)

def log_losses_disc(epoch, disc_real, disc_fake, writer):
    writer.add_scalar('Loss/Discriminator Real', disc_real, epoch)
    writer.add_scalar('Loss/Discriminator Fake', disc_fake, epoch)

def log_images(epoch, images, writer):
    batch_size = images.size(0)  # Get batch size dynamically
    for i in range(batch_size):
        img = images[i]
        # Convert the image tensor to a format suitable for TensorBoard
        writer.add_image(f'Image/epoch_{epoch}_image_{i}', img, epoch)

def display_tensor_board_image(val_loader, DEVICE, gen) :
    num_batches = len(val_loader)
    random_index = random.randint(0, num_batches - 1)
    random_batch = next(itertools.islice(val_loader, random_index, None))
    batch_data, batch_targets = random_batch
    x, y = batch_data.to(DEVICE), batch_targets.to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
    y_fake = y_fake * 0.5 + 0.5
    x = x * 0.5 + 0.5
    y = y * 0.5 + 0.5
    x_3channel = x.repeat(1, 3, 1, 1)
    comparison = torch.cat([x_3channel, y, y_fake], dim=3)
    gen.train()
    return comparison

def main(rank: int, world_size: int):
    print("lol")
    ddp_setup(rank, world_size)

    DEVICE = rank
    TRAIN_DIR = "train/"
    VAL_DIR = "val/"
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8
    NUM_WORKERS = 2
    IMAGE_SIZE = 256
    CHANNELS_IMG = 3
    L1_LAMBDA = 100
    LAMBDA_GP = 10
    NUM_EPOCHS = 250


    log_dir = "/sar_model/logs/"
    #tensorboard_writer = SummaryWriter(log_dir=log_dir)

    disc = Discriminator().to(DEVICE)
    disc = DDP(disc, device_ids=[DEVICE])
    gen = Generator(in_channels=1, features=64).to(DEVICE)
    gen = DDP(gen, device_ids=[DEVICE])
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = MapDataset(root_dir=TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        sampler=DistributedSampler(train_dataset)
    )
    val_dataset = MapDataset(root_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, sampler=DistributedSampler(val_dataset))
    epoch = 0
    best_gen_loss = 100000
    while epoch < 250:
        mean_train_l1_loss, mean_train_l1_d_real, mean_train_d_fake = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, L1_LAMBDA, DEVICE)
        mean_test_l1_loss = evaluate_generator_fn(gen, L1_LOSS, val_loader, DEVICE)

        if mean_test_l1_loss < best_gen_loss:
            best_gen_loss = mean_test_l1_loss
            disc_best = disc
            gen_best = gen

        #images = display_tensor_board_image()  # Ensure this returns a tensor of images
        #log_losses_generator(epoch, mean_train_l1_loss, mean_test_l1_loss, tensorboard_writer)
        #log_losses_disc(epoch, mean_train_l1_d_real, mean_train_d_fake, tensorboard_writer)
        #grid = vutils.make_grid(images, nrow=8, normalize=True)
        #tensorboard_writer.add_image('image_grid', grid)
        print(mean_train_l1_loss)
        print(mean_train_l1_d_real)
        print(mean_train_d_fake)
        print(mean_test_l1_loss)
        checkpoint = {
            'epoch': epoch,
            'disc_state_dict_best': disc_best.module.state_dict(),
            'gen_state_dict_best': gen_best.module.state_dict(),
            'disc_state_dict_last': disc.module.state_dict(),
            'gen_state_dict_last': gen.module.state_dict(),
            'optimizer_gen_state_dict': opt_gen.state_dict(),
            'optimizer_disc_state_dict': opt_disc.state_dict(),
            'g_scaler_state_dict': g_scaler.state_dict(),
            'd_scaler_state_dict': d_scaler.state_dict(),
            'best_gen_loss' : best_gen_loss
        }

        epoch += 1
        #torch.save(checkpoint, 'sar_model/checkpoints/check.pth')

    tensorboard_writer.close()
    destroy_process_group()

    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("lol")
    mp.spawn(main, args=(world_size,), nprocs=world_size)
    
    print("lol")

