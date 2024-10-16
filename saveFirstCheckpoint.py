import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from models import Discriminator, Generator
from collections import namedtuple

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6000"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def pain(rank: int, world_size: int):
    ddp_setup(rank, world_size)
    Config = namedtuple('Config', ['DATA', 'MODEL'])
    ModelConfig = namedtuple('ModelConfig', ['SWIN', 'DROP_RATE', 'DROP_PATH_RATE', 'PRETRAIN_CKPT'])
    SwinConfig = namedtuple('SwinConfig', ['PATCH_SIZE', 'IN_CHANS', 'EMBED_DIM', 'DEPTHS', 'NUM_HEADS', 'WINDOW_SIZE', 'MLP_RATIO', 'QKV_BIAS', 'QK_SCALE', 'APE', 'PATCH_NORM'])
    
    config = Config(
        DATA=namedtuple('DataConfig', ['IMG_SIZE'])(IMG_SIZE=256),
        MODEL=ModelConfig(
            SWIN=SwinConfig(
                PATCH_SIZE=4,
                IN_CHANS=3,
                EMBED_DIM=192,
                DEPTHS=[2, 2, 6, 2],
                NUM_HEADS=[3, 6, 12, 24],
                WINDOW_SIZE=8,
                MLP_RATIO=4.,
                QKV_BIAS=True,
                QK_SCALE=None,
                APE=False,
                PATCH_NORM=True
            ),
            DROP_RATE=0.0,
            DROP_PATH_RATE=0.1,
            PRETRAIN_CKPT=None
        ),
    
    )
    DEVICE = rank
    LEARNING_RATE = 2e-4
    disc = Discriminator().to(DEVICE)
    disc = DDP(disc, device_ids=[DEVICE])
    gen = Generator(config, img_size=256, num_classes=3).to(DEVICE)
    gen = DDP(gen, device_ids=[DEVICE])
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    checkpoint = {
        'epoch': 0,
        'disc_state_dict_best': disc.module.state_dict(),
        'gen_state_dict_best': gen.module.state_dict(),
        'disc_state_dict_last': disc.module.state_dict(),
        'gen_state_dict_last': gen.module.state_dict(),
        'optimizer_gen_state_dict': opt_gen.state_dict(),
        'optimizer_disc_state_dict': opt_disc.state_dict(),
        'g_scaler_state_dict': g_scaler.state_dict(),
        'd_scaler_state_dict': d_scaler.state_dict(),
        'best_gen_loss' : 1000000.0
    }

    torch.save(checkpoint, 'sar_model/checkpoints/check.pth')
    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(pain, args=(world_size,), nprocs=world_size)
    