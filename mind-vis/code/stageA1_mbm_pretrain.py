import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import argparse
import time
import timm.optim.optim_factory as optim_factory
import datetime
import matplotlib.pyplot as plt
import wandb
import copy
import sys
import os


from config import Config_MBM_fMRI
from dataset import hcp_dataset
from sc_mbm.mae_for_fmri import MAEforFMRI
from sc_mbm.trainer import train_one_epoch
from sc_mbm.trainer import NativeScalerWithGradNormCount as NativeScaler
from sc_mbm.utils import save_model
sys.path.insert(0,'./Color2Embed')
import models
from models import ColorEncoder as TargetColorEncoder

os.environ["WANDB_START_METHOD"] = "thread"
os.environ['WANDB_DIR'] = "."

class wandb_logger:
    def __init__(self, config):
        wandb.init(
                    project="mind-vis",
                    anonymous="allow",
                    group='stageA_sc-mbm',
                    config=config,
                    reinit=True)

        self.config = config
        self.step = None
    
    def log(self, name, data, step=None):
        if step is None:
            wandb.log({name: data})
        else:
            wandb.log({name: data}, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, name, fig):
        if self.step is None:
            wandb.log({name: wandb.Image(fig)})
        else:
            wandb.log({name: wandb.Image(fig)}, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

def get_args_parser():
    parser = argparse.ArgumentParser('MBM pre-training for fMRI', add_help=False)
    
    # Training Parameters
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--batch_size', type=int)

    # Model Parameters
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--mask_ratio', type=float)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--decoder_embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--decoder_num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)

    # Project setting
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--roi', type=str)
    parser.add_argument('--aug_times', type=int)
    parser.add_argument('--num_sub_limit', type=int)

    parser.add_argument('--include_hcp', type=bool)
    parser.add_argument('--include_kam', type=bool)

    parser.add_argument('--use_nature_img_loss', type=bool)
    parser.add_argument('--img_recon_weight', type=float)
    
    # distributed training parameters
    parser.add_argument('--local_rank', type=int)
                        
    return parser

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)
        
class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        return x

class ColorEncoder(nn.Module):
    def __init__(self, target_model, color_dim=512,num_voxels=224, patch_size=16, embed_dim=1024, in_chans=1):
        super(ColorEncoder, self).__init__()

        # self.vgg = vgg19(pretrained_path=None)
        # self.vgg = vgg19()
        self.target_model = target_model
        self.feature2vector = nn.Sequential(
            nn.Conv1d(97, color_dim, 2, 1, 2), # 8x8
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(color_dim, color_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(color_dim, color_dim, 2, 1, 2), # 4x4
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(color_dim, color_dim, 2, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool1d(16), # 1x1
            nn.Conv1d(color_dim, color_dim//2, 1), # linear-1
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(color_dim//2, color_dim//2, 1), # linear-2
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(color_dim//2, 291, 1), # linear-3
        )
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)
        self.patch_size = patch_size
        self.color_dim = color_dim
        
    def patchify(self, imgs):
        """
        imgs: (N, 1, num_voxels)
        x: (N, L, patch_size)
        """
        p = self.patch_embed.patch_size
        assert imgs.ndim == 3 and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size)
        imgs: (N, 1, num_voxels)
        """
        p = self.patch_embed.patch_size
        h = x.shape[1]
        
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs
        
    def forward_loss(self, imgs, pred):
        """
        imgs: [N, 1, num_voxels]
        pred: [N, L, p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        print(imgs.shape)
        target = self.target_model(imgs)
        print(target.shape)
        target = self.patchify(imgs)
        print(imgs.shape, target.shape, pred.shape)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = loss.sum() # mean loss on removed patches
        return loss

    def forward(self, x, img_features=None, valid_idx=None, mask_ratio=0.75):
        # x #[0, 1] RGB
        # vgg_fea = self.vgg(x, layer_name='relu5_2') # [B, 512, 16, 16]
        # print("fmri data shape: ", x.reshape(x.shape[0], 97, -1).shape)
        x_color = self.feature2vector(x.reshape(x.shape[0], 97, -1)) # [B, 512, 1, 1]
        # print("color data shape: ", x_color.shape)
        x_color.reshape(x_color.shape[0], -1)
        loss = self.forward_loss(x,x_color)
        return loss, x_color, torch.ones([x.shape[0],x.shape[1]],device=x.device)


def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)

def main(config):
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(config.local_rank) 
        torch.distributed.init_process_group(backend='nccl')
    output_path = os.path.join(config.root_path, 'results', 'fmri_pretrain',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    # output_path = os.path.join(config.root_path, 'results', 'fmri_pretrain')
    config.output_path = output_path
    logger = wandb_logger(config) if config.local_rank == 0 else None
    
    if config.local_rank == 0:
        os.makedirs(output_path, exist_ok=True)
        create_readme(config, output_path)
    
    device = torch.device(f'cuda:{config.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # create dataset and dataloader
    dataset_pretrain = hcp_dataset(path=os.path.join(config.root_path, 'data/HCP/npz'), roi=config.roi, patch_size=config.patch_size,
                transform=fmri_transform, aug_times=config.aug_times, num_sub_limit=config.num_sub_limit, 
                include_kam=config.include_kam, include_hcp=config.include_hcp)
   
    print(f'Dataset size: {len(dataset_pretrain)}\nNumber of voxels: {dataset_pretrain.num_voxels}')
    sampler = torch.utils.data.DistributedSampler(dataset_pretrain, rank=config.local_rank) if torch.cuda.device_count() > 1 else None 

    dataloader_hcp = DataLoader(dataset_pretrain, batch_size=config.batch_size, sampler=sampler, 
                shuffle=(sampler is None), pin_memory=True)

    # create model
    config.num_voxels = dataset_pretrain.num_voxels
    if config.model_name == 'MAEforFMRI':
        model = MAEforFMRI(num_voxels=dataset_pretrain.num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim,
                    decoder_embed_dim=config.decoder_embed_dim, depth=config.depth, 
                    num_heads=config.num_heads, decoder_num_heads=config.decoder_num_heads, mlp_ratio=config.mlp_ratio,
                    focus_range=config.focus_range, focus_rate=config.focus_rate, 
                    img_recon_weight=config.img_recon_weight, use_nature_img_loss=config.use_nature_img_loss)  
    elif config.model_name == 'ColorEncoder':
        target_model = TargetColorEncoder(color_dim=512)
        print(torch.load('/scratch/jz5952/mind-vis/045000.pt').keys())
        target_model.load_state_dict(torch.load('/scratch/jz5952/mind-vis/045000.pt')['colorEncoder'])
        model = ColorEncoder(target_model, num_voxels=dataset_pretrain.num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim)
    else:
        raise NotImplementedError 
    model.to(device)
    model_without_ddp = model
    if torch.cuda.device_count() > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[config.local_rank], output_device=config.local_rank, find_unused_parameters=config.use_nature_img_loss)

    param_groups = optim_factory.add_weight_decay(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if logger is not None:
        logger.watch_model(model,log='all', log_freq=1000)

    cor_list = []
    start_time = time.time()
    print('Start Training the fmri MAE ... ...')
    img_feature_extractor = None
    preprocess = None
    if config.use_nature_img_loss:
        from torchvision.models import resnet50, ResNet50_Weights
        from torchvision.models.feature_extraction import create_feature_extractor
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        m = resnet50(weights=weights)   
        img_feature_extractor = create_feature_extractor(m, return_nodes={f'layer2': 'layer2'}).to(device).eval()
        for param in img_feature_extractor.parameters():
            param.requires_grad = False
    img_feature_extractor = TargetColorEncoder(color_dim=512).to(device)
    img_feature_extractor.load_state_dict(torch.load('/scratch/jz5952/mind-vis/045000.pt')['colorEncoder'])
    for ep in range(config.num_epoch):
        if torch.cuda.device_count() > 1: 
            sampler.set_epoch(ep) # to shuffle the data at every epoch
        cor = train_one_epoch(model, dataloader_hcp, optimizer, device, ep, loss_scaler, logger, config, start_time, model_without_ddp,
                            img_feature_extractor, preprocess)
        cor_list.append(cor)
        if (ep % 20 == 0 or ep + 1 == config.num_epoch) and ep != 0 and config.local_rank == 0:
            # save models
            if config.model_name == 'MAEforFMRI':
                ckpt_name = 'checkpoints'
            else:
                ckpt_name = 'color_checkpoints'
            save_model(config, ep, model_without_ddp, optimizer, loss_scaler, os.path.join(output_path,ckpt_name))
            # plot figures
            plot_recon_figures(model, device, dataset_pretrain, output_path, 5, config, logger, model_without_ddp)
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if logger is not None:
        logger.log('max cor', np.max(cor_list), step=config.num_epoch-1)
        logger.finish()
    return

@torch.no_grad()
def plot_recon_figures(model, device, dataset, output_path, num_figures = 5, config=None, logger=None, model_without_ddp=None):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()
    fig, axs = plt.subplots(num_figures, 3, figsize=(30,15))
    fig.tight_layout()
    axs[0,0].set_title('Ground-truth')
    axs[0,1].set_title('Masked Ground-truth')
    axs[0,2].set_title('Reconstruction')

    for ax in axs:
        sample = next(iter(dataloader))['fmri']
        sample = sample.to(device)
        _, pred, mask = model(sample, mask_ratio=config.mask_ratio)
        sample_with_mask = model_without_ddp.patchify(sample).to('cpu').numpy().reshape(-1, model_without_ddp.patch_size)
        pred = model_without_ddp.unpatchify(pred).to('cpu').numpy().reshape(-1)
        sample = sample.to('cpu').numpy().reshape(-1)
        mask = mask.to('cpu').numpy().reshape(-1)
        # cal the cor
        cor = np.corrcoef([pred, sample])[0,1]

        x_axis = np.arange(0, sample.shape[-1])
        # groundtruth
        ax[0].plot(x_axis, sample)
        # groundtruth with mask
        s = 0
        for x, m in zip(sample_with_mask,mask):
            if m == 0:
                ax[1].plot(x_axis[s:s+len(x)], x, color='#1f77b4')
            s += len(x)
        # pred
        ax[2].plot(x_axis, pred)
        ax[2].set_ylabel('cor: %.4f'%cor, weight = 'bold')
        ax[2].yaxis.set_label_position("right")

    fig_name = 'reconst-%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    fig.savefig(os.path.join(output_path, f'{fig_name}.png'))
    if logger is not None:
        logger.log_image('reconst', fig)
    plt.close(fig)

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_MBM_fMRI()
    config = update_config(args, config)
    print("config: ",config)
    main(config)
    