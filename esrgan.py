"""
Font: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/esrgan.py
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch


#TODO change pathname
node_id = "TEST_NODE"
experiment_id = "TEST_EXPERIMENT_ID"
directory = f"mo833-trabalho/experiments/{experiment_id}/{time.time()}/PIs_logs/"
file_name = f"{directory}{node_id}-{experiment_id}-{time.time()}.out"

os.makedirs(directory, exist_ok=True)
os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=128, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=128, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=5, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
parser.add_argument("--local_rank", type=int, help="Local rank. Necessary for using the \
                        torch.distributed.launch utility.")
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available():
    device = torch.device('cuda')
    device_ids = list(range(torch.cuda.device_count()))
    gpus = len(device_ids)
    print('GPU detected')
else:
    device = torch.device("cpu")
    print('No GPU. switching to CPU')

#Initialize distributed backend
dist.init_process_group(backend="gloo")
rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
print(f"Rank: {rank}. World Size: {world_size}")


hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

#Wrap as Distributed
generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[opt.local_rank] if opt.cuda else None, \
                            output_device=opt.local_rank if opt.cuda else None)
discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[opt.local_rank] if opt.cuda else None, \
                        output_device=opt.local_rank if opt.cuda else None)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#TODO make work with this configuration
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
#Tensor = torch.Tensor

#Load wrapped in Distributed Sampler
train_sampler = DistributedSampler(dataset=ImageDataset("./data/%s" % opt.dataset_name, hr_shape=hr_shape))

#TODO Original solution of dataload
#dataloader = DataLoader(
#    ImageDataset("./data/%s" % opt.dataset_name, hr_shape=hr_shape),
#    batch_size=opt.batch_size,
#    shuffle=True,
#    num_workers=opt.n_cpu,
#)
dataloader = DataLoader(dataset=ImageDataset("./data/%s" % opt.dataset_name, hr_shape=hr_shape), batch_size=opt.batch_size, sampler=train_sampler, \
                                num_workers=opt.n_cpu)

# ----------
#  Training
# ----------

initialization_time = time.time() - start_time
with open(file_name, "a") as experiment_file:
    experiment_file.write(f"[MO833] Rank,{rank},Initialization Time: {initialization_time:.4f}\n")
#print(f"[MO833] Rank,{rank},Initialization Time: {initialization_time:.4f}")

for epoch in range(opt.epoch, opt.n_epochs):
    epoch_start_time = time.time()
    for i, imgs in enumerate(dataloader):

        iteration_start_time = time.time()

        batches_done = epoch * len(dataloader) + i

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.module.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done < opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
            )

            iteration_end_time = time.time()
            iteration_time = iteration_end_time - iteration_start_time
            elapsed_time = iteration_end_time - start_time
            #TODO rename experiment output file
            #<node_id>-<experiment_id>-<timestamp>.out,(em UNIX Time)

            with open(file_name, "a") as experiment_file:
                experiment_file.write(f"[MO833] Rank,{rank},Epoch,{epoch},Iteration,{i},It. time,{iteration_time:.4f},Elapsed time,{elapsed_time:.4f}\n")
            #print(f"[MO833] Rank,{rank},Epoch,{epoch},Iteration,{i},It. time,{iteration_time:.4f},Elapsed time,{elapsed_time:.4f}")

            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr)

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # Total generator loss
        loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_content.item(),
                loss_GAN.item(),
                loss_pixel.item(),
            )
        )

        
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))
            save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

        if batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" %epoch)

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    elapsed_time = epoch_end_time - start_time
    with open(file_name, "a") as experiment_file:
        experiment_file.write(f"[MO833] Rank,{rank},Epoch,{epoch},Epoch time,{epoch_time:.4f},Elapsed time,{elapsed_time:.4f}\n")
    #print(f"[MO833] Rank,{rank},Epoch,{epoch},Epoch time,{epoch_time:.4f},Elapsed time,{elapsed_time:.4f}")