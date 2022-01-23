import argparse
import os
import random
import time
import datetime
import warnings

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.VIT import VIT
from utils.scheduler import *

parser = argparse.ArgumentParser(description='PyTorch VIT Training')
"""
Option: https://github.com/google-research/vision_transformer/blob/main/vit_jax/configs/common.py
"""
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument("--img-size", default=224, type=int,
                    help="image Size(Resolution)")
parser.add_argument("--patch-size", default=16, type=int,
                    help="Patch Size(Resolution)")
parser.add_argument("--num-classes", default=100, type=int)
parser.add_argument("--dim", default=1024, type=int)
parser.add_argument("--n_layers", default=6, type=int)
parser.add_argument("--heads", default=16, type=int)
parser.add_argument("--mlp-dim", default=2048, type=int)

parser.add_argument('--lr', '--learning-rate', default=3e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=0., type=float, metavar='M',
                    help='weight decay')
parser.add_argument('--decay-type', default="cosine", choices=['linear', 'cosine'])
parser.add_argument('--grad-norm-clip', default=1.0, type=float)
parser.add_argument('--n-steps', default=10_000, type=int)
parser.add_argument('--warmup-steps', default=500, type=int)

parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc1 = 0
    args.gpu = gpu
    summary = SummaryWriter()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Models
    model = VIT(image_size=args.img_size,
                patch_size=args.patch_size,
                num_classes=args.num_classes,
                dim=args.dim,
                n_layers=args.n_layers,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dropout=0.1,
                emb_dropout=0.1)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("this")
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    else:
        model = torch.nn.DataParallel(model).cuda()

    # Optimizer / Criterion
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.n_steps)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.n_steps)

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # DataSet / DataLoader
    train_dataset = datasets.CIFAR100(
        root='./dataset',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                                           std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])])
    )
    test_dataset = datasets.CIFAR100(
        root='./dataset',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.RandomResizedCrop((args.img_size, args.img_size)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5088964127604166, 0.48739301317401956, 0.44194221124387256],
                                                           std=[0.2682515741720801, 0.2573637364478126, 0.2770957707973042])])
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, scheduler, summary, args)


def train(train_loader, model, criterion, optimizer, epoch, scheduler, summary, args):
    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_norm_clip)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        optimizer.zero_grad()
        optimizer.step()
        scheduler.step()

        niter = epoch * len(train_loader) + i
        summary.add_scalar('Train/loss', loss.item(), niter)
        summary.add_scalar('Train/acc1', acc1[0], niter)
        summary.add_scalar('Train/acc5', acc5[0], niter)

        if i % args.print_freq == 0:
            print(" Epoch [%d][%d/%d] | Loss: %f | acc1: %f | acc5: %f"
                  % (epoch + 1, i, len(train_loader), loss, acc1[0], acc5[0]))

    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()