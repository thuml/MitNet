import os
import time
import random
import warnings
import argparse
import numpy as np
from collections import Counter

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from dataset import TCGA
from models import TARNet, MitNet
from functions import factual_mse_loss, mutual_info, pehe_sqrt
from utils import AverageMeter, ProgressMeter, CompleteLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(train_loader, model, loss_func_dict, optimizer, epoch, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = {k: AverageMeter(f'{k} Loss', ':3.2f') for k in loss_func_dict.keys()}
    losses["total"] = AverageMeter('Total Loss', ':3.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time] + list(losses.values()),
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        data = {k: v.to(device) for k, v in data.items()}
        bsz = data["treatment"].shape[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(**data)

        # compute loss
        loss_dict = {k: func(**data, **output) for k, (func, _) in loss_func_dict.items()}
        for k, v in loss_dict.items():
            losses[k].update(v.item(), bsz)

        total_loss = sum([loss_dict[k] * trade_off for k, (_, trade_off) in loss_func_dict.items()])
        losses["total"].update(total_loss.item(), bsz)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, loss_func_dict, args, device):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = {k: AverageMeter(f'{k} Loss', ':3.2f') for k in loss_func_dict.keys()}

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time] + list(losses.values()),
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            data = {k: v.to(device) for k, v in data.items()}
            bsz = data["treatment"].shape[0]

            # compute output
            output = model(**data)

            # compute loss
            loss_dict = {k: func(**data, **output) for k, func in loss_func_dict.items()}
            for k, v in loss_dict.items():
                losses[k].update(v.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return {k: v.avg for k, v in losses.items()}


def main(args):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Load and split data
    dataset = TCGA(os.path.join(args.data, "tcga.npz"), normalize=True)
    t_list = dataset[:]["treatment"].reshape(-1).tolist()
    t_counter = Counter(t_list)
    prop = torch.tensor([t_counter[i] / len(t_list) for i in t_counter.keys()]).to(device)

    train_length = int(0.63 * len(dataset))
    val_length = int(0.27 * len(dataset))
    test_length = len(dataset) - train_length - val_length
    train_set, val_set, test_set = random_split(dataset, [train_length, val_length, test_length])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Creat model
    input_dim = dataset[0]["inputs"].shape[0]
    if args.model == "TARNet":
        model = TARNet(input_dim, prop)
        train_loss_func_dict = {
            "factual": (factual_mse_loss, 1),
        }
        val_loss_func_dict = {
            "factual": factual_mse_loss,
            "pehe_sqrt": pehe_sqrt,
        }
    elif args.model == "MitNet":
        model = MitNet(input_dim, prop)
        train_loss_func_dict = {
            "factual": (factual_mse_loss, 1),
            "mutual_info": (mutual_info, -args.alpha),
        }
        val_loss_func_dict = {
            "factual": factual_mse_loss,
            "pehe_sqrt": pehe_sqrt,
            "mutual_info": mutual_info
        }
    else:
        raise NotImplementedError
    
    print(model)
    model = model.to(device)

    if args.phase == "train":
        # define optimizer
        optimizer = SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)

        best_pehe = 1e10

        for epoch in range(args.epochs):
            # train for one epoch
            train(train_loader, model, train_loss_func_dict, optimizer, epoch, args, device)

            # evaluate on validation set
            losses = validate(val_loader, model, val_loss_func_dict, args, device)

            print(losses)

            if np.isnan(losses["pehe_sqrt"]):
                break
            
            # remember best pehe and save checkpoint
            if best_pehe > losses["pehe_sqrt"]:
                best_pehe = losses["pehe_sqrt"]
                torch.save(model.state_dict(), logger.get_checkpoint_path('best'))
            
            torch.save(model.state_dict(), logger.get_checkpoint_path('latest'))

    print("Loading the best checkpoint...")
    model.load_state_dict(torch.load(logger.get_checkpoint_path('best')))

    test_losses = validate(test_loader, model, val_loss_func_dict, args, device)
    print(test_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MitNet for TCGA')
    # data parameters
    parser.add_argument('-p', '--path', metavar='PATH', default='../data/')
    # model parameters
    parser.add_argument('-m', '--model', metavar='MODEL', default='TARNet', choices=['TARNet', 'MitNet'])
    parser.add_argument('--alpha', metavar='ALPHA', default=0.3, type=float)
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.005, type=float,
                        metavar='W', help='weight decay (default: 5e-3)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    # miscs
    parser.add_argument('-f', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='exp0',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    args = parser.parse_args()
    main(args)
