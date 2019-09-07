""" Search cell """
import os
import sys
import argparse
import time
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from tensorboardX import SummaryWriter
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from models.model_search import Network
import torch.backends.cudnn as cudnn
from dataset.mpii import MPIIDataset
from core.config import config
from core.config import update_config
from utils import utils
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


device = torch.device("cuda")

def parse_args():

    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # searching
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)

    args = parser.parse_args()

    return args

def reset_config(config, args):

    if args.gpus:
        config.GPUS = args.gpus



def main():

    args = parse_args()
    reset_config(config, args)
    #device = torch.device("cuda")
    # tensorboard
    if not os.path.exists(config.SEARCH.PATH):
        os.makedirs(config.SEARCH.PATH)
    writer = SummaryWriter(log_dir=os.path.join(config.SEARCH.PATH, "log"))
    logger = utils.get_logger(os.path.join(config.SEARCH.PATH, "{}.log".format(config.SEARCH.NAME)))
    logger.info("Logger is set - training start")
    
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # set seed
    #np.random.seed(config.SEARCH.SEED)
    #torch.manual_seed(config.SEARCH.SEED)
    #torch.cuda.manual_seed_all(config.SEARCH.SEED)

    torch.backends.cudnn.benchmark = True

    gpus = [int(i) for i in config.GPUS.split(',')]
    criterion = JointsMSELoss(use_target_weight = config.LOSS.USE_TARGET_WEIGHT).to(device)
    model = Network(config)
    if len(gpus)>1:
        model = nn.DataParallel(model)
    model = model.cuda()
    #for name,p in model.module.named_parameters():
    #    logger.info(name)
    
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    
    # weights optimizer
    params = model.parameters()
    #arch_params = list(map(id, model.module.arch_parameters()))
    #weight_params = filter(lambda p: id(p) not in arch_params, model.parameters())
    #params = [{'params': weight_params},
    #          {'params': model.module.arch_parameters(), 'lr': 0.0004}]

    optimizer = torch.optim.Adam(params, config.SEARCH.W_LR)
                               
    # split data to train/validation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    train_data = MPIIDataset(config,
                             config.DATASET.ROOT,
                             config.SEARCH.TRAIN_SET,
                             True,
                             transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                             ]))
    valid_data = MPIIDataset(config,
                             config.DATASET.ROOT,
                             config.SEARCH.TEST_SET,
                             False,
                             transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                             ]))
                           

    print(len(train_data),len(valid_data))
  
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)
                                               
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.SEARCH.BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=config.WORKERS,
                                               pin_memory=True)
                                             

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.SEARCH.LR_STEP, config.SEARCH.LR_FACTOR)

    # training loop
    best_top1 = 0.
    for epoch in range(config.SEARCH.EPOCHS):
    
        lr_scheduler.step()


        # training
        train(config, train_loader, model, criterion, optimizer, epoch, logger, writer)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1 = validate(config, valid_loader, valid_data, epoch+1, model, criterion, logger, writer)

        # log
        # genotype
        genotype = model.module.genotype()
        logger.info(F.softmax(model.module.alphas_normal, dim=-1))
        logger.info(F.softmax(model.module.alphas_reduce, dim=-1))
        logger.info("genotype = {}".format(genotype))

        # save
        state = {'state_dict':model.state_dict(),
                 'schedule':lr_scheduler.state_dict(),
                 'epoch':epoch+1}
        if best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(state, config.SEARCH.PATH, is_best)

    logger.info("Final best Accuracy = {:.3f}".format(best_top1))
    logger.info("Best Genotype = {}".format(best_genotype))


if __name__ == "__main__":
    main()
