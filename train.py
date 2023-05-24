import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from config import config
from dataloader.dataloader import get_train_loader
from models.builder import EncoderDecoder as segmodel
# from models.build_eaef import EAEF_EncoderDecoder as eaefmodal
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor
from eval import Evaluator, SegEvaluator
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from dataloader.dataloader import ValPre

from tensorboardX import SummaryWriter
import eval

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
scaler = GradScaler()

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=config, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255),
                     norm_layer=nn.BatchNorm2d)

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.999),
                                      weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()

    if __name__ == '__main__':
        for epoch in range(engine.state.epoch, config.nepochs + 1):
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            dataloader = iter(train_loader)

            sum_loss = 0

            for idx in pbar:
                engine.update_iteration(epoch, idx)

                minibatch = dataloader.next()
                imgs = minibatch['data']
                gts = minibatch['label']
                modal_xs = minibatch['modal_x']

                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                modal_xs = modal_xs.cuda(non_blocking=True)

                criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
                logits = model(imgs, modal_xs)
                loss = criterion(logits, gts)

                # reduce the whole loss over multi-gpu
                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                with autocast():
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                current_idx = (epoch - 1) * config.niters_per_epoch + idx
                lr = lr_policy.get_lr(current_idx)

                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                if engine.distributed:
                    sum_loss += reduce_loss.item()
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                                + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                + ' lr=%.4e' % lr \
                                + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                else:
                    sum_loss += loss
                    print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                                + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                                + ' lr=%.4e' % lr \
                                + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))

                del loss
                pbar.set_description(print_str, refresh=False)

            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (
                    epoch == config.nepochs):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint("D:/RGBX_Semantic_Segmentation/checkpoint",
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint("D:/RGBX_Semantic_Segmentation/checkpoint",
                                                    config.log_dir,
                                                    config.log_dir_link)

                model_name = 'D:/RGBX_Semantic_Segmentation/checkpoint/epoch-' + str(epoch) + '.pth'
                parser = argparse.ArgumentParser()
                parser.add_argument('-e', '--epochs',
                                    default=model_name, type=str)
                parser.add_argument('-d', '--devices', default='0', type=str)
                parser.add_argument('-v', '--verbose', default=False, action='store_true')
                parser.add_argument('--show_image', '-s', default=False,
                                    action='store_true')
                parser.add_argument('--save_path', '-p', default=None)

                args = parser.parse_args()
                all_dev = parse_devices(args.devices)

                data_setting = {'rgb_root': config.rgb_root_folder,
                                'rgb_format': config.rgb_format,
                                'gt_root': config.gt_root_folder,
                                'gt_format': config.gt_format,
                                'transform_gt': config.gt_transform,
                                'x_root': config.x_root_folder,
                                'x_format': config.x_format,
                                'x_single_channel': config.x_is_single_channel,
                                'train_source': config.train_source,
                                'eval_source': config.eval_source,
                                'class_names': config.class_names}
                val_pre = ValPre()
                dataset = RGBXDataset(data_setting, 'val', val_pre)
                with torch.no_grad():
                    segmentor = SegEvaluator(dataset, config.num_classes, config.norm_mean,
                                             config.norm_std, model,
                                             config.eval_scale_array, config.eval_flip,
                                             all_dev, args.verbose, args.save_path,
                                             args.show_image)
                    segmentor.run(config.checkpoint_dir, args.epochs, config.val_log_file,
                                  config.link_val_log_file)



