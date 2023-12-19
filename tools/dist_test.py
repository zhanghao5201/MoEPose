 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms


import torch.utils.data
import torch.utils.data.distributed

from lib.config import cfg
from lib.config import update_config
from lib.core.loss import JointsMSELoss
from lib.core.function import train
from lib.core.function import distilling
from lib.core.function import validate as validate
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger
from lib.utils.utils import get_model_summary

import lib.dataset as dataset
import lib.models as models
from lib.utils.distributed import is_distributed
import lib.models.pose_resnet as pose_resnet
import lib.models.pose_hrnet as pose_hrnet
import lib.models.pose_hrnet1 as pose_hrnet1
import lib.models.pose_hrnet_kd as pose_hrnet_kd
import lib.models.pose_hrnet_kdstu as pose_hrnet_kdstu
import lib.models.pose_litehrnet_kd as pose_litehrnet_kd
import lib.models.pose_litehrnet_kdstu as pose_litehrnet_kdstu
import lib.models.pose_litehrnet as pose_litehrnet
import lib.models.pose_ditehrnet as pose_ditehrnet
import lib.models.pose_hrformerxin as pose_hrformerxin
import lib.models.pose_ditehrnet_kd as pose_ditehrnet_kd
import lib.models.pose_ditehrnet_kdstu as pose_ditehrnet_kdstu
import lib.models.pose_swinL as pose_swinL
import lib.models.pose_swinL_kd as pose_swinL_kd
import lib.models.pose_swinL_kdstu as pose_swinL_kdstu
import lib.models.pose_tcformer1 as pose_tcformer1
import lib.models.pose_hrformer_kd as pose_hrformer_kd
import lib.models.pose_pvt2_kd as pose_pvt2_kd
import lib.models.pose_hrformer_kdstu as pose_hrformer_kdstu
import lib.models.pose_pvt2_kdstu as pose_pvt2_kdstu
import lib.models.pose_liteposeS_kd as pose_liteposeS_kd
import lib.models.pose_liteposeS_kdstu as pose_liteposeS_kdstu

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--seed', type=int, default=305)
    parser.add_argument("--local_rank", type=int, default=-1) 
    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def get_sampler(dataset):    
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None

def main():
    args = parse_args()
    update_config(cfg, args)
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed) 

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)
    #torch.cuda.set_device(6)###########
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    gpus = list(cfg.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))  
        #print(device)  
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )   


    student = eval(cfg.MODEL.NAME+'_kdstu'+'.get_pose_net_kd')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    
    # logger.info(pprint.pformat(model))     
   # writer_dict['writer'].add_graph(student, (dump_input, ))

    #logger.info(get_model_summary(student, dump_input))
    #torch.cuda.set_device(2)
    if distributed:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = cfg.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    
    if cfg.TEST.MODEL_FILE:
        print("sddd")
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        student.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cfg.TEST.MODEL_FILE).items()})
        #teacher.load_state_dict(torch.load(cfg.MODEL.TEACHER), strict=False)  


    if distributed:
        #print("od")
        #model = model.to(device)
        
        student=student.to(device)
        
        student = torch.nn.parallel.DistributedDataParallel(
            student,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        #model = nn.DataParallel(model, device_ids=gpus).cuda()
        student = torch.nn.DataParallel(student, device_ids=cfg.GPUS).cuda()
        
    
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    )

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
   
    test_sampler = get_sampler(valid_dataset)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf_s = 0.0
    best_model_s = False
    last_epoch = -1
    optimizer_s = get_optimizer(cfg, student)

    for epoch in range(1):
        if args.local_rank <= 0:
            perf_indicator_s = validate(
            cfg, valid_loader, valid_dataset, student, criterion,
            final_output_dir, tb_log_dir,  'student'
        )
            


if __name__ == '__main__':
    main()
#python tools/kd.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3_kd.yaml 
