
import os
import sys
import argparse
import random
import torch
import logging
import numpy as np
from trainerr import trainer
from configs.config import get_config
from network.DHUnet import DHUnet
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/input_dataset/', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='BCSS', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='input_dataset', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--output_dir', type=str, default='logs',help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=64, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=301, help='random seed')
parser.add_argument('--cfg', type=str, required=False, default='configs/DHUnet_224.yaml',metavar="FILE", help='path to config file', )

# add
parser.add_argument('--network', type=str, default='DHUnet',help='the model of network')  
parser.add_argument('--fold_no', type=int,default=-1, help='the i th fold')
parser.add_argument('--total_fold', type=int,default=5, help='total k fold cross-validation')


parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset

    if args.fold_no == -1:
        args.output_dir = os.path.join(args.output_dir, 'all')
    else:
        args.output_dir = os.path.join(args.output_dir, str(args.total_fold) + 'fold_' + str(args.fold_no))
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    net = DHUnet(config, num_classes=args.num_classes).cuda()
    net.load_from(config)

    logging.basicConfig(filename=args.output_dir + "/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    print(str(net))
    trainer(args, net, args.output_dir)
