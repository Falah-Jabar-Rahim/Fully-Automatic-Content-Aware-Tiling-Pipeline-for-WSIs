import argparse
import os
from config import get_config
from PIL import Image
from torchvision import transforms

Image.MAX_IMAGE_PIXELS = None
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libffi.so.7'  # update this path if needed

parser = argparse.ArgumentParser()


def inti_model_artifact():
    ### network parameters
    parser.add_argument('--volume_path', type=str,
                        default='my_dataset/',
                        help='root dir for validation volume data')  # for acdc volume_path=root_dir
    parser.add_argument('--dataset', type=str,
                        default='BCSS', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=4, help='output channel of network')
    parser.add_argument('--list_dir', type=str,
                        default='my_dataset', help='list dir')
    parser.add_argument('--pretrained_ckpt', type=str,
                        default='pretrained_ckpt/WSI-QA.pth', help='ckpt')
    parser.add_argument('--output_dir', type=str, default='output', help='output dir')
    parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
    parser.add_argument('--is_savenii', default=True, action="store_true",
                        help='whether to save results during inference')
    parser.add_argument('--test_save_dir', type=str, default='output', help='saving prediction as nii!')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
    parser.add_argument('--seed', type=int, default=301, help='random seed')
    parser.add_argument('--cfg', type=str, default="configs/DHUnet_224.yaml", metavar="FILE",
                        help='path to config file', )
    parser.add_argument('--network', type=str, default='DHUnet', help='the model of network')
    parser.add_argument('--fold_no', type=int, default=1, help='the i th fold')
    parser.add_argument('--total_fold', type=int, default=5, help='total k fold cross-validation')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
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
    parser.add_argument('--z_spacing', default=1, type=int, help='')


    ### WSI parameters, tune these parameters if needed
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=270, help='required tile size')
    parser.add_argument('--wsilevel', default=0, type=int, help='level from open slide to read')
    parser.add_argument('--thumbnail_size', default=5000, type=int, help='required wsi thumbnail resolution')
    parser.add_argument('--wsi_folder', default="input_WSI/", type=str,
                        help='folder contains wsi images')
    parser.add_argument('--cpu_workers', default=40, type=int, help='number of cpu workers')
    parser.add_argument('--save_seg', default=1, type=int, help='to save tile segmentation results')
    parser.add_argument('--back_thr', default=50, type=int, help='% of background for each tile (eg., remove tiles with 50% background)')
    parser.add_argument('--blur_fold_thr', default=10, type=int, help='% of blur and fold (eg., remove tiles with 10% fold or blur)')
    parser.add_argument('--overlap', default=25, type=int, help='overlapping between neighbouring tiles')
    parser.add_argument('--max_marker_thr', default=0.9, type=float, help='max thr for pen-marking detection')
    parser.add_argument('--min_marker_thr', default=0.3, type=float, help='min thr for pen-marking detection')
    parser.add_argument('--clean_penmarker', default=1, type=int, help='to remove pen-marking or 0 do not remove pen-marking')


    args = parser.parse_args()
    config = get_config(args)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

    return args, config, test_transform
