# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# ACKNOWLEDGEMENT
# Part of the code within this file was taken and revised accordingly from the following repo:
# https://github.com/princeton-vl/RAFT-Stereo/blob/main/demo.py
# Please refer to the above repo for license requirement for using the RAFT-stereo code. 


import sys
import torch
import argparse
from pathlib import Path

from utils import *
from third_party.RAFT_Stereo.core.raft_stereo import RAFTStereo
from third_party.RAFT_Stereo.core.utils.utils import InputPadder


def disparity_from_stereo_raft2d(left_img, right_img, pretrained_model, device):

    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

    # high-quality model
    if pretrained_model == 'high_quality_model':
        args = parser.parse_args(['--restore_ckpt', '%s/third_party/RAFT_Stereo/models/raftstereo-middlebury.pth' % str(Path(sys.path[0]))])
        args.corr_implementation = 'reg_cuda'
        args.mixed_precision = True

    # real-time model
    elif pretrained_model == 'real_time_model':
        args = parser.parse_args(['--restore_ckpt', '%s/third_party/RAFT_Stereo/models/raftstereo-realtime.pth' % str(Path(sys.path[0]))])
        args.shared_backbone = True
        args.n_downsample = 3
        args.n_gru_layers = 2
        args.slow_fast_gru = True
        args.valid_iters = 7 
        args.corr_implementation = 'reg' # reg_cuda
        args.mixed_precision = True    


    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    model.to(device)
    model.eval()

    with torch.no_grad():
        left_img = left_img.copy()
        right_img = right_img.copy()

        image1 = torch.from_numpy(left_img).permute(2, 0, 1).float()[None].to(device)
        image2 = torch.from_numpy(right_img).permute(2, 0, 1).float()[None].to(device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        disparity = flow_up.detach().cpu().numpy().squeeze()    

        height, width = left_img.shape[0], left_img.shape[1]
        height_disp, width_disp = disparity.shape[0], disparity.shape[1]
        tmp_crop_height = int((height_disp-height)/2)
        tmp_crop_width = int((width_disp-width)/2)
        disparity = disparity[tmp_crop_height:height_disp-tmp_crop_height,tmp_crop_width:width_disp-tmp_crop_width]
        #print("croped boundary each side y: %d   x:%d" % (tmp_crop_height, tmp_crop_width))        

    return disparity
