# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import warnings
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils import *
from custombilateral import *
from stereo_depth import *
from net import NPTnet 



if __name__ == "__main__":
    
    ## select test data
    folder_proto = "%s/data/test/synthetic/" % (str(Path(sys.path[0]).parent))  
    
    ###########################################################
    # select scene
    scene, seed, num_frames = 'dancestudio', 4, 30 
    #scene, seed, num_frames = 'electronicroom', 3, 30
    ###########################################################

    print("test scene: %s, seed: %d, frames: %d" % (scene, seed, num_frames))
    s = 1.0 # temporary image intensity scaling parameter 
    flag_quality_evaluation = True

    folder = "%s/%s/seed%03d/" % (folder_proto, scene, seed)
    output_folder = "%s/output-%s-seed%03d/" % (str(Path(sys.path[0]).parent), scene, seed)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)  
    flag_visualize = True # save intermediate images

    ## define output view location
    tgt_view_x = [0.02, 0.08] # horizontal translation of target views w.r.t. the left input camera
    tgt_view_z = -0.093 # depth-axis offset of target views w.r.t. the input cameras

    ## set device
    assert torch.cuda.is_available(), "CUDA is not found."
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    cudnn.enabled = True
    cudnn.benchmark = True        
    warnings.filterwarnings('ignore', category=UserWarning, append=True) 
       
       
    ## stereo camera calibration parameters
    # cam 1 (left)
    h, w, fov_x = 720, 1280, 90 # height, width, horizontal field of view in degree
    input_baseline = 0.1 # in meter

    mtx_color = get_intrinsics(w,h,fov_x)
    dis_color = np.array([0.,0.,0.,0.,0.,0.,0.,0.]).T
    rotation_none = np.array([[1., 0., 0.],\
    [0., 1., 0.],\
    [0., 0., 1.]])
    translation_none = np.array([[0.,0.,0.]]).T

    # cam 2 (right)
    mtx_color_2 = mtx_color
    dis_color_2 = dis_color        
    rotation_color_2 = rotation_none
    translation_color_2 = np.array([[-input_baseline, 0., 0.]]).T


    ## stereo rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_color, dis_color, mtx_color_2, dis_color_2, (w, h), rotation_color_2, translation_color_2, alpha=1.)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx_color, dis_color, R1, P1, (w, h), cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(mtx_color_2, dis_color_2, R2, P2, (w, h), cv2.CV_32F)    
    translation_color_2_after_rectification = np.array([[P2[0][3]/P2[0][0], 0, 0]]).T

            
    ## load trained mode
    model_path = "%s/model/model.pth" % (str(Path(sys.path[0]).parent))
    model = NPTnet().to(device)  
    if os.path.isfile(model_path):
        state = model.state_dict()
        state.update(torch.load(model_path)['model_state_dict'])
        model.load_state_dict(state)
    else:
        input("can't find the trained model")

    if flag_quality_evaluation:
        metric_psnr = np.zeros((len(tgt_view_x), num_frames),dtype=float)
        metric_ssim = np.zeros((len(tgt_view_x), num_frames),dtype=float)

    ## run neural passthrough method
    for frame in range(0,num_frames,1):
        print("process frame %03d..." % frame)
  
        # read input stereo mages
        # no need to run stereo rectification for synthetic images    
        img_rect = np.flip(cv2.imread("%s/cam%03d/rgb_frame.%04d.png" % (folder, 0, frame)), 2) * s / 255.
        img_rect_2 = np.flip(cv2.imread("%s/cam%03d/rgb_frame.%04d.png" % (folder, 1, frame)), 2) * s / 255.           
        if flag_visualize:
            save_png_from_numpy(img_rect/s, "%s/rectified_image_left_frame%04d.png" % (output_folder, frame))
            save_png_from_numpy(img_rect_2/s, "%s/rectified_image_right_frame%04d.png" % (output_folder, frame))

        # load precomputed mask for each camera that boundaries to discard
        # usually none for synthetic data
        mask_cam = np.zeros((h,w), dtype=bool) 
        mask_cam_2 = np.zeros((h,w), dtype=bool)
        
        # stereo depth estimation
        print("   run stereo depth estimation...")
        disparity = disparity_from_stereo_raft2d(img_rect*255., img_rect_2*255., 'real_time_model', device)
        disparity2 = np.flip(disparity_from_stereo_raft2d(np.flip(img_rect_2*255., 1), np.flip(img_rect*255., 1), 'real_time_model', device), 1)

        # sharpen disparity to reduce flying pixels in reprojections
        scale = np.maximum( np.amax(-disparity[~mask_cam]), np.amax(-disparity2[~mask_cam]) )      
        img_rect, disparity = sharpen_at_discontinuity(img_rect, disparity, scale, thres=0.025, dk=1)
        img_rect_2, disparity2 = sharpen_at_discontinuity(img_rect_2, disparity2, scale, thres=0.025, dk=1)
        
        # compute depth from disparity
        pointcloud = cv2.reprojectImageTo3D(disparity, Q)
        est_depth = -pointcloud[:,:,2:3]       
        est_depth[mask_cam] = 1e3      
      
        pointcloud2 = cv2.reprojectImageTo3D(disparity2, Q) 
        est_depth_2 = -pointcloud2[:,:,2:3]
        est_depth_2[mask_cam_2] = 1e3      
               
        if flag_visualize:
            vmin_depth, vmax_depth = 0.1, 3.0
            plt.imsave('%s/inverse_depth_left_frame%04d.png' % (output_folder, frame), np.clip(1./(est_depth[:,:,0]+1e-4), a_min=vmin_depth, a_max=vmax_depth), vmin=vmin_depth, vmax=vmax_depth, cmap='inferno')
            plt.imsave('%s/inverse_depth_right_frame%04d.png' % (output_folder, frame), np.clip(1./(est_depth_2[:,:,0]+1e-4), a_min=vmin_depth, a_max=vmax_depth), vmin=vmin_depth, vmax=vmax_depth, cmap='inferno')
        
        # reconstruction each target (eye) view 
        for vv in range(0, len(tgt_view_x)):

            print("   reconstruct view %d..." % vv)
            
            translation_target = np.array([[tgt_view_x[vv], 0., tgt_view_z]]).T

            # view reprojection
            warped_rgbd_torch = view_reprojection(pointcloud, rotation_none, translation_none, translation_target, P1, h, w, mask_cam, img_rect, est_depth)
            warped_rgbd_torch_2 = view_reprojection(pointcloud2, rotation_none, translation_color_2_after_rectification, translation_target, P2, h, w, mask_cam_2, img_rect_2, est_depth_2)

            model.eval()

            with torch.no_grad():

                # hard-threshold warped depth (noticed disoccluded regions have small but non-zero values after softsplat, not ideal)
                eps = 0.1
                warped_rgbd_torch[:,3,:,:] = torch.where(warped_rgbd_torch[:,3,:,:] < eps, 0., warped_rgbd_torch[:,3,:,:].to(torch.double))
                warped_rgbd_torch_2[:,3,:,:] = torch.where(warped_rgbd_torch_2[:,3,:,:] < eps, 0., warped_rgbd_torch_2[:,3,:,:].to(torch.double))
                
                # compute mask of the disoccluded regions that are invisible at both input views 
                mask_disocc = (warped_rgbd_torch[0:1,3:4,:,:] < eps).cpu().detach().numpy()[0,0,:,:]
                mask_disocc_2 = (warped_rgbd_torch_2[0:1,3:4,:,:] < eps).cpu().detach().numpy()[0,0,:,:]
                mask_full_disocc = mask_disocc & mask_disocc_2

                 
                # partial disocclusion filtering
                warped_rgbd_torch, warped_rgbd_torch_2 = run_partial_disocclusion_filter(warped_rgbd_torch, warped_rgbd_torch_2, mask_disocc, mask_disocc_2, device)


                # full disocclusion filtering process        
                warped_rgbd_torch = run_full_disocclusion_filter(warped_rgbd_torch, mask_full_disocc, device)
                warped_rgbd_torch_2 = run_full_disocclusion_filter(warped_rgbd_torch_2, mask_full_disocc, device)


                # run reconstruction network
                result_torch = model(warped_rgbd_torch[:,:3,:,:], warped_rgbd_torch_2[:,:3,:,:])
                result = np.clip(np.transpose(result_torch.cpu().detach().numpy()[0,:,:,:], (1,2,0))/s, a_min=0., a_max=1.)
                save_png_from_numpy(result, "%s/recon_view%04d_frame%04d.png" % (output_folder, vv, frame))   


                # evaluate quality metrics
                if flag_quality_evaluation:
                    gt_image =  np.clip(np.flip(cv2.imread("%s/eye%03d/rgb_frame.%04d.png" % (folder, vv, frame)), 2) / 255., a_min=0., a_max=1.)
                    if flag_visualize:
                        save_png_from_numpy(gt_image, "%s/target_image_left_frame%04d.png" % (output_folder, frame))
                    cs = 32 # crop boundaries before evaluation
                    metric_psnr[vv, frame] = psnr(gt_image[cs:-cs,cs:-cs,:], result[cs:-cs,cs:-cs,:], data_range=1)
                    metric_ssim[vv, frame] = ssim(gt_image[cs:-cs,cs:-cs,:], result[cs:-cs,cs:-cs,:], data_range=1., channel_axis=2) #multichannel=True)
                    print("   current frame PSNR: %.4f" % metric_psnr[vv, frame])
                    print("   current frame SSIM: %.4f" % metric_ssim[vv, frame])


    if flag_quality_evaluation:
        avg_psnr_each_view = np.mean(metric_psnr, axis=1)
        avg_psnr_all = np.mean(metric_psnr)    
        avg_ssim_each_view = np.mean(metric_ssim, axis=1)
        avg_ssim_all = np.mean(metric_ssim)    
        print("\naverage PSNR per view: " , avg_psnr_each_view)
        print("average SSIM per view: " , avg_ssim_each_view)
        print("average PSNR all: " , avg_psnr_all)
        print("average SSIM all: " , avg_ssim_all)

    print('done')


