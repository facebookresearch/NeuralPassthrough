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

from utils import *
from custombilateral import *
from stereo_depth import *
from net import NPTnet 



if __name__ == "__main__":
    
    ## select test data
    folder_proto = "%s/data/test/real/" % (str(Path(sys.path[0]).parent))  
    
    ###########################################################
    # select scene
    scene, num_frames = 'bookstatic', 1
    #scene, num_frames = 'personvideo', 150
    #scene, num_frames = 'desktopvideo', 200 
    ###########################################################
    
    print("test scene: %s, frames: %d" % (scene, num_frames))
    s = 1.0 if scene=='bookstatic' else 0.5 # temporary image intensity scaling parameter, it appears that image intensity can affect depth estimation quality sometimes
    crop_from_left, crop_from_top, crop_from_bottom = (0,0,0) if scene=='bookstatic' else (110, 85, 55) # image cropping parameters, cropping due to inaccurate depth estimation at image boundaries 
    
    folder = "%s/%s/" % (folder_proto, scene)
    output_folder = "%s/output-%s/" % (str(Path(sys.path[0]).parent), scene)
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
    mtx_color = np.array([[6.2332931772538961e+02, 0., 6.3888145428055736e+02], \
    [0., 6.1953917327840998e+02, 3.7066713576436177e+02], \
    [0., 0., 1.]])

    dis_color = np.array([ -5.0697015067885678e+00, 6.1600339159069026e+00,\
    2.5148074413211077e-05, -3.6152071916631872e-04,\
    9.7464488594834560e+00, -5.1645013593153148e+00,\
    6.7119689279310357e+00, 8.7058302855374414e+00])

    rotation_none = np.array([[1., 0., 0.],\
    [0., 1., 0.],\
    [0., 0., 1.]])

    translation_none = np.array([[0.,0.,0.]]).T

    # cam 2 (right)
    mtx_color_2 = np.array([[6.2768896934116754e+02, 0., 6.4084635563890026e+02], \
    [0.,6.2425635328270471e+02, 3.5200413881051998e+02], \
    [0., 0., 1.]])

    dis_color_2 = np.array([-5.4929986798949635e+00, 9.2187276816481525e+00, \
    -6.7275804390598624e-04, 3.2580906779884947e-04, \
    1.1097712101258605e+01, -5.5869316321111366e+00, \
    9.7952782193306707e+00, 9.8010110427594608e+00])       
        
    rotation_color_2 = np.array([[0.9999874032375143, -0.004658404957022066, -0.0018688578192557017],\
    [0.004665694206197396, 0.9999814512234607, 0.003915163674180608],\
    [0.001850584736362267, -0.003923833874893243, 0.9999905893876482]])

    translation_color_2 = np.array([[-0.09891927641394618, 0.0018008355990449393, 0.0029031303437694334]]).T


    ## stereo rectification parameters
    h, w = 720, 1280
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

    

    ## run neural passthrough method
    for frame in range(0,num_frames,1):
        print("process frame %03d..." % frame)
  
        # read input stereo mages
        color_image = np.flip(cv2.imread("%s/cam%d_frame%04d.png" % (folder, 1, frame)), 2) * s / 255.
        color_image_2 = np.flip(cv2.imread("%s/cam%d_frame%04d.png" % (folder, 2, frame)), 2) * s / 255.
                
        # run stereo rectification
        img_rect = cv2.remap(color_image, mapx1, mapy1, cv2.INTER_LINEAR)
        img_rect_2 = cv2.remap(color_image_2, mapx2, mapy2, cv2.INTER_LINEAR)
        if flag_visualize:
            save_png_from_numpy(img_rect/s, "%s/rectified_image_left_frame%04d.png" % (output_folder, frame))
            save_png_from_numpy(img_rect_2/s, "%s/rectified_image_right_frame%04d.png" % (output_folder, frame))

        # load precomputed mask for each camera that indidate black boundary regions due to stereo rectification
        mask_cam = np.array(cv2.imread("%s/mask_cam%d.png" % (folder_proto, 1))[:,:,0], dtype=bool)
        mask_cam_2 = np.array(cv2.imread("%s/mask_cam%d.png" % (folder_proto, 2))[:,:,0], dtype=bool)
        # for additional boundary cropping
        mask_cam, mask_cam_2 = crop_mask_boundary(mask_cam, mask_cam_2, crop_from_left, crop_from_top, crop_from_bottom)    

        # stereo depth estimation
        print("   run stereo depth estimation...")
        disparity = disparity_from_stereo_raft2d(img_rect*255., img_rect_2*255., 'real_time_model', device)
        disparity2 = np.flip(disparity_from_stereo_raft2d(np.flip(img_rect_2*255., 1), np.flip(img_rect*255., 1), 'real_time_model', device), 1)

        # sharpen disparity to reduce flying pixels in reprojections
        scale = np.maximum( np.amax(-disparity[~mask_cam]), np.amax(-disparity2[~mask_cam]) )
        img_rect, disparity = sharpen_at_discontinuity(img_rect, disparity, scale)
        img_rect_2, disparity2 = sharpen_at_discontinuity(img_rect_2, disparity2, scale)
        
        # compute depth from disparity
        pointcloud = cv2.reprojectImageTo3D(disparity, Q)
        est_depth = -pointcloud[:,:,2:3]       
        est_depth[mask_cam] = 1e3      
      
        pointcloud2 = cv2.reprojectImageTo3D(disparity2, Q) 
        est_depth_2 = -pointcloud2[:,:,2:3]
        est_depth_2[mask_cam_2] = 1e3      
               
        if flag_visualize:
            vmin_depth, vmax_depth = 0.1, 1.5
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
                 

           
    print('done')


