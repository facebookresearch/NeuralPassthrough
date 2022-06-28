# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import Image
import torch
from skimage.filters import sobel
from scipy.interpolate import griddata
from scipy.ndimage.morphology import binary_dilation
import cupy

from custombilateral import *
import softsplat


def convert_device(data, device):
    out = [x.to(device, non_blocking=True) for x in data if isinstance(x, torch.Tensor)]
    out += [[x.to(device, non_blocking=True) for x in arr] for arr in data if isinstance(arr, list)]
    return out

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def safe_division(num, denom):
    eps = 1e-8
    tmp = denom + eps * (denom < 1e-20).astype(float)
    return num / tmp

def save_png_from_numpy(x, fn):
    x = x.clip(0,1)
    x = (255.*x).astype('uint8')
    im = Image.fromarray(x)
    im.convert('RGB').save(fn)

def crop_mask_boundary(mask_cam, mask_cam_2, crop_from_left, crop_from_top, crop_from_bottom):
    # mask_cam and mask_cam_2 correspond to left and right camera respectively
    h, w = mask_cam.shape   
    mask_cam[0:crop_from_top+1,:] = 1 # crop top
    mask_cam[h-crop_from_bottom-1:,:] = 1 # crop bottom
    mask_cam[:,:crop_from_left+1] = 1 # crop left
    mask_cam_2[0:crop_from_top+1,:] = 1 # crop top
    mask_cam_2[h-crop_from_bottom-1:,:] = 1 # crop bottom
    mask_cam_2[:,w-crop_from_left-1:] = 1 # crop right 
    return mask_cam, mask_cam_2


def unproject_2d_to_3d(depth_image_undistored_2, mtx_color_2, height, width):
    xs = np.linspace(0, width-1, width, True)
    ys = np.linspace(0, height-1, height, True)
    us, vs = np.meshgrid(xs, ys, sparse=False)
    zs = depth_image_undistored_2[:,:,0] #
    zs_flatten = zs.flatten()

    inv_k = np.linalg.inv(mtx_color_2)
    points_2d = np.stack((us.flatten(), vs.flatten(), np.ones_like(us).flatten()),axis=1)
    points_camera = np.matmul(inv_k, points_2d.T)
    scale = safe_division(zs_flatten, points_camera[2,0])
    points_3d = np.concatenate( (points_camera[0:1,:] * scale, points_camera[1:2,:] * scale, points_camera[2:3,:] * scale), axis=0)

    return points_3d


def transform_3d(points_3d, rotation_color_2, translation_color_2):
    transformed_points_3d = np.matmul(rotation_color_2, points_3d) + translation_color_2
    return transformed_points_3d

def project_3d_to_2d(projected_points_3d, mtx_color, height, width):
    projected_points_3d[0,:] = safe_division(projected_points_3d[0,:], projected_points_3d[2,:])
    projected_points_3d[1,:] = safe_division(projected_points_3d[1,:], projected_points_3d[2,:])
    projected_points_3d[2,:] = 1.0 #projected_points_2d[2,:] / projected_points_2d[2,:]
    projected_points_2d = np.matmul(mtx_color, projected_points_3d)
    projected_points_2d = np.transpose(np.reshape(projected_points_2d, (3, height, width)), (1,2,0))
    return projected_points_2d


def set_softsplat_weight(x):
    # x in diopter unit
    x_min = torch.min(x)
    x_max = torch.max(x)
    y = ((x - x_min) / (x_max - x_min) + 1./9.) * 0.9 * 40
    return y



def view_reprojection(pointcloud, rotation_color, translation_color, translation_target, P1, h, w, mask_cam, img_rect, est_depth):
    
    #point cloud reprojection
    point3d1_x = pointcloud[:,:,0]
    point3d1_y = pointcloud[:,:,1]
    point3d1_z = pointcloud[:,:,2]
    point3d1 =  np.stack((point3d1_x.flatten(), point3d1_y.flatten(), point3d1_z.flatten()), axis=0)

    transformed_point3d1 = transform_3d(point3d1, rotation_color, translation_color)

    transformed_point3d1 = transform_3d(transformed_point3d1, rotation_color, translation_target)

    projected_point2d1 = project_3d_to_2d(transformed_point3d1, P1[:,0:3], h, w)

    # forward flow
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:,:,0] = np.clip(projected_point2d1[:,:,0], a_min=0, a_max = w-1)
    flow[:,:,1] = np.clip(projected_point2d1[:,:,1], a_min=0, a_max = h-1)
            
    # forward warp
    xs = np.linspace(0, w-1, w, True)
    ys = np.linspace(0, h-1, h, True)
    us, vs = np.meshgrid(xs, ys, sparse=False)
    flow[:,:,0] = (flow[:,:,0] - us)
    flow[:,:,1] = (flow[:,:,1] - vs)
    # discard boundary
    flow[:,:,0][mask_cam] = 0 
    flow[:,:,1][mask_cam] = 0 
    img_rect[:,:,0][mask_cam] = 0
    img_rect[:,:,1][mask_cam] = 0
    img_rect[:,:,2][mask_cam] = 0

    
    src_color = img_rect 
    src_color_torch = torch.from_numpy( np.expand_dims(np.transpose(src_color,(2,0,1)), axis=0).astype('float32') )
    flow_torch = torch.from_numpy( np.expand_dims(np.transpose(flow[:,:,:],(2,0,1)), axis=0).astype('float32') )
    src_depth = est_depth[:,:,0:1] 
    src_depth_torch = torch.from_numpy( np.expand_dims(np.transpose(src_depth,(2,0,1)), axis=0).astype('float32') )
    src_disp_torch = 1. / (src_depth_torch + 1e-4)
    src_disp_torch = torch.clip(src_disp_torch, min=0., max=5.0) # ignore if depth is closer than 5 diopters
    weight_torch = set_softsplat_weight(src_disp_torch)
    src_rgbd_torch = torch.cat((src_color_torch, src_disp_torch), dim=1)
    warped_rgbd_torch = softsplat.FunctionSoftsplat(tenInput=src_rgbd_torch.cuda().contiguous(), tenFlow=flow_torch.cuda().contiguous(), tenMetric=weight_torch.cuda().contiguous(), strType='softmax') #

    return warped_rgbd_torch


def run_partial_disocclusion_filter(warped_rgbd_torch, warped_rgbd_torch_2, mask_disocc, mask_disocc_2, device):
                
    mask_disocc = torch.from_numpy(mask_disocc.astype(float))[None, None,:,:].repeat(1,4,1,1).to(device)
    mask_disocc_2 = torch.from_numpy(mask_disocc_2.astype(float))[None, None,:,:].repeat(1,4,1,1).to(device)
    warped_rgbd_torch = torch.mul(warped_rgbd_torch, 1.-mask_disocc) + torch.mul(warped_rgbd_torch_2, mask_disocc)
    warped_rgbd_torch_2 = torch.mul(warped_rgbd_torch_2, 1.-mask_disocc_2) + torch.mul(warped_rgbd_torch, mask_disocc_2)

    return warped_rgbd_torch, warped_rgbd_torch_2

def filter_color_disocclusion_cupy(mask, depth, rgb, inv_var_s, rad, grid, block):
    
    mask = mask.astype(np.float32)
    depth = depth.astype(np.float32)
    rgb = rgb.astype(np.float32)


    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    inv_var_s = np.float32(inv_var_s) 
    rad = np.int32(rad) 
    kernel = cupy.RawKernel(kernel_adaptive_color_bilateral_filter, 'kernel_adaptive_color_bilateral_filter')

    h, w = depth.shape[0], depth.shape[1] 
    v_depth = cupy.asarray(depth)

    v_r = cupy.asarray(r)
    v_g = cupy.asarray(g)
    v_b = cupy.asarray(b)
    v_out_r = cupy.empty_like(v_r)
    v_out_g = cupy.empty_like(v_g)
    v_out_b = cupy.empty_like(v_b)
    v_mask = cupy.asarray(mask)


    kernel(grid, block, (v_out_r, v_out_g, v_out_b, v_r, v_g, v_b, v_depth, w, h, rad, inv_var_s, v_mask))
    
    out = np.zeros((h,w,3), dtype=float)
    out[:,:,0] = cupy.asnumpy(v_out_r)
    out[:,:,1] = cupy.asnumpy(v_out_g)
    out[:,:,2] = cupy.asnumpy(v_out_b)

    return out


def run_full_disocclusion_filter(warped_rgbd_torch, mask_full_disocc, device):
    grid = tuple([40,40]) 
    block = tuple([32,18]) #for image resolution 1280x720
    inv_var_s = 1e-2
    rad = 15 
    depth = warped_rgbd_torch[0,3,:,:].cpu().detach().numpy()
    rgb = np.transpose(warped_rgbd_torch[0,0:3,:,:].cpu().detach().numpy(), (1,2,0))
    out_rgb = filter_color_disocclusion_cupy(mask_full_disocc, depth, rgb, inv_var_s, rad, grid, block)
    warped_rgbd_torch[0,0:3,:,:] = torch.permute(torch.from_numpy(out_rgb).to(device), (2,0,1))   
    
    return warped_rgbd_torch


def sharpen_at_discontinuity(rgb, x, scale, thres=0.01, dk=2):
    # sharpen color and disparity at image edges to reduce flying pixels that would happen after later-on forward warping
    x = -x / scale
    height, width = x.shape[:2]         
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))    

    # 1) detect image edges with sobel filter, and set the edge pixels to zero   
    edges = sobel(x) > thres
    dilate_kernel = np.ones((dk,dk)) 
    edges = binary_dilation(edges, dilate_kernel, iterations=1)
    x[edges] = 0.
    mask = x > 0

    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    r[edges] = 0
    g[edges] = 0
    b[edges] = 0

    # 2) interpolate the edge pixel to be its nearest-neighbor non-edge pixel's value
    # at C++ inference optimization, this process is only done at the edge pixels; but in the python code here, it's done at all pixels causing computation overhead.
    try:
        x = griddata(np.stack([ys[mask].ravel(), xs[mask].ravel()], 1),
                    x[mask].ravel(), np.stack([ys.ravel(), xs.ravel()], 1),
                    method='nearest').reshape(height, width)
        r = griddata(np.stack([ys[mask].ravel(), xs[mask].ravel()], 1),
                    r[mask].ravel(), np.stack([ys.ravel(), xs.ravel()], 1),
                    method='nearest').reshape(height, width)
        g = griddata(np.stack([ys[mask].ravel(), xs[mask].ravel()], 1),
                    g[mask].ravel(), np.stack([ys.ravel(), xs.ravel()], 1),
                    method='nearest').reshape(height, width)
        b = griddata(np.stack([ys[mask].ravel(), xs[mask].ravel()], 1),
                    b[mask].ravel(), np.stack([ys.ravel(), xs.ravel()], 1),
                    method='nearest').reshape(height, width)
    except (ValueError, IndexError) as e:
        pass  # just return original values
    x = -x * scale

    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b

    return rgb, x


def get_intrinsics(w, h, fov_x):
    f_x = w/(2.*np.tan(fov_x*np.pi/360.))
    f_y = f_x 
    intrinsics = np.array([[f_x, 0, w/2],
                           [0, f_y, h/2],
                           [0, 0, 1]])
    return intrinsics