# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import cupy



kernel_adaptive_color_bilateral_filter = '''
extern "C" __global__ void kernel_adaptive_color_bilateral_filter(
        float* outputR,
        float* outputG,
        float* outputB,
		const float* inputR,
        const float* inputG,
        const float* inputB,
		const float* guidance,
		const int width,
        const int height,
        const int rad,
        const float inv_var_s, 
        const float* mask )
{ 
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x>=width) || (y>=height))
        return;

    const int idx_center = y * width + x; 

  
    // note: can consider only process the black hole pixels
    if (!mask[idx_center])
    {
        outputR[idx_center] = inputR[idx_center];
        outputG[idx_center] = inputG[idx_center];
        outputB[idx_center] = inputB[idx_center];
    }
    else
    {  
        const int w_up = (y-rad+1>0)?(y-rad+1):(0);
        const int w_down = ((y+rad-1)<(height-1))?(y+rad-1):(height-1);
        const int w_left = (x-rad+1>0)?(x-rad+1):(0);
        const int w_right = ((x+rad-1)<(width-1))?(x+rad-1):(width-1);

        // get the min and max of the non-zero depth values in local neighborhood (defined by rad) of each pixel
        // the result will be used to define the parameters of bilateral filters
        float min_depth_local = 1000.; // diopter (inverse depth) actually
        float max_depth_local = 0.;
        for(int v = w_up; v <= w_down; v++)
          for(int u = w_left; u <= w_right; u++)
           {
               const int idx = v * width + u;
               if (guidance[idx] > 0.1) // skip zero-depth pixels
               {
                   if(guidance[idx] < min_depth_local)
                      min_depth_local = guidance[idx];
                   if(guidance[idx] > max_depth_local)
                      max_depth_local = guidance[idx];              
               }               
            }



        float acc_resultR = 0;
        float acc_resultG = 0;
        float acc_resultB = 0;
        float acc_weight = 0;


        for(int v = w_up; v <= w_down; v++)
          for(int u = w_left; u <= w_right; u++)
           {
            const int idx = v * width + u;
            if (guidance[idx] < 0.1) 
               continue;

            if (guidance[idx] > 0.5*(max_depth_local + min_depth_local)) //debug
               continue;

            const float weight_s = -( (x-u)*(x-u) + (y-v)*(y-v) ) * inv_var_s;
            const float weight = expf( weight_s);

            acc_resultR = acc_resultR + inputR[idx] * weight;
            acc_resultG = acc_resultG + inputG[idx] * weight;
            acc_resultB = acc_resultB + inputB[idx] * weight;
            acc_weight = acc_weight + weight;
            } 
  
        if (acc_weight < 1e-5)
          {
            outputR[idx_center] = inputR[idx_center];
            outputG[idx_center] = inputG[idx_center];
            outputB[idx_center] = inputB[idx_center];
          }
        else
          {
            outputR[idx_center] = acc_resultR / acc_weight;
            outputG[idx_center] = acc_resultG / acc_weight;
            outputB[idx_center] = acc_resultB / acc_weight;
          }
    }

}
'''

