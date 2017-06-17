#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include "roi_pooling_op_gpu.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

using std::max;
using std::min;
using std::abs;

// namespace tensorflow {
using namespace tensorflow;

template <typename Dtype>
__global__ void ROIPoolForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int height, const int width, 
    const int channels, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data, float* argmax_data_x, float* argmax_data_y) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, ph, pw, c) is an element in the pooled output
    int n = index;
    int c = n % channels;
    n /= channels;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    float roi_start_w = bottom_rois[1] * spatial_scale;
    float roi_start_h = bottom_rois[2] * spatial_scale;
    float roi_end_w = bottom_rois[3] * spatial_scale;
    float roi_end_h = bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    float roi_width = roi_end_w - roi_start_w;
    float roi_height = roi_end_h - roi_start_h;
    Dtype bin_size_h = static_cast<Dtype>(roi_height)
                       / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)
                       / static_cast<Dtype>(pooled_width);

    float hstart = static_cast<float>(static_cast<Dtype>(ph) * bin_size_h);
    float wstart = static_cast<float>(static_cast<Dtype>(pw) * bin_size_w);
    float hend = static_cast<float>(static_cast<Dtype>(ph + 1) * bin_size_h);
    float wend = static_cast<float>(static_cast<Dtype>(pw + 1) * bin_size_w);
    // int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
    //                                     * bin_size_h));
    // int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
    //                                     * bin_size_w));
    // int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1)
    //                                  * bin_size_h));
    // int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1)
    //                                  * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, (float)0), (float) height);
    hend = min(max(hend + roi_start_h, (float)0), (float) height);
    wstart = min(max(wstart + roi_start_w, (float)0), (float) width);
    wend = min(max(wend + roi_start_w, (float)0), (float) width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    float maxidx_x = -1.0;
    float maxidx_y = -1.0;
    bottom_data += roi_batch_ind * channels * height * width;
    // for (int h = hstart; h < hend; ++h) {
    //   for (int w = wstart; w < wend; ++w) {
    //     int bottom_index = (h * width + w) * channels + c;
    //     if (bottom_data[bottom_index] > maxval) {
    //       maxval = bottom_data[bottom_index];
    //       maxidx = bottom_index;
    //     }
    //   }
    // }
    curandState_t state;
    curand_init(clock64(), 0, 0, &state);

    if(!is_empty){
      for (int i = 0; i < 4; ++i)
      {
        float randPoint[2];
        float rh = static_cast<float>((curand(&state)%1000)/1000.0);
        //rh = 0.5;
	randPoint[0] = rh * (hend - hstart) + hstart;
        float rw = static_cast<float>((curand(&state)%1000)/1000.0);
        //rw = 0.3;
	randPoint[1] = rw * (wend - wstart) + wstart;

        // Notes: Calculate the interpolation for the point
        int topleft[2] = {static_cast<int>(floor(randPoint[0])), static_cast<int>(floor(randPoint[1]))};
        int tl_index = (topleft[0] * width + topleft[1]) * channels + c;

        int topright[2] = {static_cast<int>(floor(randPoint[0])), static_cast<int>(ceil(randPoint[1]))};
        int tr_index = (topright[0] * width + topright[1]) * channels + c;

        int botleft[2] = {static_cast<int>(ceil(randPoint[0])), static_cast<int>(floor(randPoint[1]))};
        int bl_index = (botleft[0] * width + botleft[1]) * channels + c;

        int botright[2] = {static_cast<int>(ceil(randPoint[0])), static_cast<int>(ceil(randPoint[1]))};
        int br_index = (botright[0] * width + botright[1]) * channels + c;

        rh = randPoint[0]-topleft[0];
        rw = randPoint[1]-topleft[1];
        // std::cout<<"rh: "<<rh<<" rw: "<<rw<<std::endl;

        float randValue = (1-rh) * (1-rw) * bottom_data[tl_index]
                      + (1-rh) * rw * bottom_data[tr_index]
                      + rh * (1-rw) * bottom_data[bl_index]
                      + rh * rw * bottom_data[br_index];
        if(randValue > maxval){
          maxval = randValue;
          maxidx_x = randPoint[0];
          maxidx_y = randPoint[1];
        }
        // printf("i: %d, rand point: [%f,%f]\n", i, randPoint[0],randPoint[1]);
        // printf("i: %d, rand value: %f\n", i, randValue);
      }
    }

    top_data[index] = maxval;
    if (argmax_data_x != nullptr && argmax_data_y != nullptr){
      argmax_data_x[index] = maxidx_x;
      argmax_data_y[index] = maxidx_y;
    }
  }
}

bool ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, float* argmax_data_x, float* argmax_data_y, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  cudaError_t err;

  ROIPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_rois, top_data, argmax_data_x, argmax_data_y);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}


template <typename Dtype>
__global__ void ROIPoolBackward(const int nthreads, const Dtype* top_diff,
    const float* argmax_data_x, const float* argmax_data_y, const int num_rois, const Dtype spatial_scale,
    const int height, const int width, const int channels, 
    const int pooled_height, const int pooled_width, Dtype* bottom_diff,
    const Dtype* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (n, h, w, c) coords in bottom data
    int n = index;
    int c = n % channels;
    n /= channels;
    int w = n % width;
    n /= width;
    int h = n % height;
    n /= height;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) 
    {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = static_cast<int>(floor(offset_bottom_rois[1] * spatial_scale));
      int roi_start_h = static_cast<int>(floor(offset_bottom_rois[2] * spatial_scale));
      int roi_end_w = static_cast<int>(ceil(offset_bottom_rois[3] * spatial_scale));
      int roi_end_h = static_cast<int>(ceil(offset_bottom_rois[4] * spatial_scale));

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                           h >= roi_start_h && h <= roi_end_h);
      if (!in_roi) {
        continue;
      }

      int offset = roi_n * pooled_height * pooled_width * channels;
      const Dtype* offset_top_diff = top_diff + offset;
      const float* offset_argmax_data_x = argmax_data_x + offset;
      const float* offset_argmax_data_y = argmax_data_y + offset;

      for (int ph = 0; ph < pooled_height; ++ph)
      {
        for (int pw = 0; pw < pooled_width; ++pw)
        {
          float maxidx_x = offset_argmax_data_x[(ph * pooled_width + pw) * channels + c];
          float maxidx_y = offset_argmax_data_y[(ph * pooled_width + pw) * channels + c];
          // If maxdix_x = maxidx_y = -1, it will skip this [if] branch.
          if(abs(maxidx_x - h) < 1 && abs(maxidx_y - w) < 1){
              float coeff = (1 - std::abs(maxidx_x - h)) * (1 - std::abs(maxidx_y - w));
              gradient += offset_top_diff[(ph * pooled_width + pw) * channels + c] * coeff;
              // printf("h: %d, w: %d, maxidx_x: %f, maxidx_y: %f\n", h,w,maxidx_x,maxidx_y);
              // printf("coeff: %f\n", coeff);
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}


bool ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const float* argmax_data_x, const float* argmax_data_y, const Eigen::GpuDevice& d) 
{
  const int kThreadsPerBlock = 1024;
  const int output_size = batch_size * height * width * channels;
  cudaError_t err;

  ROIPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                       kThreadsPerBlock, 0, d.stream()>>>(
      output_size, top_diff, argmax_data_x, argmax_data_y, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, bottom_diff, bottom_rois);

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  return d.ok();
}

// }  // namespace tensorflow

#endif  // GOOGLE_CUDA
