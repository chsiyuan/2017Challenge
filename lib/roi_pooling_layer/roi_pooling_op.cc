/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// An example Op.

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "work_sharder.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;

REGISTER_OP("RoiPool")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Output("top_data: T")
    .Output("argmax_x: T")
    .Output("argmax_y: T");

REGISTER_OP("RoiPoolGrad")
    .Attr("T: {float, double}")
    .Attr("pooled_height: int")
    .Attr("pooled_width: int")
    .Attr("spatial_scale: float")
    .Input("bottom_data: T")
    .Input("bottom_rois: T")
    .Input("argmax_x: T")
    .Input("argmax_y: T")
    .Input("grad: T")
    .Output("output: T");

template <typename Device, typename T>
class RoiPoolOp : public OpKernel {
 public:
  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_rois_flat = bottom_rois.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->template flat<T>();

    // Tensor* argmax_tensor = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
    // auto argmax = argmax_tensor->template flat<T>();
    Tensor* argmax_x_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_x_tensor));
    auto argmax_x = argmax_x_tensor->template flat<T>();

    Tensor* argmax_y_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, output_shape, &argmax_y_tensor));
    auto argmax_y = argmax_y_tensor->template flat<T>();

    int pooled_height = pooled_height_;
    int pooled_width = pooled_width_;
    float spatial_scale = spatial_scale_;

    auto shard = [pooled_height, pooled_width, spatial_scale,
                  num_rois, batch_size, data_height, data_width, num_channels,
                  &bottom_data_flat, &bottom_rois_flat, &output, &argmax_x, &argmax_y]
                  (int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b)
      {
        // (n, ph, pw, c) is an element in the pooled output
        int n = b;
        int c = n % num_channels;
        n /= num_channels;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;

        // Notes: Get the roi in the feature map
        // For roi align, this step should not use round
        const float* bottom_rois = bottom_rois_flat.data() + n * 5;
        // roi_batch_ind means the index of current roi.
        int roi_batch_ind = bottom_rois[0];

        float roi_start_w = bottom_rois[1] * spatial_scale;
        float roi_start_h = bottom_rois[2] * spatial_scale;
        float roi_end_w = bottom_rois[3] * spatial_scale;
        float roi_end_h = bottom_rois[4] * spatial_scale;

        // Notes: This step is not needed for roi align

        float roi_width = roi_end_w - roi_start_w;
        float roi_height = roi_end_h - roi_start_h;
        const T bin_size_h = static_cast<T>(roi_height)
                           / static_cast<T>(pooled_height);
        const T bin_size_w = static_cast<T>(roi_width)
                           / static_cast<T>(pooled_width);

        // Notes: here these are coordinates of every bin relative to every bbox
        float hstart = static_cast<float>(ph * bin_size_h);
        float wstart = static_cast<float>(pw * bin_size_w);
        float hend = static_cast<float>((ph + 1) * bin_size_h);
        float wend = static_cast<float>((pw + 1) * bin_size_w);

        // Add roi offsets and clip to input boundaries
        hstart = std::min(std::max(hstart + roi_start_h, (float)0), (float) data_height);
        hend = std::min(std::max(hend + roi_start_h, (float)0), (float) data_height);
        wstart = std::min(std::max(wstart + roi_start_w, (float)0), (float) data_width);
        wend = std::min(std::max(wend + roi_start_w, (float)0), (float) data_width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        // Define an empty pooling region to be zero
        float maxval = is_empty ? 0 : -FLT_MAX;
        // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
        float maxidx_x = -1.0;
        float maxidx_y = -1.0;
        const float* bottom_data = bottom_data_flat.data() + roi_batch_ind * num_channels * data_height * data_width;

        // Notes: Get four random points in the bin
        if(!is_empty){
          for (int i = 0; i < 4; ++i)
          {
            float randPoint[2];
            float rh = (rand() % 1000) / 1000.0;
            //rh = 0.5;
	    randPoint[0] = rh * (hend - hstart) + hstart;
            float rw = (rand() % 1000) / 1000.0;
            //rw = 0.3;
	    randPoint[1] = rw * (wend - wstart) + wstart;

            // Notes: Calculate the interpolation for the point
            int topleft[2] = {static_cast<int>(floor(randPoint[0])), static_cast<int>(floor(randPoint[1]))};
            int tl_index = (topleft[0] * data_width + topleft[1]) * num_channels + c;

            int topright[2] = {static_cast<int>(floor(randPoint[0])), static_cast<int>(ceil(randPoint[1]))};
            int tr_index = (topright[0] * data_width + topright[1]) * num_channels + c;

            int botleft[2] = {static_cast<int>(ceil(randPoint[0])), static_cast<int>(floor(randPoint[1]))};
            int bl_index = (botleft[0] * data_width + botleft[1]) * num_channels + c;

            int botright[2] = {static_cast<int>(ceil(randPoint[0])), static_cast<int>(ceil(randPoint[1]))};
            int br_index = (botright[0] * data_width + botright[1]) * num_channels + c;

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
            // std::cout << "rand value " << i << ": " << randValue << std::endl;
            // std::cout << "rand point " << i << ": " << randPoint[0] << " " << randPoint[1] << std::endl;
          }
        }
        output(b) = maxval;
        argmax_x(b) = maxidx_x;
        argmax_y(b) = maxidx_y;
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        num_rois * num_channels * pooled_height * pooled_width * spatial_scale;
    Shard(worker_threads.num_threads, worker_threads.workers,
          output.size(), shard_cost, shard);
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

bool ROIPoolForwardLaucher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, float* argmax_data_x, float* argmax_data_y, const Eigen::GpuDevice& d);

static void RoiPoolingKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois,
    const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape)
{
  Tensor* output = nullptr;
  Tensor* argmax_x = nullptr;
  Tensor* argmax_y = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape, &argmax_x));
  OP_REQUIRES_OK(context, context->allocate_output(2, tensor_output_shape, &argmax_y));

  if (!context->status().ok()) {
    return;
  }

  ROIPoolForwardLaucher(
    bottom_data->flat<float>().data(), spatial_scale, num_rois, height,
    width, channels, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax_x->flat<float>().data(), argmax_y->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
}

template <class T>
class RoiPoolOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  typedef Eigen::GpuDevice Device;

  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    RoiPoolingKernel(context, &bottom_data, &bottom_rois, spatial_scale_, num_rois, data_height,
      data_width, num_channels, pooled_height_, pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

// compute gradient
template <class Device, class T>
class RoiPoolGradOp : public OpKernel {
 public:
  explicit RoiPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data_x = context->input(2);
    const Tensor& argmax_data_y = context->input(3);
    const Tensor& out_backprop = context->input(4);

    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_rois_flat = bottom_rois.flat<T>();
    auto argmax_data_x_flat = argmax_data_x.flat<T>();
    auto argmax_data_y_flat = argmax_data_y.flat<T>();
    auto out_backprop_flat = out_backprop.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data_x.dims() == 4,
                errors::InvalidArgument("argmax_data_x must be 4-dimensional"));

    OP_REQUIRES(context, argmax_data_y.dims() == 4,
                errors::InvalidArgument("argmax_data_y must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int data_height = bottom_data.dim_size(1);
    // data width
    int data_width = bottom_data.dim_size(2);
    // Number of channels
    int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    // Create output tensors
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->template flat<T>();

    int pooled_height = pooled_height_;
    int pooled_width = pooled_width_;
    float spatial_scale = spatial_scale_;

    auto shard = [pooled_height, pooled_width, spatial_scale,
                  num_rois, batch_size, data_height, data_width, num_channels,
                  &bottom_data_flat, &bottom_rois_flat, &argmax_data_x_flat, &argmax_data_y_flat,
                  &out_backprop_flat, &output](int64 start, int64 limit) {
      for (int64 b = start; b < limit; ++b)
      {
        // (n, h, w, c) coords in bottom data
        int n = b;
        int c = n % num_channels;
        n /= num_channels;
        int w = n % data_width;
        n /= data_width;
        int h = n % data_height;
        n /= data_height;

        float gradient = 0.0;
        // Accumulate gradient over all ROIs that pooled this element
        for (int roi_n = 0; roi_n < num_rois; ++roi_n)
        {
          const float* offset_bottom_rois = bottom_rois_flat.data() + roi_n * 5;
          int roi_batch_ind = offset_bottom_rois[0];
          // Skip if ROI's batch index doesn't match n
          if (n != roi_batch_ind) {
            continue;
          }

          // Notes: Calculate the region in feature map that is possible to be used by roi.
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

          int offset = roi_n * pooled_height * pooled_width * num_channels;
          const float* offset_top_diff = out_backprop_flat.data() + offset;
          const float* offset_argmax_data_x = argmax_data_x_flat.data() + offset;
          const float* offset_argmax_data_y = argmax_data_y_flat.data() + offset;

          // Compute feasible set of pooled units that could have pooled
          // this bottom unit
          float roi_width = roi_end_w - roi_start_w;
          float roi_height = roi_end_h - roi_start_h;

          for (int ph = 0; ph < pooled_height; ++ph)
          {
            for (int pw = 0; pw < pooled_width; ++pw)
            {
              float maxidx_x = offset_argmax_data_x[(ph * pooled_width + pw) * num_channels + c];
              float maxidx_y = offset_argmax_data_y[(ph * pooled_width + pw) * num_channels + c];
              // If maxdix_x = maxidx_y = -1, it will skip this [if] branch.
              if(std::abs(maxidx_x - h) < 1 && std::abs(maxidx_y - w) < 1){

                float coeff = (1 - std::abs(maxidx_x - h)) * (1 - std::abs(maxidx_y - w));
                gradient += offset_top_diff[(ph * pooled_width + pw) * num_channels + c] * coeff;
                // std::cout<<"h: "<< h <<" w: "<<w<<" maxidx_x: "<<maxidx_x<<" maxidx_y:"<<maxidx_y<<std::endl;
                // std::cout<<" coeff: "<<coeff<<std::endl;
              }
            }
          }
        }
        output(b) = gradient;
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads =
        *(context->device()->tensorflow_cpu_worker_threads());
    const int64 shard_cost =
        num_rois * num_channels * pooled_height * pooled_width * spatial_scale;
    Shard(worker_threads.num_threads, worker_threads.workers,
          output.size(), shard_cost, shard);
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

bool ROIPoolBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* bottom_diff, const float* argmax_data_x, const float* argmax_data_y, const Eigen::GpuDevice& d);

static void RoiPoolingGradKernel(
    OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_rois, const Tensor* argmax_data_x, 
    const Tensor* argmax_data_y, const Tensor* out_backprop,
    const float spatial_scale, const int batch_size, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const TensorShape& tensor_output_shape)
{
  Tensor* output = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));

  if (!context->status().ok()) {
    return;
  }

  ROIPoolBackwardLaucher(
    out_backprop->flat<float>().data(), spatial_scale, batch_size, num_rois, height,
    width, channels, pooled_height, pooled_width, bottom_rois->flat<float>().data(),
    output->flat<float>().data(), argmax_data_x->flat<float>().data(), argmax_data_y->flat<float>().data(),
    context->eigen_device<Eigen::GpuDevice>());
}


template <class T>
class RoiPoolGradOp<Eigen::GpuDevice, T> : public OpKernel {
 public:
  explicit RoiPoolGradOp(OpKernelConstruction* context) : OpKernel(context) {

    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override
  {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data_x = context->input(2);
    const Tensor& argmax_data_y = context->input(3);
    const Tensor& out_backprop = context->input(4);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data_x.dims() == 4,
                errors::InvalidArgument("argmax_data_x must be 4-dimensional"));

    OP_REQUIRES(context, argmax_data_y.dims() == 4,
                errors::InvalidArgument("argmax_data_y must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    int num_rois = bottom_rois.dim_size(0);
    // batch size
    int batch_size = bottom_data.dim_size(0);
    // data height
    int height = bottom_data.dim_size(1);
    // data width
    int width = bottom_data.dim_size(2);
    // Number of channels
    int channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    RoiPoolingGradKernel(
      context, &bottom_data, &bottom_rois, &argmax_data_x, &argmax_data_y, &out_backprop,
      spatial_scale_, batch_size, num_rois, height, width, channels, pooled_height_,
      pooled_width_, output_shape);

  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RoiPoolGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), RoiPoolGradOp<CPUDevice, float>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("RoiPool").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiPoolOp<Eigen::GpuDevice, float>);
REGISTER_KERNEL_BUILDER(Name("RoiPoolGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), RoiPoolGradOp<Eigen::GpuDevice, float>);
#endif
