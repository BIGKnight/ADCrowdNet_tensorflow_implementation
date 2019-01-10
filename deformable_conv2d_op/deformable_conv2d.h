#ifndef KERNEL_DEFORMABLE_CONV_2D_H_
#define KERNEL_DEFORMABLE_CONV_2D_H_
// #include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/tensor_types.h"
// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// #include "iostream"
// #include <cstdint>
// #include <tuple>
// #include <limits>
// #include <iostream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstring>
#include <vector>

namespace tensorflow{
typedef std::vector<int> TShape;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

inline int ProdShape(const TShape &shape, int start, int end) {
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}    

inline int ProdShape(const TensorShape& shape, int start, int end){
    int res = 1;
    for(int i=start; i<end; i++) {
        res*=shape.dim_size(i);
    }
    return res;
}

template <typename Device, typename DType>
struct pureAddTo {
    void operator() (const Device& d, const int n, DType* result_data, const DType* right_data);
};

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

struct DeformableConv2DParameters {
  TShape dilations;
  TShape strides;
  Padding padding;
  int32_t num_groups;
  int32_t deformable_groups;
  int32_t im2col_step;
  bool no_bias;
  TensorFormat data_format;
};

// Convolution dimensions inferred from parameters, input and filter tensors.
struct DeformableConv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int out_rows;
  int out_cols;
  int pad_rows;
  int pad_cols;
};

template <typename DType>
inline void DeformableConv2DCol2ImCoord(const CPUDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation, const int32_t deformable_group,
  DType* grad_offset, DType* grad_mask){
    LOG(FATAL) << "only implemented in GPU";
  }

template <typename DType>
inline void SwapAxis(const CPUDevice& d, DType* input_data, const TShape& origin_shape, const int axis_x, const int axis_y){
    LOG(FATAL) << "only implemented in GPU";
} 

template <typename DType>
inline void DeformableConv2DCol2Im(const CPUDevice& d, 
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im){
        LOG(FATAL) << "only implemented in GPU";
    }

template <typename DType>
inline void DeformableConv2DIm2Col(const CPUDevice& d, 
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col){
        if (2 == kernel_shape.size()) {
        LOG(FATAL) << "only implemented in GPU";
        } else {
        LOG(FATAL) << "not implemented";
        }
    }


#if GOOGLE_CUDA == 1
template <typename DType>
inline void DeformableConv2DCol2ImCoord(const GPUDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation, const int32_t deformable_group,
  DType* grad_offset, DType* grad_mask);


template <typename DType>
inline void DeformableConv2DCol2Im(const GPUDevice& d, 
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const int32_t deformable_group,
    DType* grad_im);


template <typename DType>
inline void DeformableConv2DIm2Col(const GPUDevice& d, 
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const int32_t deformable_group, DType* data_col);
#endif

template <typename Device, typename DType>
struct setZero {
    void operator() (const Device& d, int n, DType* result_data);
};

#if GOOGLE_CUDA == 1
template <typename DType>
inline void SwapAxis(const GPUDevice& d, DType*& input_data, const TShape& origin_shape, const int axis_x, const int axis_y);
#endif
}
#endif