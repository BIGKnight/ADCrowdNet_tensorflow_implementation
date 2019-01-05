#ifndef KERNEL_DEFORMABLE_CONV_2D_H_
#define KERNEL_DEFORMABLE_CONV_2D_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "iostream"
#include <cstdint>
#include <tuple>
#include <limits>
#include <iostream>

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using namespace tensorflow;
typedef std::vector<int32> TShape;
// Convolution parameters specified by Op attributes.

inline int ProdShape(const TShape &shape, int start, int end) {
    int64 res = 1;
    for(int i=start; i<end; i++) {
        res*=shape[i];
    }
    return res;
}    

inline int ProdShape(const TensorShape &shape, int start, int end) {
    int64 res = 1;
    for(int i=start; i<end; i++) {
        res*=shape.dim_size(i);
    }
    return res;
}    

inline std::vector<int> ToVector(const TensorShape &shape) {
    // int64 res = 1;
    std::vector<int> res;
    for(int i=0; i<shape.dims(); i++) {
        res.push_back(shape.dim_size(i));
    }
    return res;
}

inline TShape ToVector(const TShape &shape) {
    // int64 res = 1;
    return shape;
}

inline TShape SubVector(cons TShape &shape, int start, int end){
    TShape res;
    for(int i = start;i<end;i++){
        res.push_back(shape[i])
    }
    return res;
} 

inline TShape SubVector(cons TensorShape &shape, int start, int end){
    TShape res;
    for(int i = start;i<end;i++){
        res.push_back(shape.dim_size(i))
    }
    return res;
} 

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

  int64 out_rows;
  int64 out_cols;
  int64 pad_rows;
  int64 pad_cols;
};

template <typename DType>
inline void DeformableConv2DCol2ImCoord(CPUDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation, const uint32_t deformable_group,
  DType* grad_offset, DType* grad_mask){
    LOG(FATAL) << "only implemented in GPU";
  }

template <typename DType>
inline void DeformableConv2DCol2Im(CPUDevice& d, 
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const uint32_t deformable_group,
    DType* grad_im){
        LOG(FATAL) << "only implemented in GPU";
    }

template <typename DType>
inline void DeformableConv2DIm2Col(CPUDevice& d, 
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const uint32_t deformable_group, DType* data_col){
        if (2 == kernel_shape.ndim()) {
        LOG(FATAL) << "only implemented in GPU";
        } else {
        LOG(FATAL) << "not implemented";
        }
    }


#if GOOGLE_CUDA == 1
template <typename DType>
inline void DeformableConv2DCol2ImCoord(GPUDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation, const uint32_t deformable_group,
  DType* grad_offset, DType* grad_mask);


template <typename DType>
inline void DeformableConv2DCol2Im(GPUDevice& d, 
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const uint32_t deformable_group,
    DType* grad_im);


template <typename DType>
inline void DeformableConv2DIm2Col(GPUDevice& d, 
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const uint32_t deformable_group, DType* data_col);
#endif

#endif