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

using tensorflow::TTypes;
using tensorflow::OpKernelContext;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GPUDevice;

using KDimSize = Eigen::
#endif