#ifndef DEFORMABLECONV2D_KERNEL_OPS_GPU_H_
#define DEFORMABLECONV2D_KERNEL_OPS_GPU_H_
#if GOOGLE_CUDA == 1
#define EIGEN_USE_GPU
#include "deformable_conv2d.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow{

typedef Eigen::GpuDevice GPUDevice;
typedef std::vector<int32> TShape;

// define the cuda kernel
template<typedef DType>
__device__ DType dmcn_im2col_bilinear(
    const DType* bottom_data,
    const int32_t data_width,
    const int32_t height,
    const int32_t width,
    DType h,
    DType w){
        int32_t h_low = floor(h);
        int32_t w_low = floor(w);
        int32_t h_high = h_low + 1;
        int32_t w_high = w_low + 1;
        DType lh = h - h_low;
        DType lw = w - w_low;
        DType hh = 1 - lh, hw = 1 - lw;
        DType v1 = 0;
        if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low * data_width + w_low];
        DType v2 = 0;
        if (h_low >=0 && w_high <= width - 1) v2 = bottom_data[h_low * data_width + w_high];
        DType v3 = 0;
        if (h_high <= height - 1 && w_low >= 0) v3 = bottom_data[h_high * data_width + w_low];
        DType v4 = 0;
        if (h_high <= height - 1 && w_high <= width - 1) v4 = bottom_data[h_high * data_width + w_high]
        DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
        DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
        return val;
}

template<typedef DType>
__device__ DType dmcn_get_gradient_weight(
    DType argmax_h,　// offset h
    DType argmax_w,　// offset w
    const int32_t h,  const int32_t w, // coordinate
    const int32_t height,  const int32_t width){
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    //empty
    return 0;
  }
  int32_t argmax_h_low = floor(argmax_h);
  int32_t argmax_w_low = floor(argmax_w);
  int32_t argmax_h_high = argmax_h_low + 1;
  int32_t argmax_w_high = argmax_w_low + 1;
  DType weight = 0;
  if (h == argmax_h_low && w == argmax_w_low) weight = (h + 1 - argmax_h) * (w + 1 - argmax_w); //1 - (argmax - h)
  if (h == argmax_h_low && w == argmax_w_high) weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low) weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high) weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}

template <typename DType>
__device__ DType dmcn_get_coordinate_weight(
    DType argmax_h,
    DType argmax_w,
    const int32_t height,
    const int32_t width,
    const DType* im_data,
    const int32_t data_width,
    const int32_t bp_dir
    ) {

  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width)
  {
    //empty
    return 0;
  }

  int32_t argmax_h_low = floor(argmax_h);
  int32_t argmax_w_low = floor(argmax_w);
  int32_t argmax_h_high = argmax_h_low + 1;
  int32_t argmax_w_high = argmax_w_low + 1;
  
  DType weight = 0;

  if (bp_dir == 0) {　//先ｘ方向 , 这个负号是把ｙ放向上的(argmax_h_high - argmax_h)替换成了1 - (argmax_h - argmax_h_low)
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_low * data_width + argmax_w_low];

    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += -1 * (argmax_w - argmax_w_low) * im_data[argmax_h_low * data_width + argmax_w_high];

    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += (argmax_w_low + 1 - argmax_w) * im_data[argmax_h_high * data_width + argmax_w_low];

    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_w - argmax_w_low) * im_data[argmax_h_high * data_width + argmax_w_high];

  }

  else if (bp_dir == 1) {　//先ｙ放向
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
        weight += -1 * (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_low];

    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
        weight += (argmax_h_low + 1 - argmax_h) * im_data[argmax_h_low * data_width + argmax_w_high];

    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
        weight += -1 * (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_low];

    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
        weight += (argmax_h - argmax_h_low) * im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename DType>
__global__ void DeformableConv2DIm2ColKernel(
    const int32_t n,  
    const DType* data_im,
    const DType* data_offset,
    const DType* data_mask,

    const int32_t height,　const int32_t width,
    const int32_t kernel_h,　const int32_t kernel_w,
    const int32_t pad_h,　const int32_t pad_w,
    const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_h, const int32_t dilation_w,

    const int32_t channel_per_deformable_group, // 输入图通道数除以deformable_group的数量,
    const int32_t batch_size, const int32_t num_channels, const int32_t deformable_group, //这里的batch_size代表的是im2col_step_, 一般就设为1了
    const int32_t height_col, const int32_t width_col, 
    DType* data_col){
    CUDA_1D_KERNEL_LOOP(index, n){
    // n = K * N / k.Size(), 这里应该是一个线程的运算内容, 所以所谓的卷积就是并行计算kernel喽, 所以一个filter得出一个通道的输出,需要n个kernel操作
    // index index of output matrix
    const int32_t w_col = index % width_col;
    const int32_t h_col = (index / width_col) % height_col;

    const int32_t b_col = (index / width_col / height_col) % batch_size; // 为什么这个地方有batch的信息. 我感觉b_col代表的是在第几个通道

    const int32_t c_im = (index / width_col / height_col) / batch_size; // batch_size 可以暂时看做1

    const int32_t c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int32_t deformable_group_index = c_im / channel_per_deformable_group; // 0

    const int32_t h_in = h_col * stride_h - pad_h; // 这是在计算h_col在输入图上感受野的左上角的位置h_in
    const int32_t w_in = w_col * stride_w - pad_w; // 这是在计算w_col在输入图上感受野的左上角的位置w_in


    // 这里在计算指针的位置
    DType* data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;

    const DType* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width; // 找到这张图当前kernel所在的那个输入层的初始像素点的位置, 如果batch_size = 1, b_col = 0 那么就等于data_im + (ci_m * height * width)
    const DType* data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col; // 找到
    const DType* data_mask_ptr = data_mask + (b_col *  deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
        for (int i = 0; i < kernel_h; ++i) { // 这里貌似kernel大小就是[3, 3]不考虑输入通道的数目, 那确实印证了之前的说法, CUDA_KERNEL_LOOP其实代表的就是一个kernel操作
            for (int j = 0; j < kernel_w; ++j) {
                const int32_t data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col; //2 * (i * kernel_w + j)代表的是找到对应的通道, 后半部分在找坐标点
                // 其实这个可以写作 2 * (i * kernel_w + j) * height_col * width_col + h_col * width_col + w_col
                const int32_t data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
                const int32_t data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
                // 取出偏移和mask
                const DType offset_h = data_offset_ptr[data_offset_h_ptr];
                const DType offset_w = data_offset_ptr[data_offset_w_ptr];
                const DType mask = data_mask_ptr[data_mask_hw_ptr];

                DType val = static_cast<DType>(0);
                const DType h_im = h_in + i * dilation_h + offset_h; // 计算加了偏移后的在输入图上的坐标
                const DType w_im = w_in + j * dilation_w + offset_w;
                //if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
                if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
                  //const DType map_h = i * dilation_h + offset_h;
                  //const DType map_w = j * dilation_w + offset_w;
                  //const int32_t cur_height = height - h_in;
                  //const int32_t cur_width = width - w_in;
                  //val = dmcn_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
                  val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im); // 计算插值
                }
                *data_col_ptr = val * mask; // 值得注意的是,此时只是将偏移后的值算出来了, 而且注意这个地方data_col_ptr这个指针是直接往下走了N,而这个循环实际上只走w * h步, 那么这个循环结束的时候，只往Ｋ的那个方向填了kernel.Size()个值
                data_col_ptr += batch_size * height_col * width_col;
                //data_col_ptr += height_col * width_col;
            }
        }    
    }
}d

template <typename DType>
__global__ void DeformableConv2DCol2ImKernel(
    const int32_t n, 
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const int32_t channels, const int32_t height, const int32_t width,
    const int32_t kernel_h, const int32_t kernel_w,
    const int32_t pad_h, const int32_t pad_w,
    const int32_t stride_h, const int32_t stride_w,
    const int32_t dilation_h, const int32_t dilation_w,
    const int32_t channel_per_deformable_group,
    const int32_t batch_size, const int32_t deformable_group,
    const int32_t height_col, const int32_t width_col,
    DType* grad_im){
    CUDA_1D_KERNEL_LOOP(index, n){
        const int32_t j = (index / width_col / height_col / batch_size) % kernel_w;
        const int32_t i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
        const int32_t c = index / width_col / height_col / batch_size / kernel_w / kernel_h;
        // compute the start and end of the output
        const int32_t deformable_group_index = c / channel_per_deformable_group;

        int32_t w_out = index % width_col;
        int32_t h_out = (index / width_col) % height_col;
        int32_t b = (index / width_col / height_col) % batch_size;
        int32_t w_in = w_out * stride_w - pad_w;
        int32_t h_in = h_out * stride_h - pad_h;

        const DType* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
        const DType* data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
        const int32_t data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
        const int32_t data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
        const int32_t data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        const DType mask = data_mask_ptr[data_mask_hw_ptr];
        const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
        const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

        const DType cur_top_grad = data_col[index] * mask;
        const int32_t cur_h = (int32_t)cur_inv_h_data;
        const int32_t cur_w = (int32_t)cur_inv_w_data;
        for (int32_t dy = -2; dy <= 2; dy++) {
        for (int32_t dx = -2; dx <= 2; dx++) {
            if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1
            ) {
                int32_t cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
                DType weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
                CudaAtomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
                }
            }
        }
    }
}

/*!
 * \brief deformable_col2im_coord gpu kernel.
 * \brief DO NOT call this directly. Use wrapper function  instead;
 */
template <typename DType>
__global__ void DeformableConv2DCol2ImCoordGPUKernel(
  const int32_t n, 
  const DType* data_col, const DType* data_im,
  const DType* data_offset, const DType* data_mask,
  const int32_t channels, const int32_t height, const int32_t width, // 输入的C, H, W
  const int32_t kernel_h, const int32_t kernel_w,
  const int32_t pad_h, const int32_t pad_w,
  const int32_t stride_h, const int32_t stride_w,
  const int32_t dilation_h, const int32_t dilation_w,
  const int32_t channel_per_deformable_group,
  const int32_t batch_size, const int32_t offset_channels, const int32_t deformable_group,
  const int32_t height_col, const int32_t width_col,
  DType* grad_offset, DType* grad_mask) {
  CUDA_1D_KERNEL_LOOP(index, n){
    DType val = 0, mval = 0;
    int32_t w = index % width_col;
    int32_t h = (index / width_col) % height_col;
    int32_t c = (index / width_col / height_col) % offset_channels;
    int32_t b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int32_t deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int32_t col_step = kernel_h * kernel_w;
    int32_t cnt = 0;
    const DType* data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const DType* data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group / kernel_h / kernel_w * height * width;
    const DType* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const DType* data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    const int32_t offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int32_t col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int32_t bp_dir = offset_c % 2;

      int32_t j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int32_t i = (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int32_t w_out = col_pos % width_col;
      int32_t h_out = (col_pos / width_col) % height_col;
      int32_t w_in = w_out * stride_w - pad_w;
      int32_t h_in = h_out * stride_h - pad_h;
      const int32_t data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int32_t data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int32_t data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      const DType mask = data_mask_ptr[data_mask_hw_ptr];
      DType inv_h = h_in + i * dilation_h + offset_h;
      DType inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_ptr[col_pos] * dmcn_im2col_bilinear(data_im_ptr + cnt * height * width, width, height, width, inv_h, inv_w);
      }
      const DType weight = dmcn_get_coordinate_weight(
        inv_h, inv_w,
        height, width, data_im_ptr + cnt * height * width, width, bp_dir);
      val  += weight * data_col_ptr[col_pos] * mask;    
      cnt  += 1;
    }

    grad_offset[index] = val;
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    if (offset_c % 2 == 0){
            grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
            // KERNEL_ASSIGN(grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w], mask_req, mval);
        }
    }
}

template <typename DType>
inline void DeformableConv2DCol2ImCoord(
  GPUDevice& d, const DType* data_col, const DType* data_im, const DType* data_offset, const DType* data_mask,
  const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
  const TShape& pad, const TShape& stride,
  const TShape& dilation, const uint32_t deformable_group,
  DType* grad_offset, DType* grad_mask){
  index_t num_spatial_axes = kernel_shape.size();
  index_t num_kernels = col_shape[1] * col_shape[2] * col_shape[3] * 2 * kernel_shape[0] * kernel_shape[1] * deformable_group;
  index_t channel_per_deformable_group = col_shape[0] / deformable_group;
  // num_axes should be smaller than block size
  CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
  CHECK_LT(num_spatial_axes, config.thread_per_block);
  switch (num_spatial_axes) {
  case 2:
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)

    DeformableConv2DCol2ImCoordGPUKernel<DType> << <config.block_count, config.thread_per_block,
      0, d.stream() >> >(
        num_kernels, data_col, data_im, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
        kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
        dilation[0], dilation[1], channel_per_deformable_group,
        col_shape[1], 2 * kernel_shape[0] * kernel_shape[1] * deformable_group, deformable_group, col_shape[2], col_shape[3], 
        grad_offset, grad_mask);
    // MSHADOW_CUDA_POST_KERNEL_CHECK(DeformableConv2DCol2ImCoordGPUKernel);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
      << num_spatial_axes << " spatial axes";
  }
}

template <typename DType>
inline void DeformableConv2DCol2Im(GPUDevice& d, 
    const DType* data_col, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride,
    const TShape& dilation, const uint32_t deformable_group,
    DType* grad_im){
    index_t num_spatial_axes = kernel_shape.size();
  index_t im_size = ProdShape(im_shape, 1, im_shape.size());
  index_t channel_per_deformable_group = im_shape[1] / deformable_group;
  index_t num_kernels = ProdShape(col_shape, 0, col_shape.size());
  // num_axes should be smaller than block size
  CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
  CHECK_LT(num_spatial_axes, config.thread_per_block);
    //   using namespace mxnet_op;
  switch (num_spatial_axes) {
  case 2:
    // To avoid involving atomic operations, we will launch one kernel per
    // bottom dimension, and then in the kernel add up the top dimensions.
    // NOLINT_NEXT_LINE(whitespace/operators)
        DeformableConv2DCol2ImKernel<DType><<<config.block_count, config.thread_per_block,
                               0, d.stream()>>>(
        num_kernels, data_col, data_offset, data_mask, im_shape[1], im_shape[2], im_shape[3],
        kernel_shape[0], kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
        dilation[0], dilation[1], channel_per_deformable_group,
        col_shape[1], deformable_group, col_shape[2], col_shape[3], grad_im, req);
    // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_col2im_gpu_kernel);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
}

template <typename DType>
inline void DeformableConv2DIm2Col(GPUDevice& d, 
    const DType* data_im, const DType* data_offset, const DType* data_mask,
    const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
    const TShape& pad, const TShape& stride, const TShape& dilation,
    const uint32_t deformable_group, DType* data_col){
        // num_axes should be smaller than block size
    index_t num_spatial_axes = kernel_shape.size();
    index_t channel_per_deformable_group = im_shape[1] / deformable_group; // imshape[1] = 输入图的通道数
    index_t num_kernels = im_shape[1] * ProdShape(col_shape, 1, col_shape.size()); // K * N / k.Size(), k = filter, col_shape = [K, im2col_step_, H, W]
    CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
    CHECK_LT(num_spatial_axes, config.thread_per_block);
    switch (num_spatial_axes) {
    case 2:
    DeformableConv2DIm2ColKernel<DType> // NOLINT_NEXT_LINE(whitespace/operators)
        <<<config.block_count, config.thread_per_block, // 注意这里申请的block的个数是num_kernel个,
           0, d.stream()>>>(
           //CUDA对device(GPU )的内存管理主要通过cudaMalloc()、cudaFree()、cudaMemcpy() 进行管理。另外，从上述代码我们可以看到，
           //add() 函数的调用比较奇怪相对于C语言来说，需要用add<<<M，N>>> 这种形式表明这是一个从host(CPU)代码调用device的代码，
           //并且括号中的数值表明，M个block，每个block有 N个线程, 所以这个函数总共有M*N个线程。
        num_kernels,
        data_im,
        data_offset,
        data_mask,
        im_shape[2], im_shape[3],
        kernel_shape[0], kernel_shape[1],
        pad[0], pad[1],
        stride[0], stride[1],
        dilation[0], dilation[1],
        channel_per_deformable_group,
        col_shape[1], im_shape[1],
        deformable_group,
        col_shape[2], col_shape[3],
        data_col);
        // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_im2col_gpu_kernel);
        break;
        default:
        LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
    }
}

}
#endif
#endif