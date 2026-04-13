#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "turbo_ocr/common/perspective_math.h"
#include "turbo_ocr/decode/gpu_image.h"

namespace turbo_ocr::kernels {

using decode::GpuImage;

// Fused batched ROI warp (perspective) + resize + normalize for recognition
void cuda_batch_roi_warp(const GpuImage &src, const float *d_M_invs,
                         const int *d_crop_widths, float *d_dst_batch,
                         int batch_size, int dst_h, int dst_w,
                         cudaStream_t stream = 0);

// ArgMax for CTC decoding
void cuda_argmax(const float *input_probs, int *output_indices,
                 float *output_scores, int batch_size, int seq_len,
                 int num_classes, cudaStream_t stream = 0);

// Fused resize + normalize + CHW for detection (eliminates intermediate buffer)
void cuda_fused_resize_normalize_det(const GpuImage &src, float *dst_chw,
                                      int dst_w, int dst_h,
                                      cudaStream_t stream = 0);

// Fused resize + normalize + CHW for PP-DocLayoutV3. Identical kernel to the
// det variant, but applies `pixel / 255` normalization (mean=0, std=1) to
// match the model's inference.yml NormalizeImage(norm_type=none) step.
// Input is expected as BGR uint8; output is float CHW at dst_h x dst_w.
void cuda_fused_resize_normalize_layout(const GpuImage &src, float *dst_chw,
                                         int dst_w, int dst_h,
                                         cudaStream_t stream = 0);

// Batched fused resize + normalize + CHW for detection
// Processes N images (each with different src dimensions) into a single
// batched CHW tensor [N, 3, dst_h, dst_w].
// d_src_ptrs[N], d_src_steps[N], d_src_heights[N], d_src_widths[N] are
// device arrays describing each source image.
void cuda_batch_fused_resize_normalize_det(
    const void *const *d_src_ptrs, const int *d_src_steps,
    const int *d_src_heights, const int *d_src_widths,
    float *dst_chw, int dst_w, int dst_h, int batch_size,
    cudaStream_t stream = 0);

// Batched threshold + float->uint8 (processes batch_size * w * h elements)
void cuda_batch_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                                int batch_size, float thresh,
                                cudaStream_t stream = 0);

// Fused threshold + float->uint8
void cuda_threshold_to_u8(const float *src, uint8_t *dst, int w, int h,
                          float thresh, cudaStream_t stream = 0);

// Compute inverse perspective transform: maps dst quad to src quad.
// Delegates to turbo_ocr::compute_perspective_inv in common/perspective_math.h
// (CUDA-free pure math). Kept here for backward compatibility.
inline void compute_perspective_inv(
    const float* dst_pts, const float* src_pts,
    float* M_inv) {
  turbo_ocr::compute_perspective_inv(dst_pts, src_pts, M_inv);
}

// --- GPU Connected Component Labeling + BBox Extraction ---

// Result struct for one connected component (transferred from GPU to CPU)
struct GpuDetBox {
  int xmin, ymin, xmax, ymax; // bounding box in resize coords
  float score;                // mean of pred_map within bbox
  int pixel_count;            // number of foreground pixels in component
};

// Maximum number of components we track on GPU
static constexpr int kMaxGpuComponents = 2048;

// Run full GPU CCL pipeline: label components, extract bboxes, compute scores.
// Returns number of valid boxes written to h_boxes (host memory).
// All GPU work is on the given stream. This function synchronizes the stream
// exactly ONCE at the end to transfer the small result array to the host.
//
// Required GPU buffers (ALL pre-allocated by caller, no per-request alloc):
//   d_labels:       int[w*h]                      -- label map
//   d_compact_ids:  int[w*h]                      -- compact component IDs
//   d_id_counter:   int[1]                        -- atomic counter for compact IDs
//   d_bboxes:       GpuDetBox[kMaxGpuComponents*2] -- per-component bbox + filtered output
//   d_num_boxes:    int[1]                        -- output count
//
// h_boxes must point to at least kMaxGpuComponents GpuDetBox entries (pinned).
int cuda_gpu_ccl_detect(
    const uint8_t *d_bitmap,     // binary bitmap (255=fg, 0=bg)
    const float *d_pred_map,     // raw probability map
    int w, int h,
    float box_thresh,            // score threshold for filtering
    int *d_labels,               // [w*h] scratch
    int *d_compact_ids,          // [w*h] scratch
    int *d_id_counter,           // [1] scratch
    GpuDetBox *d_bboxes,         // [kMaxGpuComponents*2] scratch
    int *d_num_boxes,            // [1] scratch
    GpuDetBox *h_boxes,          // host output (pinned)
    int *h_num_boxes,            // host output count
    cudaStream_t stream);

} // namespace turbo_ocr::kernels
