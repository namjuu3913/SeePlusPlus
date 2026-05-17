#include "FindAvgFrame.hpp"
#include "SPP_STRUCTS.hpp"
#include "SPP_UTILS.hpp"

#include <logger.hpp>

#include <immintrin.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Accumulate frames [start_frame, end_frame) into accum (float[]).
static void accumulate_window(const SPP_STRUCTS::VideoData* vd,
                                size_t start_frame, size_t end_frame,
                                float* accum)
{
  const size_t frame_size = vd->frame_size; // width * height * 3 

  for (size_t fi = start_frame; fi < end_frame; fi++)
  {
    const uint8_t* src = SPP_UTILS::get_frame(vd, fi);
    if (!src) continue;

    // AXV2 logical: process 8 pixels channels at a time.
    // 'i' is a flat byte index - covers R G B bytes of consecutive
    // pixels in a row-major order without any speciaal channel handing.
    size_t i = 0;
    for (; i + 8 <= frame_size; i += 8)
    {
      // Load 8 raw unit8 bytes (64 bytes total) from the frame at offset i.
      // These may span pixel channels or maybe pixel boundaries, BUT
      // those does not matter because each byte index is independent.
      __m128i bytes  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src + i));

      // Zero-extend each uint8 lane (or whatever it call) to int32
      // so we can safely add without overflow inside a single iteration.
      __m256i ints   = _mm256_cvtepu8_epi32(bytes);

      // Convert int32 → float32 for the accumulate.
      __m256  floats = _mm256_cvtepi32_ps(ints);

      // Load the current accumulate values at the same 8 positions.
      __m256  acc_v  = _mm256_loadu_ps(accum + i);

      // Add and store back after all frames these hold the sum of
      // every frame's byte value at each position.
      _mm256_storeu_ps(accum + i, _mm256_add_ps(acc_v, floats));
    }

    // Scalar tail for leftover bytes.
    // (Rarely, only if frame_size is a multiple of 8)
    for (; i < frame_size; i++)
      accum[i] += static_cast<float>(src[i]);
  }
}

// TODO: Add src into dst, both float[frame_size].
//       Mirror the AVX2 pattern in accumulate_window:
//         - loop 8 floats at a time with _mm256_loadu_ps / _mm256_add_ps / _mm256_storeu_ps ...
//         - scalar tail for the remaining (frame_size % 8) elements
//       This is called once per thread buffer after join() to merge
//       all per-thread sums into accums[0].
static void reduce_accum(float* dst, const float* src, size_t frame_size)
{
  // TODO
}

// TODO: Divide accum by inv_count, clamp to [0,255], and pack into out (uint8[frame_size]).
//       AVX2 pack-down chain per 8 elements:
//         1. _mm256_mul_ps(accum[i..i+7], vinv)          float sum → float average
//         2. _mm256_cvtps_epi32(avg)                     float → int32 (nearest-even)
//         3. split __m256i into lo/hi __m128i
//         4. _mm_packs_epi32(lo, hi)                     int32[4]+int32[4] → int16[8]
//         5. _mm_packus_epi16(i16, _mm_setzero_si128())  int16[8] → uint8[8] (clamps 0-255)
//         6. _mm_storel_epi64(out + i, u8)               store 8 bytes
//       Scalar tail: out[i] = clamp(accum[i] * inv_count, 0, 255)
static void divide_and_pack(const float* accum,
                             float        inv_count,
                             size_t       frame_size,
                             uint8_t*     out)
{
  // TODO
}

// TODO: Compute the centroid of frames [start_frame, start_frame+window_size) and write to dst.
//       Steps:
//         1. Clamp num_threads to window_size (never more threads than frames).
//         2. Allocate num_threads private float[] accumulators, all zero-initialised.
//         3. Split window_size frames evenly across threads; last thread takes the remainder.
//         4. Spawn threads, each calling accumulate_window() on its slice into its own accum.
//         5. join() all threads.
//         6. reduce_accum(): merge accums[1..N] into accums[0].
//         7. divide_and_pack(accums[0], 1/window_size) → dst.
static void compute_centroid(const SPP_STRUCTS::VideoData* vd,
                              size_t   start_frame,
                              size_t   window_size,
                              int      num_threads,
                              uint8_t* dst)
{
  // TODO
}

/*
 * find_average_frame pipeline:
 *
 *   VideoData* vd  (full decoded video)
 *        |
 *        | window_size = floor(fps * 0.5)   e.g. 30fps → 15 frames per window
 *        | num_segments = floor(count / window_size)
 *        |
 *        |---> segment 0: frames  0 ..  14  → compute_centroid() → out_data[0]
 *        |---> segment 1: frames 15 ..  29  → compute_centroid() → out_data[1]
 *        |---> segment 2: frames 30 ..  44  → compute_centroid() → out_data[2]
 *        |     ...
 *        |---> segment N: frames .. .. count → compute_centroid() → out_data[N]
 *        |
 *        |     (trailing frames that don't fill a window are discarded)
 *        |     (Since those didn't fit up to 15 frames)
 *        |
 *        v
 *   VideoData* result
 *     count  = num_segments       (one centroid image per segment)
 *     fps    = 2.0                (one centroid per 0.5 s)
 *     data   = aligned buffer of all centroid images packed contiguously
 *
 *   KNN receives (vd, result) and for each centroid finds the closest
 *   real frame in that window via cosine similarity.
 *   Caller frees with SPP_UTILS::free_video_data().
 *
 * TODO: Implement the body:
 *         1. Validate vd (null check, count > 0, fps > 0).
 *         2. Compute window_size and num_segments; guard against both being 0.
 *         3. std::aligned_alloc(64, aligned_frame_size * num_segments) for out_data.
 *         4. std::memset the buffer to 0 (clean padding bytes).
 *         5. Get num_threads from std::thread::hardware_concurrency().
 *         6. Loop over segments, call compute_centroid() for each, writing into
 *            out_data + seg * aligned_frame_size.
 *         7. Fill and return a new VideoData* with count=num_segments, fps=2.0.
 */
SPP_STRUCTS::VideoData* find_average_frame(SPP_STRUCTS::VideoData* vd)
{
  // TODO
  return vd; //tmp, just to make IDE not show any error :L
}
