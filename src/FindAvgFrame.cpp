#include "SPP_STRUCTS.hpp"
#include "SPP_UTILS.hpp"

#include "logger.hpp"
#include "FindAvgFrame.hpp"

#include <cstddef>
#include <cstring>
#include <vector>

// TODO: Highway or CUDA to make it run faster

SPP_STRUCTS::FrameData* find_average_frame(SPP_STRUCTS::VideoData* video)
{
  if (!video)
  {
    LOG_ERR("VideoData* pointer is null.");
    return nullptr;
  }

  if (!video->data)
  {
    LOG_ERR("VideoData data* is null.");
    return nullptr;
  }

  if (video->count ==0)
  {
    LOG_ERR("Frame number is 0. How?");
    return nullptr;
  }

  const size_t frame_size     = video->frame_size;
  const size_t count          = video->count;

  // Accumulate
  std::vector<uint8_t> accumulate(frame_size, 0); // Max 255 frame

  for (size_t i = 0; i < count; i++)
  {
    const uint8_t* src = SPP_UTILS::get_frame(video, i);
    if (!src) continue;

    for (size_t p = 0; p < frame_size; p++)
    {
      accumulate[p] += src[p];
    }
  }

  // Allocate result
  // Round-up to multiplie of N
  constexpr size_t ALIGN          = 64;
  // ALIGN-byte alignment - e.g. frame_size 60 -> aligned_frame_size 64 ; frame_size 100 -> aligned_frame_size 128 ...
  const size_t aligned_frame_size = (frame_size + ALIGN - 1) & ~(ALIGN - 1);

  void*    raw = std::aligned_alloc(ALIGN, aligned_frame_size);
  uint8_t* dst = static_cast<uint8_t*>(raw);

  if (!dst)
  {
    LOG_ERR("dst pointer point to null");
    return nullptr;
  }

  // Divide and Store result
  for (size_t p = 0; p < frame_size; ++p)
    dst[p] = static_cast<uint8_t>(accumulate[p] / count);

  if (aligned_frame_size > frame_size)
    std::memset(dst + frame_size, 0, aligned_frame_size - frame_size);

  // Create FrameData and return
  SPP_STRUCTS::FrameData* fd = new SPP_STRUCTS::FrameData();
  fd->width                 = video->width;
  fd->height                = video->height;
  fd->channels              = video->channels;
  fd->frame_size            = video->frame_size;
  fd->aligned_frame_size    = video->aligned_frame_size;
  fd->data                  = dst;
  return fd;
}
