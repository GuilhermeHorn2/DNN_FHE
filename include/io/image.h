#ifndef MVB_IO_IMAGE_H
#define MVB_IO_IMAGE_H

#include <cstdint>
#include <vector>

namespace io {

// Loads an image, thresholds each byte at 127, and produces a {-1, +1} vector
// of length `dim`. `channels` selects how stb_image decodes the file:
//   1 = grayscale (default, used by MNIST), 3 = RGB (used by CIFAR), 4 = RGBA.
// Bytes beyond `width * height * channels` are filled with -1.
// Returns an empty vector on failure.
std::vector<std::int64_t> LoadImageBipolar(const char* path, int dim, int channels = 1);

// Loads an image normalized to [0, 255] -> int64 (no thresholding).
// Useful for non-bipolar networks. Pads with 0 up to `dim`. `channels` works
// the same way as in LoadImageBipolar.
std::vector<std::int64_t> LoadImageGrayscale(const char* path, int dim, int channels = 1);

}  // namespace io

#endif  // MVB_IO_IMAGE_H
