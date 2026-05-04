#ifndef MVB_IO_IMAGE_H
#define MVB_IO_IMAGE_H

#include <cstdint>
#include <vector>

namespace io {

// Loads a grayscale image, thresholds at 127, and produces a {-1, +1} vector
// of length `dim`. Pixels beyond width*height are filled with -1.
// Returns an empty vector on failure.
std::vector<std::int64_t> LoadImageBipolar(const char* path, int dim);

// Loads a grayscale image normalized to [0, 255] -> int64 (no thresholding).
// Useful for non-bipolar networks. Pads with 0 up to `dim`.
std::vector<std::int64_t> LoadImageGrayscale(const char* path, int dim);

}  // namespace io

#endif  // MVB_IO_IMAGE_H
