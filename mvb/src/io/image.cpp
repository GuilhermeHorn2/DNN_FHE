#include "io/image.h"

// stb_image is header-only and lives next to main.cpp. Define implementation
// in exactly one TU; this is the only one.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

namespace io {

std::vector<std::int64_t> LoadImageBipolar(const char* path, int dim) {
    int            w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(path, &w, &h, &ch, 1);
    if (!data) {
        std::cerr << "[ERROR] Cannot load: " << path << "\n";
        return {};
    }
    std::vector<std::int64_t> out(dim, -1);
    const int                 limit = std::min(dim, w * h);
    for (int i = 0; i < limit; ++i) out[i] = (data[i] > 127) ? 1 : -1;
    stbi_image_free(data);
    return out;
}

std::vector<std::int64_t> LoadImageGrayscale(const char* path, int dim) {
    int            w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(path, &w, &h, &ch, 1);
    if (!data) {
        std::cerr << "[ERROR] Cannot load: " << path << "\n";
        return {};
    }
    std::vector<std::int64_t> out(dim, 0);
    const int                 limit = std::min(dim, w * h);
    for (int i = 0; i < limit; ++i)
        out[i] = static_cast<std::int64_t>(data[i]);
    stbi_image_free(data);
    return out;
}

}  // namespace io
