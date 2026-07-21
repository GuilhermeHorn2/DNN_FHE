#ifndef MVB_IO_ACCURACY_H
#define MVB_IO_ACCURACY_H

#include "network/network.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace io {

// Per-project image-loading hook. Returns the integer pixel vector that
// `Network::Run` expects (bipolar / binary / scaled / etc). The harness
// itself never picks an encoding — it just calls back into the project.
using PixelLoader =
    std::function<std::vector<std::int64_t>(const std::string& imagePath)>;

// Optional per-project plaintext scorer, called with the same pixels that
// went into `Network::Run` to derive a plain-side argmax. When supplied,
// the harness also reports plaintext accuracy and FHE-vs-plain agreement.
// Returned vector length must match `outDim`.
using PlainScorer =
    std::function<std::vector<double>(const std::vector<std::int64_t>& pixels)>;

struct AccuracyResult {
    int total           = 0;
    int correct         = 0;   // FHE argmax vs truth
    int correctPlain    = 0;   // plain argmax vs truth (only when plainScorer is set)
    int fheMatchesPlain = 0;   // FHE argmax vs plain argmax (idem)
    bool hasPlain       = false;
    // confusion[truth][pred], always indexed by the FHE prediction.
    std::vector<std::vector<int>> confusion;
};

// Iterates `<testRoot>/<label>/*.{png,jpg,jpeg}` for label in [0, outDim),
// scores each image via `net.Run(loadPixels(path))`, and tallies a
// confusion matrix. `net` must already be Compile()d. Skips silently on
// unknown extensions, missing class folders, or empty pixel vectors.
AccuracyResult RunAccuracyLoop(fhednn::Network&   net,
                               const std::string& testRoot,
                               const PixelLoader& loadPixels,
                               int                outDim,
                               const PlainScorer& plainScorer = {});

// Pretty-prints accuracy + confusion matrix, plus the plaintext comparison
// blocks when `r.hasPlain` is set.
void PrintAccuracySummary(const AccuracyResult& r);

}  // namespace io

#endif  // MVB_IO_ACCURACY_H
