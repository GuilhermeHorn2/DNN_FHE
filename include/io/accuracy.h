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

// Optional per-project plaintext scorer. When supplied, the harness calls
// it with the same pixel vector that went into `Network::Run` and uses
// the returned class scores to derive a plain-side argmax. The harness
// then reports two extra numbers alongside FHE accuracy: plaintext
// accuracy (plain vs truth) and FHE-vs-plain agreement (a pure FHE
// correctness signal, isolated from model error).
//
// Length of the returned vector must match `outDim`. The function is
// project-specific: it has to know the layer topology, activation, and
// any weight pre-scaling that `main.cpp` applied before Compile().
using PlainScorer =
    std::function<std::vector<double>(const std::vector<std::int64_t>& pixels)>;

struct AccuracyResult {
    int total           = 0;
    int correct         = 0;   // FHE argmax vs truth
    int correctPlain    = 0;   // plain argmax vs truth (only when plainScorer is set)
    int fheMatchesPlain = 0;   // FHE argmax vs plain argmax (idem)
    bool hasPlain       = false;
    // confusion[truth][pred]; size outDim × outDim. Always indexes the FHE
    // prediction so the matrix is comparable across runs with/without
    // plainScorer.
    std::vector<std::vector<int>> confusion;
};

// Iterates `<testRoot>/<label>/*.{png,jpg,jpeg}` for label in [0, outDim),
// scores each image via `net.Run(loadPixels(path))`, and tallies a
// confusion matrix. Prints per-image OK/MISS lines as it goes — when
// `plainScorer` is supplied each line also includes `plain=<pred>`.
//
// `net` must already be Compile()d. `loadPixels` is invoked once per image.
// Skips silently on unknown extensions, missing class folders, or empty
// pixel vectors (e.g. an unreadable file).
AccuracyResult RunAccuracyLoop(fhednn::Network&   net,
                               const std::string& testRoot,
                               const PixelLoader& loadPixels,
                               int                outDim,
                               const PlainScorer& plainScorer = {});

// Pretty-prints accuracy + confusion matrix. When `r.hasPlain` is set,
// also prints the "FHE vs PlainText" agreement and "PlainText Acc"
// blocks above the FHE-vs-truth one.
void PrintAccuracySummary(const AccuracyResult& r);

}  // namespace io

#endif  // MVB_IO_ACCURACY_H
