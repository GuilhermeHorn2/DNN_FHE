// Accuracy benchmark: build network ONCE, iterate over labeled images.
//
// Expected directory layout:
//   <root>/0/*.png   <root>/1/*.png   ...   <root>/9/*.png
//
// Built only when -DBUILD_ACCURACY=ON.

#include "bench/bench.h"
#include "io/csv.h"
#include "io/image.h"
#include "network/activation.h"
#include "network/context.h"
#include "network/network.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static constexpr int IN_DIM  = 784;
static constexpr int HID_DIM = 30;
static constexpr int OUT_DIM = 10;

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Accuracy benchmark");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <test_root> [weights_dir=..]\n"
                  << "  Iterates <test_root>/<label>/*.{png,jpg} and reports accuracy.\n";
        return 1;
    }

    const std::string testRoot   = argv[1];
    const std::string weightsDir = (argc >= 3) ? argv[2] : "..";

    auto W1 = io::LoadCsv2D(weightsDir + "/dinn30_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = io::LoadCsv1D(weightsDir + "/dinn30_b1.csv");
    auto W2 = io::LoadCsv2D(weightsDir + "/dinn30_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = io::LoadCsv1D(weightsDir + "/dinn30_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights from " << weightsDir << "\n";
        return 1;
    }

    using namespace fhednn;
    FHEContext ctx;
    Network    net;
    net.SetInputShift(1)
       .Linear(W1, b1)
       .Activate(activations::Sign(256, ctx.params().pInput))
       .Linear(W2, b2);
    net.Compile(ctx);
    BENCH_MEM("after-compile");

    int                       total       = 0;
    int                       correct     = 0;
    std::vector<std::vector<int>> confusion(OUT_DIM, std::vector<int>(OUT_DIM, 0));

    for (int label = 0; label < OUT_DIM; ++label) {
        const fs::path classDir = fs::path(testRoot) / std::to_string(label);
        if (!fs::exists(classDir)) continue;

        for (const auto& entry : fs::directory_iterator(classDir)) {
            if (!entry.is_regular_file()) continue;
            const auto& path = entry.path();
            const auto  ext  = path.extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

            auto pixels = io::LoadImageBipolar(path.c_str(), IN_DIM);
            if (pixels.empty()) continue;

            auto scores = net.Run(pixels);
            int predicted = 0;
            std::int64_t best = scores.empty() ? 0 : scores[0];
            for (int j = 1; j < static_cast<int>(scores.size()) && j < OUT_DIM; ++j) {
                if (scores[j] > best) {
                    best      = scores[j];
                    predicted = j;
                }
            }

            ++total;
            if (predicted == label) ++correct;
            confusion[label][predicted] += 1;

            std::printf("[%4d] %s -> pred=%d  truth=%d  %s\n",
                        total, path.filename().c_str(), predicted, label,
                        predicted == label ? "OK" : "MISS");
            std::fflush(stdout);
        }
    }

    std::printf("\n=== Accuracy ===\n");
    std::printf("Correct: %d / %d  (%.2f%%)\n",
                correct, total, total > 0 ? 100.0 * correct / total : 0.0);

    std::printf("\nConfusion matrix (row=truth, col=pred):\n     ");
    for (int j = 0; j < OUT_DIM; ++j) std::printf(" %4d", j);
    std::printf("\n");
    for (int i = 0; i < OUT_DIM; ++i) {
        std::printf("  %d :", i);
        for (int j = 0; j < OUT_DIM; ++j) std::printf(" %4d", confusion[i][j]);
        std::printf("\n");
    }

    BENCH_MEM("end");
    return 0;
}
