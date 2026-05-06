#include "bench/bench.h"
#include "io/accuracy.h"
#include "io/csv.h"
#include "io/image.h"
#include "network/activation.h"
#include "network/context.h"
#include "network/network.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// MNIST-like DiNN100 dimensions (784 -> 100 -> 10).
static constexpr int IN_DIM  = 784;
static constexpr int HID_DIM = 100;
static constexpr int OUT_DIM = 10;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png | test_root>\n"
                  << "  Single-image:  pass an image path -> runs one inference + plaintext reference\n"
                  << "  Batch:         pass a directory of <label>/*.{png,jpg,jpeg} -> accuracy + confusion matrix\n"
                  << "  Weights expected at ../signal100_W{1,2}.csv and ../signal100_b{1,2}.csv\n";
        return 1;
    }

    std::cout << "Loading weights...\n";
    auto W1 = io::LoadCsv2D("../signal100_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = io::LoadCsv1D("../signal100_b1.csv");
    auto W2 = io::LoadCsv2D("../signal100_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = io::LoadCsv1D("../signal100_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights.\n";
        return 1;
    }

    using namespace fhednn;

    // FHEParams params;
    // params.securityLevel = lbcrypto::HEStd_128_classic;
    // params.ringDim       = 1u << 17;
    // FHEContext ctx{params};
    FHEContext ctx;
    Network    net;
    net.SetInputShift(1)
       .Linear(W1, b1)
       .Activate(activations::Sign(/*preShift=*/256, ctx.params().pInput))
       .Linear(W2, b2);

    net.Compile(ctx);
    BENCH_MEM("after-compile");

    // ── Batch mode: argv[1] is a directory of <label>/*.{png,jpg,jpeg} ─────
    // The harness only knows how to walk + score; it asks us how to turn a
    // single image path into the {-1,+1} pixel vector this Sign network
    // expects, and (optionally) how to score the same pixels in plaintext
    // so it can report FHE vs plain agreement.
    if (fs::is_directory(argv[1])) {
        auto loadPixels = [](const std::string& p) {
            return io::LoadImageBipolar(p.c_str(), IN_DIM);
        };

        // Plain-side reference: same topology as the FHE network (Linear ->
        // Sign -> Linear). Captures W1/b1/W2/b2 by reference; they are not
        // mutated after Compile() in this driver, so this is safe.
        auto plainScore = [&](const std::vector<std::int64_t>& px) {
            std::vector<double> hidden(HID_DIM, 0.0);
            for (int j = 0; j < HID_DIM; ++j) {
                double acc = b1[j];
                for (int i = 0; i < IN_DIM; ++i)
                    acc += W1[j][i] * static_cast<double>(px[i]);
                hidden[j] = (acc >= 0.0) ? 1.0 : -1.0;
            }
            std::vector<double> scores(OUT_DIM, 0.0);
            for (int j = 0; j < OUT_DIM; ++j) {
                scores[j] = b2[j];
                for (int i = 0; i < HID_DIM; ++i)
                    scores[j] += W2[j][i] * hidden[i];
            }
            return scores;
        };

        auto result = io::RunAccuracyLoop(net, argv[1], loadPixels, OUT_DIM, plainScore);
        io::PrintAccuracySummary(result);
        BENCH_MEM("end");
        return 0;
    }

    // ── Single-image mode (existing behavior) ─────────────────────────────
    std::cout << "Loading image: " << argv[1] << "\n";
    auto pixels = io::LoadImageBipolar(argv[1], IN_DIM);
    if (pixels.empty()) return 1;

    auto scores = net.Run(pixels);

    std::cout << "\n--- Final Class Scores ---\n";
    int           predicted = 0;
    std::int64_t  maxScore  = LLONG_MIN;
    for (int j = 0; j < OUT_DIM; ++j) {
        std::printf("  Digit %d: %lld\n", j, static_cast<long long>(scores[j]));
        if (scores[j] > maxScore) {
            maxScore  = scores[j];
            predicted = j;
        }
    }

    // ── Plaintext reference (uses the originals we kept by value) ─────────
    std::vector<double> hidden(HID_DIM, 0.0);
    for (int j = 0; j < HID_DIM; ++j) {
        double acc = b1[j];
        for (int i = 0; i < IN_DIM; ++i) acc += W1[j][i] * static_cast<double>(pixels[i]);
        hidden[j] = (acc >= 0.0) ? 1.0 : -1.0;
    }
    std::vector<double> refScores(OUT_DIM, 0.0);
    for (int j = 0; j < OUT_DIM; ++j) {
        refScores[j] = b2[j];
        for (int i = 0; i < HID_DIM; ++i) refScores[j] += W2[j][i] * hidden[i];
    }
    const int refPred = static_cast<int>(
        std::max_element(refScores.begin(), refScores.end()) - refScores.begin());

    std::cout << "\n================================\n";
    std::printf(" PREDICTED DIGIT  : %d\n", predicted);
    std::printf(" REFERENCE (plain): %d  %s\n", refPred,
                predicted == refPred ? "(matches)" : "(MISMATCH)");
    std::cout << "================================\n";

    BENCH_MEM("end");
    return 0;
}
