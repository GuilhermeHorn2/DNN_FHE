#include "bench/bench.h"
#include "io/csv.h"
#include "io/image.h"
#include "network/activation.h"
#include "network/context.h"
#include "network/network.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <iostream>
#include <vector>

// MNIST-like DiNN100 dimensions (784 -> 100 -> 10), Heaviside variant.
static constexpr int IN_DIM  = 784;
static constexpr int HID_DIM = 100;
static constexpr int OUT_DIM = 10;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png>\n"
                  << "  Weights expected at ../heaviside100_W{1,2}.csv and ../heaviside100_b{1,2}.csv\n";
        return 1;
    }

    std::cout << "Loading weights...\n";
    auto W1 = io::LoadCsv2D("../heaviside100_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = io::LoadCsv1D("../heaviside100_b1.csv");
    auto W2 = io::LoadCsv2D("../heaviside100_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = io::LoadCsv1D("../heaviside100_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights.\n";
        return 1;
    }

    std::cout << "Loading image: " << argv[1] << "\n";
    // Heaviside networks are trained on binary {0, 1} pixels, not bipolar
    // {-1, +1}. Re-map the bipolar loader output to {0, 1}.
    auto pixels = io::LoadImageBipolar(argv[1], IN_DIM);
    if (pixels.empty()) return 1;
    for (auto& p : pixels) p = (p + 1) / 2;  // {-1,+1} -> {0,1}

    using namespace fhednn;

    FHEContext ctx;
    Network    net;
    // No input shift needed: pixels are already in [0, pInput).
    net.Linear(W1, b1)
       .Activate(activations::Heaviside(/*preShift=*/256, ctx.params().pInput))
       .Linear(W2, b2);

    net.Compile(ctx);
    BENCH_MEM("after-compile");

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

    // ── Plaintext reference (same {0,1} inputs the FHE side received) ────
    std::vector<double> hidden(HID_DIM, 0.0);
    for (int j = 0; j < HID_DIM; ++j) {
        double acc = b1[j];
        for (int i = 0; i < IN_DIM; ++i) acc += W1[j][i] * static_cast<double>(pixels[i]);
        hidden[j] = (acc >= 0.0) ? 1.0 : 0.0;
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
