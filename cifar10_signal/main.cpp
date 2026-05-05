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

// CIFAR-10: 32 x 32 x 3 RGB → flat 3072-dim input, then a 30-neuron hidden
// bipolar layer and a 10-class output (same topology as DiNN30, just a
// fatter first-layer matrix and a much bigger ring dimension).
static constexpr int IN_DIM  = 3072;
static constexpr int HID_DIM = 30;
static constexpr int OUT_DIM = 10;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png>\n"
                  << "  Weights expected at ../cifar10_weights_W{1,2}.csv "
                  << "and ../cifar10_weights_b{1,2}.csv\n";
        return 1;
    }

    std::cout << "Loading weights...\n";
    auto W1 = io::LoadCsv2D("../cifar10_weights_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = io::LoadCsv1D("../cifar10_weights_b1.csv");
    auto W2 = io::LoadCsv2D("../cifar10_weights_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = io::LoadCsv1D("../cifar10_weights_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights.\n";
        return 1;
    }

    std::cout << "Loading image: " << argv[1] << "\n";
    // CIFAR images are 3-channel RGB; passing channels=3 forces stb_image
    // to deliver 3072 = 32*32*3 bytes regardless of the file's encoding.
    auto pixels = io::LoadImageBipolar(argv[1], IN_DIM, /*channels=*/3);
    if (pixels.empty()) return 1;

    using namespace fhednn;

    // CIFAR's flat input is 3072 entries. The framework defaults assume
    // numSlots = 1024 / ringDim = 2048; we have to scale both up to fit.
    FHEParams p;
    p.numSlots = 4096;             // next power of two ≥ 3072
    p.ringDim  = 1u << 13;         // 8192 (≥ 2 * numSlots)

    FHEContext ctx{p};
    Network    net;
    net.SetInputShift(1)
       .Linear(W1, b1)
       .Activate(activations::Sign(/*preShift=*/256, ctx.params().pInput))
       .Linear(W2, b2);

    net.Compile(ctx);
    BENCH_MEM("after-compile");

    auto scores = net.Run(pixels);

    std::cout << "\n--- Final Class Scores ---\n";
    int           predicted = 0;
    std::int64_t  maxScore  = LLONG_MIN;
    for (int j = 0; j < OUT_DIM; ++j) {
        std::printf("  Class %d: %lld\n", j, static_cast<long long>(scores[j]));
        if (scores[j] > maxScore) {
            maxScore  = scores[j];
            predicted = j;
        }
    }

    // ── Plaintext reference ───────────────────────────────────────────────
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
    std::printf(" PREDICTED CLASS  : %d\n", predicted);
    std::printf(" REFERENCE (plain): %d  %s\n", refPred,
                predicted == refPred ? "(matches)" : "(MISMATCH)");
    std::cout << "================================\n";

    BENCH_MEM("end");
    return 0;
}
