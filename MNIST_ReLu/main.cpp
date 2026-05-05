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

// MNIST-like DiNN100 dimensions (784 -> 100 -> 10), ReLU variant.
static constexpr int IN_DIM  = 784;
static constexpr int HID_DIM = 100;
static constexpr int OUT_DIM = 10;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png>\n"
                  << "  Weights expected at ../relu100_W{1,2}.csv and ../relu100_b{1,2}.csv\n";
        return 1;
    }

    std::cout << "Loading weights...\n";
    auto W1 = io::LoadCsv2D("../relu100_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = io::LoadCsv1D("../relu100_b1.csv");
    auto W2 = io::LoadCsv2D("../relu100_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = io::LoadCsv1D("../relu100_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights.\n";
        return 1;
    }

    std::cout << "Loading image: " << argv[1] << "\n";
    // ReLU networks here are trained on binary {0, 1} pixels (same convention
    // as the Heaviside variant), not bipolar {-1, +1}. Re-map the bipolar
    // loader output to {0, 1}.
    auto pixels = io::LoadImageBipolar(argv[1], IN_DIM);
    if (pixels.empty()) return 1;
    for (auto& p : pixels) p = (p + 1) / 2;  // {-1,+1} -> {0,1}

    // ── Plain-side magnitude diagnostics (raw, unscaled weights) ─────────
    // Quick sanity check that explains why FHE results may be wrong: the
    // final decryption is mod pOutput = 1024, interpreted as a signed window
    // (-512, +512). Any class score outside that window will wrap and
    // destroy the argmax.
    {
        std::vector<double> hiddenDiag(HID_DIM, 0.0);
        double preActMin = 1e18, preActMax = -1e18;
        double hidMin    = 1e18, hidMax    = -1e18;
        for (int j = 0; j < HID_DIM; ++j) {
            double acc = b1[j];
            for (int i = 0; i < IN_DIM; ++i) acc += W1[j][i] * static_cast<double>(pixels[i]);
            preActMin = std::min(preActMin, acc);
            preActMax = std::max(preActMax, acc);
            hiddenDiag[j] = (acc > 0.0) ? acc : 0.0;
            hidMin = std::min(hidMin, hiddenDiag[j]);
            hidMax = std::max(hidMax, hiddenDiag[j]);
        }
        double scoreMin = 1e18, scoreMax = -1e18;
        for (int j = 0; j < OUT_DIM; ++j) {
            double s = b2[j];
            for (int i = 0; i < HID_DIM; ++i) s += W2[j][i] * hiddenDiag[i];
            scoreMin = std::min(scoreMin, s);
            scoreMax = std::max(scoreMax, s);
        }
        std::printf("[Diag] pre-activation y range : [%8.1f, %8.1f]   (LUT safe range is [-256, +768))\n",
                    preActMin, preActMax);
        std::printf("[Diag] hidden ReLU(y) range  : [%8.1f, %8.1f]\n", hidMin, hidMax);
        std::printf("[Diag] raw refScores range   : [%8.1f, %8.1f]   (decoder safe range is (-512, +512))\n",
                    scoreMin, scoreMax);
    }

    // ── Down-scale weights so internal magnitudes fit the FHE pipeline.
    //    Two independent scales:
    //      HIDDEN_SCALE_K -> shrinks pre-activations into [-256, +768) so the
    //                        ReLU LUT does not wrap (period = pInput = 1024).
    //      OUTPUT_SCALE_K -> shrinks final class scores into (-512, +512) so
    //                        DecryptCoeff (mod pOutput = 1024) does not wrap.
    //    argmax is invariant under any positive scalar, so the plaintext
    //    reference below — which uses the same W/b vectors — stays in lockstep
    //    with the FHE result. Bump either constant up if the diagnostics above
    //    still show out-of-range values.
    constexpr double HIDDEN_SCALE_K = 4.0;
    constexpr double OUTPUT_SCALE_K = 64.0;
    for (auto& row : W1) for (auto& w : row) w /= HIDDEN_SCALE_K;
    for (auto& v : b1)                       v /= HIDDEN_SCALE_K;
    for (auto& row : W2) for (auto& w : row) w /= OUTPUT_SCALE_K;
    for (auto& v : b2)                       v /= OUTPUT_SCALE_K;
    std::printf("[Diag] scaled W1/b1 by 1/%.1f, W2/b2 by 1/%.1f (argmax invariant)\n",
                HIDDEN_SCALE_K, OUTPUT_SCALE_K);

    // Re-print magnitudes after scaling so we can confirm the choice fits
    // both the LUT and the decoder windows.
    {
        std::vector<double> hiddenDiag(HID_DIM, 0.0);
        double preActMin = 1e18, preActMax = -1e18;
        double hidMin    = 1e18, hidMax    = -1e18;
        for (int j = 0; j < HID_DIM; ++j) {
            double acc = b1[j];
            for (int i = 0; i < IN_DIM; ++i) acc += W1[j][i] * static_cast<double>(pixels[i]);
            preActMin = std::min(preActMin, acc);
            preActMax = std::max(preActMax, acc);
            hiddenDiag[j] = (acc > 0.0) ? acc : 0.0;
            hidMin = std::min(hidMin, hiddenDiag[j]);
            hidMax = std::max(hidMax, hiddenDiag[j]);
        }
        double scoreMin = 1e18, scoreMax = -1e18;
        for (int j = 0; j < OUT_DIM; ++j) {
            double s = b2[j];
            for (int i = 0; i < HID_DIM; ++i) s += W2[j][i] * hiddenDiag[i];
            scoreMin = std::min(scoreMin, s);
            scoreMax = std::max(scoreMax, s);
        }
        std::printf("[Diag] scaled pre-act y'    : [%8.1f, %8.1f]   (need (-256, +768))\n",
                    preActMin, preActMax);
        std::printf("[Diag] scaled hidden h'     : [%8.1f, %8.1f]\n", hidMin, hidMax);
        std::printf("[Diag] scaled refScores     : [%8.1f, %8.1f]   (need (-512, +512))\n",
                    scoreMin, scoreMax);
    }

    using namespace fhednn;

    FHEContext ctx;
    Network    net;
    // No input shift needed: pixels are already in [0, pInput).
    net.Linear(W1, b1)
       .Activate(activations::ReLU(/*preShift=*/256, ctx.params().pInput))
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
        hidden[j] = (acc > 0.0) ? acc : 0.0;
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
