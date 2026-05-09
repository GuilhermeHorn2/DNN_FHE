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
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// MNIST-like DiNN dimensions: 784 -> HID_DIM -> 10, ReLU variant. The
// hidden width is a runtime parameter so the same driver can load DiNN30,
// DiNN100, or any other size whose weight files exist next to this binary.
static constexpr int IN_DIM  = 784;
static constexpr int OUT_DIM = 10;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " <image_or_folder_path> [hidden_size]\n\n";
        std::cerr << "  <image_or_folder_path>:\n";
        std::cerr << "      - a regular file -> single-image mode (verbose output + diagnostics)\n";
        std::cerr << "      - a directory    -> folder mode, expects layout\n";
        std::cerr << "                          <root>/0/*.{png,jpg,jpeg}\n";
        std::cerr << "                          ...\n";
        std::cerr << "                          <root>/9/*.{png,jpg,jpeg}\n\n";
        std::cerr << "  hidden_size: number of neurons in the hidden layer (default: 30).\n";
        std::cerr << "               Supported values: 30, 100.\n";
        std::cerr << "               Selects ../relu<HID>_W{1,2}.csv and ../relu<HID>_b{1,2}.csv\n\n";
        std::cerr << "Examples:\n";
        std::cerr << "  " << argv[0] << " ../img_1.jpg\n";
        std::cerr << "  " << argv[0] << " ../img_1.jpg 100\n";
        std::cerr << "  " << argv[0] << " /path/to/test_root\n";
        std::cerr << "  " << argv[0] << " /path/to/test_root 100\n";
        return 1;
    }

    const std::string inputPath = argv[1];

    int HID_DIM = 30;
    if (argc >= 3) {
        try {
            HID_DIM = std::stoi(argv[2]);
        } catch (const std::exception&) {
            std::cerr << "Error: hidden_size must be an integer (got '"
                      << argv[2] << "').\n";
            return 1;
        }
    }

    if (HID_DIM != 30 && HID_DIM != 100) {
        std::cerr << "Error: unsupported hidden_size " << HID_DIM
                  << ". Supported values: 30, 100.\n";
        return 1;
    }

    const std::string prefix = "../relu" + std::to_string(HID_DIM);
    const std::string W1Path = prefix + "_W1.csv";
    const std::string b1Path = prefix + "_b1.csv";
    const std::string W2Path = prefix + "_W2.csv";
    const std::string b2Path = prefix + "_b2.csv";

    std::cout << "Loading weights for hidden size " << HID_DIM
              << " from " << prefix << "_*.csv ...\n";
    auto W1 = io::LoadCsv2D(W1Path.c_str(), IN_DIM,  HID_DIM);
    auto b1 = io::LoadCsv1D(b1Path.c_str());
    auto W2 = io::LoadCsv2D(W2Path.c_str(), HID_DIM, OUT_DIM);
    auto b2 = io::LoadCsv1D(b2Path.c_str());
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights from " << prefix << "_*.csv\n";
        return 1;
    }

    const bool batchMode = fs::is_directory(inputPath);

    // Single-image mode loads the image up-front so the per-image plain-side
    // diagnostics (which need pixels) can run before we mutate W1/b1 with the
    // scaling step. Batch mode skips this — each image goes through the
    // PixelLoader callback inside the harness instead.
    std::vector<std::int64_t> pixels;
    if (!batchMode) {
        std::cout << "Loading image: " << inputPath << "\n";
        // ReLU networks here are trained on binary {0, 1} pixels, not bipolar
        // {-1, +1}, so use the dedicated {0, 1} loader.
        pixels = io::LoadImageBinary(inputPath.c_str(), IN_DIM);
        if (pixels.empty()) return 1;

        // ── Plain-side magnitude diagnostics (raw, unscaled weights) ─────
        // Quick sanity check that explains why FHE results may be wrong:
        // the final decryption is mod pOutput = 1024, interpreted as a
        // signed window (-512, +512). Any class score outside that window
        // will wrap and destroy the argmax.
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
    //    argmax is invariant under any positive scalar. Required regardless
    //    of single-image or batch mode — the FHE side will not produce a
    //    valid argmax without it.
    constexpr double HIDDEN_SCALE_K = 4.0;
    constexpr double OUTPUT_SCALE_K = 64.0;
    for (auto& row : W1) for (auto& w : row) w /= HIDDEN_SCALE_K;
    for (auto& v : b1)                       v /= HIDDEN_SCALE_K;
    for (auto& row : W2) for (auto& w : row) w /= OUTPUT_SCALE_K;
    for (auto& v : b2)                       v /= OUTPUT_SCALE_K;
    std::printf("[Diag] scaled W1/b1 by 1/%.1f, W2/b2 by 1/%.1f (argmax invariant)\n",
                HIDDEN_SCALE_K, OUTPUT_SCALE_K);

    if (!batchMode) {
        // Re-print magnitudes after scaling so we can confirm the choice
        // fits both the LUT and the decoder windows.
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

    // ── Batch mode: dispatch into the shared accuracy harness ─────────────
    if (batchMode) {
        auto loadPixels = [](const std::string& p) {
            return io::LoadImageBinary(p.c_str(), IN_DIM);
        };
        auto result = io::RunAccuracyLoop(net, inputPath, loadPixels, OUT_DIM);
        io::PrintAccuracySummary(result);
        BENCH_MEM("end");
        return 0;
    }

    // ── Single-image mode (existing behavior) ─────────────────────────────
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
