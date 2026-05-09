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

// MNIST-like DiNN dimensions: 784 -> HID_DIM -> 10. The hidden width is a
// runtime parameter so the same driver can load DiNN30, DiNN100, or any
// other size whose weight files exist next to this binary.
static constexpr int IN_DIM  = 784;
static constexpr int OUT_DIM = 10;

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " <image_or_folder_path> [hidden_size]\n\n";
        std::cerr << "  <image_or_folder_path>:\n";
        std::cerr << "      - a regular file -> single-image mode (verbose output)\n";
        std::cerr << "      - a directory    -> folder mode, expects layout\n";
        std::cerr << "                          <root>/0/*.{png,jpg,jpeg}\n";
        std::cerr << "                          ...\n";
        std::cerr << "                          <root>/9/*.{png,jpg,jpeg}\n\n";
        std::cerr << "  hidden_size: number of neurons in the hidden layer (default: 30).\n";
        std::cerr << "               Supported values: 30, 100.\n";
        std::cerr << "               Selects ../signal<HID>_W{1,2}.csv and ../signal<HID>_b{1,2}.csv\n\n";
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

    const std::string prefix = "../signal" + std::to_string(HID_DIM);
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

    // ── Batch mode: inputPath is a directory of <label>/*.{png,jpg,jpeg} ──
    // The harness only knows how to walk + score; it asks us how to turn a
    // single image path into the {-1,+1} pixel vector this Sign network
    // expects, and (optionally) how to score the same pixels in plaintext
    // so it can report FHE vs plain agreement.
    if (fs::is_directory(inputPath)) {
        auto loadPixels = [](const std::string& p) {
            return io::LoadImageBipolar(p.c_str(), IN_DIM);
        };

        // Plain-side reference: same topology as the FHE network (Linear ->
        // Sign -> Linear). Captures W1/b1/W2/b2 by reference; they are not
        // mutated after Compile() in this driver, so this is safe.
        auto plainScore = [&, HID_DIM](const std::vector<std::int64_t>& px) {
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

        auto result = io::RunAccuracyLoop(net, inputPath, loadPixels, OUT_DIM, plainScore);
        io::PrintAccuracySummary(result);
        BENCH_MEM("end");
        return 0;
    }

    // ── Single-image mode (existing behavior) ─────────────────────────────
    std::cout << "Loading image: " << inputPath << "\n";
    auto pixels = io::LoadImageBipolar(inputPath.c_str(), IN_DIM);
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
