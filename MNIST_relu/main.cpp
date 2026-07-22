#include "bench/bench.h"
#include "io/accuracy.h"
#include "io/csv.h"
#include "io/image.h"
#include "network/activation.h"
#include "network/context.h"
#include "network/network.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// MNIST DiNN: 784 -> HID_DIM -> 10, ReLU variant.
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

    using namespace fhednn;

    FHEParams params;
    params.BIGQ    = lbcrypto::BigInteger(1) << 55;
    params.Q       = lbcrypto::BigInteger(1) << 55;
    FHEContext ctx{params};

    // Pre-scale so both stages stay in range; weights are divided, originals
    // kept for the plaintext oracle.
    //   K1: max|W1*x+b1| ~ 993 must fit the ReLU LUT window +-512  -> K1 = 4
    //   K2: max|score| ~ 57k must fit the decoder window +-512     -> K2 = 64
    // Without K2 the FHE scores wrap mod pOutput and argmax is meaningless.
    constexpr double HIDDEN_SCALE_K = 4.0;
    constexpr double OUTPUT_SCALE_K = 64.0;
    auto W1_scaled = W1;
    auto b1_scaled = b1;
    auto W2_scaled = W2;
    auto b2_scaled = b2;
    for (auto& row : W1_scaled) for (auto& w : row) w /= HIDDEN_SCALE_K;
    for (auto& v   : b1_scaled)                     v /= HIDDEN_SCALE_K;
    for (auto& row : W2_scaled) for (auto& w : row) w /= OUTPUT_SCALE_K;
    for (auto& v   : b2_scaled)                     v /= OUTPUT_SCALE_K;
    std::printf("[Scale] HIDDEN_SCALE_K = %.2f  OUTPUT_SCALE_K = %.2f  "
                "(pOutput = %llu, BIGQ = 2^%u)\n",
                HIDDEN_SCALE_K, OUTPUT_SCALE_K,
                static_cast<unsigned long long>(
                    ctx.params().pOutput.ConvertToInt<std::uint64_t>()),
                ctx.params().BIGQ.GetMSB() - 1);

    Network    net;
    net.Linear(W1_scaled, b1_scaled)
       .Activate(activations::ReLU(/*preShift=*/256, ctx.params().pInput))
       .Linear(W2_scaled, b2_scaled);

    net.Compile(ctx);
    BENCH_MEM("after-compile");

    // Batch mode: inputPath is a directory of <label>/*.{png,jpg,jpeg}.
    if (fs::is_directory(inputPath)) {
        auto loadPixels = [](const std::string& p) {
            return io::LoadImageBinary(p.c_str(), IN_DIM);
        };

        // Plain reference: Linear -> ReLU -> Linear.
        auto plainScore = [&, HID_DIM](const std::vector<std::int64_t>& px) {
            std::vector<double> hidden(HID_DIM, 0.0);
            for (int j = 0; j < HID_DIM; ++j) {
                double acc = b1[j];
                for (int i = 0; i < IN_DIM; ++i)
                    acc += W1[j][i] * static_cast<double>(px[i]);
                hidden[j] = (acc > 0.0) ? acc : 0.0;
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

    // Single-image mode.
    std::cout << "Loading image: " << inputPath << "\n";
    // Trained on {0,1} pixels, not bipolar.
    auto pixels = io::LoadImageBinary(inputPath.c_str(), IN_DIM);
    if (pixels.empty()) return 1;

    auto scores = net.Run(pixels);

    // Lift FHE scores back to full integer scale for comparison with plain.
    constexpr double SCORE_LIFT = HIDDEN_SCALE_K * OUTPUT_SCALE_K;
    for (auto& v : scores)
        v = static_cast<std::int64_t>(std::llround(static_cast<double>(v) * SCORE_LIFT));

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

    // Plaintext reference.
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


    for (int j=0; j<OUT_DIM; ++j) std::printf(" plain[%d] = %.0f fhe[%d] = %lld\n", j, refScores[j], j, (long long)scores[j]);


    std::cout << "\n================================\n";
    std::printf(" PREDICTED DIGIT  : %d\n", predicted);
    std::printf(" REFERENCE (plain): %d  %s\n", refPred,
                predicted == refPred ? "(matches)" : "(MISMATCH)");
    std::cout << "================================\n";

    BENCH_MEM("end");
    return 0;
}
