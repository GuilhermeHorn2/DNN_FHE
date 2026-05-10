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

    // ── Plaintext-side pre-scaling (argmax invariant) ─────────────────────
    //
    // HIDDEN_SCALE_K — keeps the hidden pre-activations y[j] = W1·x + b1
    //   inside the ReLU LUT's safe range [-preShift, pInput - preShift).
    //   With preShift = 512 and pInput = 1024 the safe range is [-512, +512)
    //   (centered on 0). Empirical max|y| ≈ 993 across the 50-image batch,
    //   so K1 ≥ 993/512 ≈ 1.94. We use 4 (next pow2 above 2) for headroom
    //   and to push y/K1 deeper into the cleanest interior of the LUT, away
    //   from the period boundaries where Gibbs ringing is largest.
    //   preShift = 512 also minimizes the LUT amplitude (pInput - preShift
    //   = 512), which is the dominant scale factor for the order-N
    //   Hermite-trig fit error per slot.
    //
    // OUTPUT_SCALE_K — keeps the final scores inside the decoder's signed
    //   window (-pOutput/2, +pOutput/2) = (-512, +512). With raw weights
    //   max|score| ≈ 57k; after dividing by HIDDEN_SCALE_K · OUTPUT_SCALE_K
    //   the constraint is 57000 / (4·K2) < 512, i.e. K2 > 27.8 → 64 for
    //   power-of-two safety margin. Without this scaling the FHE scores wrap
    //   modulo pOutput and the argmax becomes meaningless.
    //
    // The original W1/b1/W2/b2 are kept untouched so the plaintext oracle
    // computes scores at full integer magnitude. We multiply the FHE scores
    // by HIDDEN_SCALE_K · OUTPUT_SCALE_K before printing so plain[k] and
    // fhe[k] live on the same scale.
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

    // ── Batch mode: inputPath is a directory of <label>/*.{png,jpg,jpeg} ──
    if (fs::is_directory(inputPath)) {
        auto loadPixels = [](const std::string& p) {
            return io::LoadImageBinary(p.c_str(), IN_DIM);
        };

        // Plain-side reference: same topology as the FHE network (Linear ->
        // ReLU -> Linear). Captures W1/b1/W2/b2 by reference; they are not
        // mutated after Compile() in this driver, so this is safe.
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

    // ── Single-image mode (existing behavior) ─────────────────────────────
    std::cout << "Loading image: " << inputPath << "\n";
    // ReLU networks here are trained on binary {0, 1} pixels, not bipolar
    // {-1, +1}, so use the dedicated {0, 1} loader.
    auto pixels = io::LoadImageBinary(inputPath.c_str(), IN_DIM);
    if (pixels.empty()) return 1;

    auto scores = net.Run(pixels);

    // The FHE pipeline saw W1/b1 divided by HIDDEN_SCALE_K and W2/b2 divided
    // by OUTPUT_SCALE_K, so each returned integer score is approximately
    // score_true / (HIDDEN_SCALE_K · OUTPUT_SCALE_K). Lift back to the
    // full-integer scale so the printed numbers are directly comparable to
    // the plaintext oracle below.
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


    for (int j=0; j<OUT_DIM; ++j) std::printf(" plain[%d] = %.0f fhe[%d] = %lld\n", j, refScores[j], j, (long long)scores[j]);


    std::cout << "\n================================\n";
    std::printf(" PREDICTED DIGIT  : %d\n", predicted);
    std::printf(" REFERENCE (plain): %d  %s\n", refPred,
                predicted == refPred ? "(matches)" : "(MISMATCH)");
    std::cout << "================================\n";

    BENCH_MEM("end");
    return 0;
}
