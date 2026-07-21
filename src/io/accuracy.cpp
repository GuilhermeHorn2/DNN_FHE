#include "io/accuracy.h"

#include "bench/bench.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace io {

namespace {

template <class Scores>
int ArgMax(const Scores& s, int cap) {
    int best = 0;
    if (s.empty()) return 0;
    auto bestVal = s[0];
    const int n = std::min(static_cast<int>(s.size()), cap);
    for (int j = 1; j < n; ++j) {
        if (s[j] > bestVal) {
            bestVal = s[j];
            best    = j;
        }
    }
    return best;
}

}  // namespace

AccuracyResult RunAccuracyLoop(fhednn::Network&   net,
                               const std::string& testRoot,
                               const PixelLoader& loadPixels,
                               int                outDim,
                               const PlainScorer& plainScorer) {
    AccuracyResult r;
    r.confusion.assign(outDim, std::vector<int>(outDim, 0));
    r.hasPlain = static_cast<bool>(plainScorer);

    for (int label = 0; label < outDim; ++label) {
        const fs::path classDir = fs::path(testRoot) / std::to_string(label);
        if (!fs::exists(classDir) || !fs::is_directory(classDir)) continue;

        for (const auto& entry : fs::directory_iterator(classDir)) {
            if (!entry.is_regular_file()) continue;
            const auto& path = entry.path();
            const auto  ext  = path.extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

            auto pixels = loadPixels(path.string());
            if (pixels.empty()) continue;

            auto fheScores = net.Run(pixels);
            const int predicted = ArgMax(fheScores, outDim);

            ++r.total;
            if (predicted == label) ++r.correct;
            r.confusion[label][predicted] += 1;

            // Only when the project supplied a plaintext scorer.
            if (r.hasPlain) {
                auto plainScores = plainScorer(pixels);
                const int plainPred = ArgMax(plainScores, outDim);
                if (plainPred == label)     ++r.correctPlain;
                if (predicted == plainPred) ++r.fheMatchesPlain;

                std::printf("[%4d] %s -> pred=%d  plain=%d  truth=%d  %s\n",
                            r.total, path.filename().c_str(),
                            predicted, plainPred, label,
                            predicted == label ? "OK" : "MISS");
            } else {
                std::printf("[%4d] %s -> pred=%d  truth=%d  %s\n",
                            r.total, path.filename().c_str(),
                            predicted, label,
                            predicted == label ? "OK" : "MISS");
            }
            std::fflush(stdout);
        }
    }

    // No-op unless a BENCH_* timer flag was enabled at build time.
    bench::PrintSummary("Batch");

    return r;
}

void PrintAccuracySummary(const AccuracyResult& r) {
    auto pct = [&](int n) {
        return r.total > 0 ? 100.0 * n / r.total : 0.0;
    };

    if (r.hasPlain) {
        std::printf("\n=== FHE vs PlainText ===\n");
        std::printf("Hit: %d / %d  (%.2f%%)\n",
                    r.fheMatchesPlain, r.total, pct(r.fheMatchesPlain));

        std::printf("\n=== PlainText Acc ===\n");
        std::printf("Correct: %d / %d  (%.2f%%)\n",
                    r.correctPlain, r.total, pct(r.correctPlain));
    }

    std::printf("\n=== Accuracy ===\n");
    std::printf("Correct: %d / %d  (%.2f%%)\n",
                r.correct, r.total, pct(r.correct));

    if (r.confusion.empty()) return;
    const int outDim = static_cast<int>(r.confusion.size());

    std::printf("\nConfusion matrix (row=truth, col=pred):\n     ");
    for (int j = 0; j < outDim; ++j) std::printf(" %4d", j);
    std::printf("\n");
    for (int i = 0; i < outDim; ++i) {
        std::printf("  %d :", i);
        for (int j = 0; j < outDim; ++j) std::printf(" %4d", r.confusion[i][j]);
        std::printf("\n");
    }
}

}  // namespace io
