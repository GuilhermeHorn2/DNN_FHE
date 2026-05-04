#include "bench/bench.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace bench {

static std::size_t ReadProcStatusKB(const char* key) {
#ifdef __linux__
    std::ifstream f("/proc/self/status");
    if (!f) return 0;
    std::string line;
    const std::size_t key_len = std::string(key).size();
    while (std::getline(f, line)) {
        if (line.compare(0, key_len, key) == 0) {
            std::istringstream iss(line.substr(key_len));
            std::size_t val_kb = 0;
            std::string unit;
            iss >> val_kb >> unit;
            return val_kb;
        }
    }
#else
    (void)key;
#endif
    return 0;
}

std::size_t PeakRSSBytes() {
    return ReadProcStatusKB("VmHWM:") * 1024ULL;
}

std::size_t CurrentRSSBytes() {
    return ReadProcStatusKB("VmRSS:") * 1024ULL;
}

void ReportRSS(const char* tag) {
    const std::size_t cur  = CurrentRSSBytes();
    const std::size_t peak = PeakRSSBytes();
    std::printf("[BENCH][MEM] %-32s  RSS=%.2f MB   peak=%.2f MB\n",
                tag,
                cur  / (1024.0 * 1024.0),
                peak / (1024.0 * 1024.0));
    std::fflush(stdout);
}

ScopedTimer::ScopedTimer(const char* tag)
    : tag_(tag), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    const auto end = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(end - start_).count();
#ifdef BENCH_MEMORY
    const std::size_t peak = PeakRSSBytes();
    std::printf("[BENCH] %-32s  %10.3f ms   (peak RSS %.2f MB)\n",
                tag_, ms, peak / (1024.0 * 1024.0));
#else
    std::printf("[BENCH] %-32s  %10.3f ms\n", tag_, ms);
#endif
    std::fflush(stdout);
}

}  // namespace bench
