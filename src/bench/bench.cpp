#include "bench/bench.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bench {

// ── Process RSS helpers ─────────────────────────────────────────────────────

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

// ── Aggregator (singleton, mutex-protected) ────────────────────────────────

namespace {

struct TagStats {
    std::vector<double> samples_ms;
    // Worst peak RSS observed at scope exit of this tag. Stays 0 when the
    // build was made without BENCH_MEMORY (no peak captured).
    std::size_t         peak_rss_max_bytes = 0;
};

std::mutex&                       Mu()    { static std::mutex m;                            return m; }
std::map<std::string, TagStats>&  Bag()   { static std::map<std::string, TagStats> b;        return b; }

}  // namespace

void RecordSample(const char* tag, double ms, std::size_t peak_rss_bytes) {
    if (!tag) return;
    std::lock_guard<std::mutex> lk(Mu());
    auto& s = Bag()[tag];
    s.samples_ms.push_back(ms);
    if (peak_rss_bytes > s.peak_rss_max_bytes) {
        s.peak_rss_max_bytes = peak_rss_bytes;
    }
}

void ResetStats() {
    std::lock_guard<std::mutex> lk(Mu());
    Bag().clear();
}

// Fixed display order. The summary is intentionally narrow: it reports only
// the four stages the experiment cares about. Other tags (Total program,
// InputEncoder, OutputDecoder, …) still get their per-call line via
// ScopedTimer but are NOT included in the summary table.
namespace {
struct Row { const char* display; const char* tag; };
constexpr Row kRows[] = {
    {"Layer 1",                "Layer 1"               },
    {"Bootstrap + Activation", "Bootstrap + Activation"},
    {"Layer 2",                "Layer 2"               },
    {"Inference (Run)",        "Inference (Run)"       },
};
}  // namespace

void PrintSummary(const char* title) {
    std::lock_guard<std::mutex> lk(Mu());

    // n = max samples seen on any of the four reported tags; if none have
    // any samples, print nothing rather than a table of em-dashes.
    std::size_t n_any = 0;
    for (const auto& r : kRows) {
        auto it = Bag().find(r.tag);
        if (it != Bag().end()) {
            n_any = std::max(n_any, it->second.samples_ms.size());
        }
    }
    if (n_any == 0) return;

    if (title && title[0]) {
        std::printf("\n[BENCH SUMMARY %s]  n=%zu\n\n", title, n_any);
    } else {
        std::printf("\n[BENCH SUMMARY]  n=%zu\n\n", n_any);
    }

    std::printf("  %-26s   %10s    %10s    %10s    %15s\n",
                "Stage", "avg (ms)", "min (ms)", "max (ms)", "peak RSS (MB)");
    std::printf("  %-26s   %10s    %10s    %10s    %15s\n",
                "------------------------",
                "----------", "----------", "----------", "---------------");

    for (const auto& r : kRows) {
        auto it = Bag().find(r.tag);
        if (it == Bag().end() || it->second.samples_ms.empty()) {
            std::printf("  %-26s   %10s    %10s    %10s    %15s\n",
                        r.display, "—", "—", "—", "—");
            continue;
        }
        const auto& v   = it->second.samples_ms;
        const double sum = std::accumulate(v.begin(), v.end(), 0.0);
        const double avg = sum / static_cast<double>(v.size());
        const double mn  = *std::min_element(v.begin(), v.end());
        const double mx  = *std::max_element(v.begin(), v.end());
        if (it->second.peak_rss_max_bytes > 0) {
            const double rss_mb =
                it->second.peak_rss_max_bytes / (1024.0 * 1024.0);
            std::printf("  %-26s   %10.2f    %10.2f    %10.2f    %15.2f\n",
                        r.display, avg, mn, mx, rss_mb);
        } else {
            std::printf("  %-26s   %10.2f    %10.2f    %10.2f    %15s\n",
                        r.display, avg, mn, mx, "—");
        }
    }
    std::printf("\n");
    std::fflush(stdout);
}

// ── ScopedTimer ────────────────────────────────────────────────────────────

ScopedTimer::ScopedTimer(const char* tag)
    : ownedTag_(), tag_(tag), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::ScopedTimer(std::string tag)
    : ownedTag_(std::move(tag)),
      tag_(ownedTag_.c_str()),
      start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
    const auto end = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(end - start_).count();
#ifdef BENCH_MEMORY
    const std::size_t peak = PeakRSSBytes();
    RecordSample(tag_, ms, peak);
    std::printf("[BENCH] %-32s  %10.3f ms   (peak RSS %.2f MB)\n",
                tag_, ms, peak / (1024.0 * 1024.0));
#else
    RecordSample(tag_, ms, 0);
    std::printf("[BENCH] %-32s  %10.3f ms\n", tag_, ms);
#endif
    std::fflush(stdout);
}

}  // namespace bench
