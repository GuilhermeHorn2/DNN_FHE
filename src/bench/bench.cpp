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

void ReportRSSAndRecord(const char* tag) {
    const std::size_t cur  = CurrentRSSBytes();
    const std::size_t peak = PeakRSSBytes();
    std::printf("[BENCH][MEM] %-32s  RSS=%.2f MB   peak=%.2f MB\n",
                tag,
                cur  / (1024.0 * 1024.0),
                peak / (1024.0 * 1024.0));
    std::fflush(stdout);
    RecordLiveRSS(tag, cur, peak);
}

// Aggregator (singleton, mutex-protected)

namespace {

struct TagStats {
    std::vector<double> samples_ms;
    // Worst peak RSS observed at scope exit; 0 without BENCH_MEMORY.
    std::size_t         peak_rss_max_bytes = 0;
};

// One entry per tag passed to RecordLiveRSS / ReportRSSAndRecord, in
// insertion order so PrintSummary can list checkpoints chronologically.
struct MemTagStats {
    std::vector<std::size_t> live_rss_samples;   // VmRSS at each call
    std::vector<std::size_t> peak_rss_samples;   // VmHWM at each call
};

std::mutex&                       Mu()    { static std::mutex m;                            return m; }
std::map<std::string, TagStats>&  Bag()   { static std::map<std::string, TagStats> b;        return b; }

std::map<std::string, MemTagStats>&  MemBag()   { static std::map<std::string, MemTagStats> b;     return b; }
std::vector<std::string>&            MemOrder() { static std::vector<std::string>           v;     return v; }

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

void RecordLiveRSS(const char* tag, std::size_t live_rss_bytes,
                   std::size_t peak_rss_bytes) {
    if (!tag) return;
    std::lock_guard<std::mutex> lk(Mu());
    auto& bag = MemBag();
    auto  it  = bag.find(tag);
    if (it == bag.end()) {
        MemOrder().emplace_back(tag);
        it = bag.emplace(tag, MemTagStats{}).first;
    }
    it->second.live_rss_samples.push_back(live_rss_bytes);
    it->second.peak_rss_samples.push_back(peak_rss_bytes);
}

void ResetStats() {
    std::lock_guard<std::mutex> lk(Mu());
    Bag().clear();
    MemBag().clear();
    MemOrder().clear();
}

// Fixed display order: only these four stages appear in the summary table.
// Other tags still print their per-call line via ScopedTimer.
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

    // Memory checkpoints sub-table: one row per marker, in registration
    // order. `avg ΔRSS` is this row's avg RSS minus the previous row's
    // (positive = allocation, negative = release); the first row has none.
    if (!MemOrder().empty()) {
        // Pick max sample count across all markers (they should all match,
        // but be defensive in case a marker was added mid-run).
        std::size_t n_mem = 0;
        for (const auto& tag : MemOrder()) {
            auto it = MemBag().find(tag);
            if (it != MemBag().end()) {
                n_mem = std::max(n_mem, it->second.live_rss_samples.size());
            }
        }
        if (n_mem > 0) {
            if (title && title[0]) {
                std::printf("[BENCH MEMORY CHECKPOINTS %s]  n=%zu\n\n",
                            title, n_mem);
            } else {
                std::printf("[BENCH MEMORY CHECKPOINTS]  n=%zu\n\n", n_mem);
            }

            std::printf("  %-32s   %15s    %15s    %15s    %15s\n",
                        "Marker",
                        "avg RSS (MB)", "avg ΔRSS (MB)",
                        "max RSS (MB)", "max peak (MB)");
            std::printf("  %-32s   %15s    %15s    %15s    %15s\n",
                        "--------------------------------",
                        "---------------",
                        "---------------",
                        "---------------",
                        "---------------");

            bool   have_prev    = false;
            double prev_avg_mb  = 0.0;
            for (const auto& tag : MemOrder()) {
                auto it = MemBag().find(tag);
                if (it == MemBag().end() ||
                    it->second.live_rss_samples.empty()) continue;
                const auto& live = it->second.live_rss_samples;
                const auto& peak = it->second.peak_rss_samples;
                const double sum_live =
                    std::accumulate(live.begin(), live.end(),
                                    static_cast<std::size_t>(0));
                const double avg_live_mb =
                    (sum_live / static_cast<double>(live.size()))
                    / (1024.0 * 1024.0);
                const double max_live_mb =
                    static_cast<double>(*std::max_element(live.begin(),
                                                          live.end()))
                    / (1024.0 * 1024.0);
                const double max_peak_mb =
                    peak.empty()
                        ? 0.0
                        : static_cast<double>(*std::max_element(peak.begin(),
                                                                peak.end()))
                          / (1024.0 * 1024.0);
                if (have_prev) {
                    const double delta_mb = avg_live_mb - prev_avg_mb;
                    // Sign-prefixed delta (+ / -) so direction is obvious.
                    char delta_buf[32];
                    std::snprintf(delta_buf, sizeof(delta_buf),
                                  "%+.2f", delta_mb);
                    std::printf("  %-32s   %15.2f    %15s    %15.2f    %15.2f\n",
                                tag.c_str(),
                                avg_live_mb, delta_buf,
                                max_live_mb, max_peak_mb);
                } else {
                    std::printf("  %-32s   %15.2f    %15s    %15.2f    %15.2f\n",
                                tag.c_str(),
                                avg_live_mb, "—",
                                max_live_mb, max_peak_mb);
                }
                prev_avg_mb = avg_live_mb;
                have_prev   = true;
            }
            std::printf("\n");
        }
    }

    std::fflush(stdout);
}

// ScopedTimer

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
