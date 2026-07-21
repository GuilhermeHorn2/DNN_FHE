#ifndef MVB_BENCH_H
#define MVB_BENCH_H

#include <chrono>
#include <cstddef>
#include <string>

namespace bench {

// Returns peak resident set size in bytes (Linux/WSL via /proc/self/status).
// Returns 0 if unsupported on the current platform.
std::size_t PeakRSSBytes();

// Returns current resident set size in bytes (live RSS, not peak).
std::size_t CurrentRSSBytes();

// One-line report at an instrumentation point.
void ReportRSS(const char* tag);

// Like ReportRSS, but also feeds the RecordLiveRSS aggregator for PrintSummary.
void ReportRSSAndRecord(const char* tag);

// Scoped timer. Logs "[BENCH] <tag>: <ms> ms" on destruction and feeds the
// per-tag aggregator below (see PrintSummary). When BENCH_MEMORY is enabled,
// also reports peak RSS at exit of scope.
class ScopedTimer {
public:
    explicit ScopedTimer(const char* tag);
    // Owns its tag string; use BENCH_LAYER_SCOPE_S for runtime-computed tags.
    explicit ScopedTimer(std::string tag);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string                                    ownedTag_;   // empty when using const-char ctor
    const char*                                    tag_;
    std::chrono::steady_clock::time_point          start_;
};

// Aggregator
//
// Backs the `[BENCH SUMMARY]` table (Layer 1 / Bootstrap + Activation /
// Layer 2 / Inference (Run)). peak-RSS is only recorded when BENCH_MEMORY
// is enabled; otherwise the column shows "—".
void RecordSample(const char* tag, double ms, std::size_t peak_rss_bytes);

// Records a (live RSS, peak RSS) sample under `tag` into a separate,
// marker-keyed aggregator; PrintSummary appends a "memory checkpoints"
// table (avg/max live RSS, max peak RSS) in registration order.
void RecordLiveRSS(const char* tag, std::size_t live_rss_bytes,
                   std::size_t peak_rss_bytes);

// Prints the four-row summary. `title` is just decorative ("Batch", "Single",
// …). Safe to call multiple times — does not reset state.
void PrintSummary(const char* title = nullptr);

void ResetStats();

}  // namespace bench

// Compile-time switches (zero overhead when disabled).
// Activate at configure time, e.g. `cmake -DBENCH_LAYERS=ON ..`.

#define BENCH_CONCAT2(a, b) a##b
#define BENCH_CONCAT(a, b)  BENCH_CONCAT2(a, b)
#define BENCH_VAR(prefix)   BENCH_CONCAT(prefix, __LINE__)

#ifdef BENCH_TOTAL
#define BENCH_TOTAL_SCOPE(tag) ::bench::ScopedTimer BENCH_VAR(_bt_)(tag)
#else
#define BENCH_TOTAL_SCOPE(tag) ((void)0)
#endif

#ifdef BENCH_INFERENCE
#define BENCH_INFERENCE_SCOPE(tag) ::bench::ScopedTimer BENCH_VAR(_bi_)(tag)
#else
#define BENCH_INFERENCE_SCOPE(tag) ((void)0)
#endif

#ifdef BENCH_BOOTSTRAP
#define BENCH_BOOTSTRAP_SCOPE(tag) ::bench::ScopedTimer BENCH_VAR(_bb_)(tag)
#else
#define BENCH_BOOTSTRAP_SCOPE(tag) ((void)0)
#endif

#ifdef BENCH_LAYERS
#define BENCH_LAYER_SCOPE(tag)    ::bench::ScopedTimer BENCH_VAR(_bl_)(tag)
// Brace-init avoids the most-vexing-parse for runtime-computed tags.
#define BENCH_LAYER_SCOPE_S(tag)  ::bench::ScopedTimer BENCH_VAR(_bl_){std::string(tag)}
#else
#define BENCH_LAYER_SCOPE(tag)    ((void)0)
#define BENCH_LAYER_SCOPE_S(tag)  ((void)0)
#endif

#ifdef BENCH_MEMORY
#define BENCH_MEM(tag)     ::bench::ReportRSS(tag)
// Also feeds the memory-checkpoint aggregator for PrintSummary's second table.
#define BENCH_MEM_REC(tag) ::bench::ReportRSSAndRecord(tag)
#else
#define BENCH_MEM(tag)     ((void)0)
#define BENCH_MEM_REC(tag) ((void)0)
#endif

#endif  // MVB_BENCH_H
