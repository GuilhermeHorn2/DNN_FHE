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

// Like ReportRSS, but ALSO feeds a per-tag aggregator (see RecordLiveRSS)
// so PrintSummary can include a "memory checkpoints" table with the avg /
// max live RSS observed at each tag across a batch. Use this for markers
// you want to appear in the summary; use ReportRSS for one-shot points.
void ReportRSSAndRecord(const char* tag);

// Scoped timer. Logs "[BENCH] <tag>: <ms> ms" on destruction.
// When BENCH_MEMORY is enabled, also reports peak RSS at exit of scope.
//
// On destruction the timer ALSO feeds the per-tag aggregator below; this lets
// us print a single `[BENCH SUMMARY]` table at the end of a batch with avg /
// min / max time and the worst peak RSS observed at that scope.
class ScopedTimer {
public:
    explicit ScopedTimer(const char* tag);
    // Variant that owns its tag string. Use BENCH_LAYER_SCOPE_S when the tag
    // is computed at runtime (e.g. "Layer 1", "Layer 2") so it stays alive
    // for the timer's lifetime without lifetime juggling at the call site.
    explicit ScopedTimer(std::string tag);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string                                    ownedTag_;   // empty when using const-char ctor
    const char*                                    tag_;
    std::chrono::steady_clock::time_point          start_;
};

// ── Aggregator ──────────────────────────────────────────────────────────────
//
// `ScopedTimer::~ScopedTimer` records every (tag, elapsed-ms, peak-RSS) triple
// here. Call `PrintSummary` once at the end of a batch to get the four-row
// table (Layer 1 / Bootstrap + Activation / Layer 2 / Inference (Run)).
//
// peak-RSS is only recorded when BENCH_MEMORY is enabled at compile time;
// otherwise the column shows "—" in the summary.

void RecordSample(const char* tag, double ms, std::size_t peak_rss_bytes);

// Records a (live RSS, peak RSS) sample under `tag` into a separate, marker-
// keyed aggregator. PrintSummary appends a second table that reports
// avg / max live RSS and max peak RSS per tag, in registration order.
//
// Like RecordSample, peak_rss_bytes is the global VmHWM at the moment of the
// reading; live_rss_bytes is VmRSS (current). Both come from /proc/self/status
// on Linux; they're forwarded as-is and only converted to MB at print time.
void RecordLiveRSS(const char* tag, std::size_t live_rss_bytes,
                   std::size_t peak_rss_bytes);

// Prints the four-row summary. `title` is just decorative ("Batch", "Single",
// …). Safe to call multiple times — does not reset state.
void PrintSummary(const char* title = nullptr);

void ResetStats();

}  // namespace bench

// ── Compile-time switches (zero overhead when disabled) ─────────────────────
//
// Activate at configure time with e.g. `cmake -DBENCH_LAYERS=ON ..`.
// Each macro below collapses to `((void)0)` when its switch is off.

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
// String-tag variant: takes anything convertible to std::string and lets the
// timer own the storage. Brace-init avoids the most-vexing-parse when the
// argument expression itself looks like a declarator.
#define BENCH_LAYER_SCOPE_S(tag)  ::bench::ScopedTimer BENCH_VAR(_bl_){std::string(tag)}
#else
#define BENCH_LAYER_SCOPE(tag)    ((void)0)
#define BENCH_LAYER_SCOPE_S(tag)  ((void)0)
#endif

#ifdef BENCH_MEMORY
#define BENCH_MEM(tag)     ::bench::ReportRSS(tag)
// Marker variant: prints the per-image [BENCH][MEM] line AND feeds the
// memory-checkpoint aggregator so PrintSummary's second table averages it.
#define BENCH_MEM_REC(tag) ::bench::ReportRSSAndRecord(tag)
#else
#define BENCH_MEM(tag)     ((void)0)
#define BENCH_MEM_REC(tag) ((void)0)
#endif

#endif  // MVB_BENCH_H
