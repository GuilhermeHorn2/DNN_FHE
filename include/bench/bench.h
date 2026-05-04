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

// Scoped timer. Logs "[BENCH] <tag>: <ms> ms" on destruction.
// When BENCH_MEMORY is enabled, also reports peak RSS at exit of scope.
class ScopedTimer {
public:
    explicit ScopedTimer(const char* tag);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&)            = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    const char*                                    tag_;
    std::chrono::steady_clock::time_point          start_;
};

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
#define BENCH_LAYER_SCOPE(tag) ::bench::ScopedTimer BENCH_VAR(_bl_)(tag)
#else
#define BENCH_LAYER_SCOPE(tag) ((void)0)
#endif

#ifdef BENCH_MEMORY
#define BENCH_MEM(tag) ::bench::ReportRSS(tag)
#else
#define BENCH_MEM(tag) ((void)0)
#endif

#endif  // MVB_BENCH_H
