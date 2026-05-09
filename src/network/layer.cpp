#include "network/layer.h"

#include "bench/bench.h"

#include "math/hermite.h"

#include <iostream>
#include <stdexcept>

namespace fhednn {

using namespace lbcrypto;

namespace {

int NextPowerOfTwo(int n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Level argument used for every plaintext encoding inside a slot-space block.
// Matches the original code's `totalDepth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation`.
std::uint32_t SlotPlaintextLevel(const FHEContext& ctx) {
    return ctx.totalDepth()
         - ctx.params().lvlb[1]
         - ctx.params().levelsAvailableAfterBootstrap
         - ctx.levelsComputation();
}

}  // namespace

// ── LinearLayer ─────────────────────────────────────────────────────────────

LinearLayer::LinearLayer(std::vector<std::vector<double>> W,
                         std::vector<double>              b,
                         int                              inDim,
                         int                              outDim)
    : W_(std::move(W)), b_(std::move(b)), inDim_(inDim), outDim_(outDim) {
    if (static_cast<int>(W_.size()) != outDim_) {
        std::cerr << "[WARNING] LinearLayer: W has " << W_.size()
                  << " rows, expected " << outDim_ << "\n";
    }
    if (static_cast<int>(b_.size()) != outDim_) {
        std::cerr << "[WARNING] LinearLayer: b has " << b_.size()
                  << " entries, expected " << outDim_ << "\n";
    }
}

Ciphertext<DCRTPoly> LinearLayer::Apply(FHEContext&                          ctx,
                                        ForwardState&                        state,
                                        const Ciphertext<DCRTPoly>&          in) {
    // No timer here: per-layer scoping lives in Network::Run, which can label
    // each call with its position ("Layer 1", "Layer 2") rather than a generic
    // "Linear" tag that collides for every fully-connected layer in the net.

    auto&             cc           = ctx.cc();
    const std::uint32_t numSlotsCKKS = ctx.numSlotsCKKS();
    const std::uint32_t plaintextLvl = SlotPlaintextLevel(ctx);
    const double      scaleTHI     = static_cast<double>(ctx.params().scaleTHI);
    const int         padded       = NextPowerOfTwo(inDim_);

    Ciphertext<DCRTPoly> ct_out;

    for (int j = 0; j < outDim_; ++j) {
        std::cout << "  [" << Name() << " " << state.linearIndex << "] neuron "
                  << (j + 1) << "/" << outDim_ << "\r" << std::flush;

        std::vector<double> w_vec(W_[j].begin(), W_[j].end());
        w_vec.resize(padded);
        Plaintext pt_weights = cc->MakeCKKSPackedPlaintext(
            w_vec, 1, plaintextLvl, nullptr, numSlotsCKKS);

        auto ct_dot = cc->EvalMult(in, pt_weights);
        auto ct_sum = cc->EvalSum(ct_dot, padded);
        cc->ModReduceInPlace(ct_sum);

        std::vector<double> mask_vec(padded, 0.0);
        mask_vec[j] = 1.0;
        Plaintext pt_mask = cc->MakeCKKSPackedPlaintext(
            mask_vec, 1, plaintextLvl, nullptr, numSlotsCKKS);

        auto ct_n = cc->EvalMult(ct_sum, pt_mask);
        cc->ModReduceInPlace(ct_n);

        std::vector<double> b_vec(padded, 0.0);
        b_vec[j] = b_[j] / scaleTHI;
        Plaintext pt_bias = cc->MakeCKKSPackedPlaintext(
            b_vec, 1, plaintextLvl, nullptr, numSlotsCKKS);
        cc->EvalAddInPlace(ct_n, pt_bias);

        if (j == 0) ct_out = ct_n;
        else        cc->EvalAddInPlace(ct_out, ct_n);
    }
    std::cout << "\n";

    state.linearIndex += 1;
    return ct_out;
}

// ── ActivationLayer ─────────────────────────────────────────────────────────

ActivationLayer::ActivationLayer(Activation act) : act_(std::move(act)) {}

void ActivationLayer::EnsureCoeffs(const FHEContext& ctx) {
    if (coeffsInitialized_) return;
    coeffs_ = GetHermiteTrigCoefficients(
        act_.f, act_.pInput.ConvertToInt(), act_.order, ctx.params().scaleTHI);
    coeffsInitialized_ = true;
}

Ciphertext<DCRTPoly> ActivationLayer::Apply(FHEContext&                          ctx,
                                            ForwardState&                        state,
                                            const Ciphertext<DCRTPoly>&          in) {
    // No timer here: scope owned by Network::Run with the friendly
    // "Bootstrap + Activation" tag so per-call print and summary line up.

    EnsureCoeffs(ctx);

    auto&               cc           = ctx.cc();
    const std::uint32_t numSlotsCKKS = ctx.numSlotsCKKS();
    const std::uint32_t plaintextLvl = SlotPlaintextLevel(ctx);
    const double        scaleTHI     = static_cast<double>(ctx.params().scaleTHI);

    Ciphertext<DCRTPoly> ct = in;

    // ── Pre-shift to make every value in [0, pInput) ────────────────────
    if (act_.preShift != 0) {
        std::vector<double> shift_vec(numSlotsCKKS,
                                      static_cast<double>(act_.preShift) / scaleTHI);
        Plaintext pt_shift = cc->MakeCKKSPackedPlaintext(
            shift_vec, 1, plaintextLvl, nullptr, numSlotsCKKS);
        cc->EvalAddInPlace(ct, pt_shift);
    }

    // ── Exit slot space ─────────────────────────────────────────────────
    ct = cc->EvalHomDecoding(ct, scaleTHI, 0);

    // ── CKKS -> RLWE -> CKKS refresh ────────────────────────────────────
    auto polys = SchemeletRLWEMP::ConvertCKKSToRLWE(ct, ctx.params().Q);
    const std::uint32_t refreshLvl =
        ctx.totalDepth() - (ctx.params().levelsAvailableBeforeBootstrap > 0);
    auto refreshed = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, polys, ctx.keyPair().publicKey, ctx.params().BIGQ,
        numSlotsCKKS, refreshLvl);

    // ── Re-enter slot space with the activation's LUT ───────────────────
    auto powers = cc->EvalMVBPrecompute(
        refreshed, coeffs_, act_.pInput.GetMSB() - 1,
        ctx.elementParams()->GetModulus(), act_.order);
    auto out = cc->EvalMVBNoDecoding(
        powers, coeffs_, act_.pInput.GetMSB() - 1, act_.order);

    state.activationIndex += 1;
    return out;
}

// ── DummyMultLayer ──────────────────────────────────────────────────────────

Ciphertext<DCRTPoly> DummyMultLayer::Apply(FHEContext&                          ctx,
                                           ForwardState&                        /*state*/,
                                           const Ciphertext<DCRTPoly>&          in) {
    BENCH_LAYER_SCOPE("DummyMult");

    if (count_ == 0) return in;

    auto&               cc           = ctx.cc();
    const std::uint32_t numSlotsCKKS = ctx.numSlotsCKKS();
    const std::uint32_t plaintextLvl = SlotPlaintextLevel(ctx);

    Ciphertext<DCRTPoly> ct = in;
    std::vector<double>  ones(numSlotsCKKS, 1.0);
    Plaintext            pt_one = cc->MakeCKKSPackedPlaintext(
        ones, 1, plaintextLvl, nullptr, numSlotsCKKS);

    for (std::uint32_t i = 0; i < count_; ++i) {
        ct = cc->EvalMult(ct, pt_one);
        cc->ModReduceInPlace(ct);
    }
    return ct;
}

}  // namespace fhednn
