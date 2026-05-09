#include "network/network.h"

#include "bench/bench.h"

#include "math/hermite.h"

#include <algorithm>
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

}  // namespace

// ── Builder ─────────────────────────────────────────────────────────────────

Network& Network::SetInputShift(std::int64_t k) {
    inputShift_ = k;
    return *this;
}

Network& Network::Linear(std::vector<std::vector<double>> W,
                         std::vector<double>              b) {
    const int outDim = static_cast<int>(W.size());
    const int inDim  = outDim > 0 ? static_cast<int>(W[0].size()) : 0;
    layers_.push_back(std::make_unique<LinearLayer>(
        std::move(W), std::move(b), inDim, outDim));
    return *this;
}

Network& Network::Activate(Activation act) {
    layers_.push_back(std::make_unique<ActivationLayer>(std::move(act)));
    return *this;
}

Network& Network::Add(std::unique_ptr<Layer> layer) {
    layers_.push_back(std::move(layer));
    return *this;
}

Ciphertext<DCRTPoly> Network::EncryptZeroes() {
    BENCH_LAYER_SCOPE("InputEncoder");

    if (!ctx_) throw std::runtime_error("Network::Run before Compile.");
    auto&         cc           = ctx_->cc();
    const auto&   params       = ctx_->params();
    const auto    numSlotsCKKS = ctx_->numSlotsCKKS();
    const auto    numSlots     = params.numSlots;
    const double  scaleTHI     = static_cast<double>(params.scaleTHI);

    std::vector<std::int64_t> x(numSlots, 0);

    // ── Encrypt RLWE / mod-switch ─────────────────────────────────────────
    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(
        x, params.qBFVInit, params.pInput,
        ctx_->keyPair().secretKey, ctx_->elementParams(), ctx_->flagBR());
    SchemeletRLWEMP::ModSwitch(ctxtBFV, params.Q, params.qBFVInit);

    BENCH_MEM("after-encrypt");

    // ── RLWE -> CKKS ──────────────────────────────────────────────────────
    const std::uint32_t firstLvl =
        ctx_->totalDepth() - (params.levelsAvailableBeforeBootstrap > 0);
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctxtBFV, ctx_->keyPair().publicKey,
        params.BIGQ, numSlotsCKKS, firstLvl);

    // ── Identity EvalMVBNoDecoding (enter slot space) ─────────────────────
    auto identity = activations::Identity(
        params.pInput, params.pOutput, params.hermiteOrder);
    auto coeffIdentity = GetHermiteTrigCoefficients(
        identity.f, identity.pInput.ConvertToInt(), identity.order, params.scaleTHI);

    auto powers = cc->EvalMVBPrecompute(
        ctxt, coeffIdentity, params.pInput.GetMSB() - 1,
        ctx_->elementParams()->GetModulus(), identity.order);
    auto ctxtSlots = cc->EvalMVBNoDecoding(
        powers, coeffIdentity, params.pInput.GetMSB() - 1, identity.order);

    return ctxtSlots;
}

// ── Compile ─────────────────────────────────────────────────────────────────

void Network::Compile(FHEContext& ctx) {
    if (compiled_) {
        throw std::runtime_error("Network::Compile called twice.");
    }
    if (layers_.empty()) {
        throw std::runtime_error("Network::Compile: no layers added.");
    }

    // ── 1. Partition into slot-space blocks (Linear+ separated by Activation).
    //       Compute cost per block, find the deepest.
    std::vector<std::uint32_t>                  blockCosts;
    std::vector<std::pair<std::size_t, std::size_t>> blockRanges;  // [begin, end)
    std::size_t                                 blockBegin = 0;
    std::uint32_t                               currCost   = 0;

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const Layer* L = layers_[i].get();
        if (auto* a = dynamic_cast<const ActivationLayer*>(L)) {
            (void)a;
            blockCosts.push_back(currCost);
            blockRanges.push_back({blockBegin, i});
            blockBegin = i + 1;
            currCost   = 0;
        } else {
            currCost += L->LevelsConsumed();
        }
    }
    // Trailing block (after the last Activation, before the OutputDecoder).
    blockCosts.push_back(currCost);
    blockRanges.push_back({blockBegin, layers_.size()});

    levelsComputation_ = blockCosts.empty()
                       ? 0
                       : *std::max_element(blockCosts.begin(), blockCosts.end());
    if (levelsComputation_ == 0) levelsComputation_ = 1;  // at least one rescale slot

    std::cout << "[Network] " << blockCosts.size() << " slot-space block(s); costs=[";
    for (std::size_t i = 0; i < blockCosts.size(); ++i) {
        std::cout << blockCosts[i] << (i + 1 < blockCosts.size() ? "," : "");
    }
    std::cout << "]; levelsComputation = " << levelsComputation_ << "\n";

    // ── 2. Pad shorter blocks with DummyMultLayer.
    //       We rebuild the layer list in a new vector to avoid invalidating
    //       indices during insertion.
    std::vector<std::unique_ptr<Layer>> newLayers;
    newLayers.reserve(layers_.size() + blockCosts.size());

    for (std::size_t b = 0; b < blockCosts.size(); ++b) {
        for (std::size_t i = blockRanges[b].first; i < blockRanges[b].second; ++i) {
            newLayers.push_back(std::move(layers_[i]));
        }
        const std::uint32_t pad = levelsComputation_ - blockCosts[b];
        if (pad > 0) {
            newLayers.push_back(std::make_unique<DummyMultLayer>(pad));
        }
        // If this block is followed by an activation, move it too.
        if (b + 1 < blockCosts.size()) {
            const std::size_t actIdx = blockRanges[b].second;
            newLayers.push_back(std::move(layers_[actIdx]));
        }
    }
    layers_ = std::move(newLayers);

    // ── 3. Identify trailing/initial dimensions.
    for (auto it = layers_.begin(); it != layers_.end(); ++it) {
        if (auto* lin = dynamic_cast<LinearLayer*>(it->get())) {
            initialInDim_ = lin->InDim();
            break;
        }
    }
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        if (auto* lin = dynamic_cast<LinearLayer*>(it->get())) {
            trailingOutDim_ = lin->OutDim();
            break;
        }
    }

    // ── 4. Collect activations for FBT depth sizing.
    //       The InputEncoder also uses an Identity activation; include it.
    std::vector<Activation> acts;
    acts.push_back(activations::Identity(ctx.params().pInput, ctx.params().pOutput,
                                          ctx.params().hermiteOrder));
    for (auto& up : layers_) {
        if (auto* a = dynamic_cast<ActivationLayer*>(up.get())) {
            acts.push_back(a->Spec());
        }
    }

    // ── 5. Build the FHE context.
    ctx.Build(acts, levelsComputation_);
    ctx_      = &ctx;
    compiled_ = true;

    BENCH_MEM("after-compile");
}

// ── Run ─────────────────────────────────────────────────────────────────────

Ciphertext<DCRTPoly> Network::EncodeInput(const std::vector<std::int64_t>& rawInput) {
    BENCH_LAYER_SCOPE("InputEncoder");

    if (!ctx_) throw std::runtime_error("Network::Run before Compile.");
    auto&         cc           = ctx_->cc();
    const auto&   params       = ctx_->params();
    const auto    numSlotsCKKS = ctx_->numSlotsCKKS();
    const auto    numSlots     = params.numSlots;
    const double  scaleTHI     = static_cast<double>(params.scaleTHI);

    // ── Pre-shift to [0, pInput) ─────────────────────────────────────────
    std::vector<std::int64_t> x_shifted(numSlots, 0);
    const std::size_t        limit = std::min<std::size_t>(rawInput.size(), numSlots);
    for (std::size_t i = 0; i < limit; ++i) {
        x_shifted[i] = rawInput[i] + inputShift_;
    }

    // ── Encrypt RLWE / mod-switch ─────────────────────────────────────────
    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(
        x_shifted, params.qBFVInit, params.pInput,
        ctx_->keyPair().secretKey, ctx_->elementParams(), ctx_->flagBR());
    SchemeletRLWEMP::ModSwitch(ctxtBFV, params.Q, params.qBFVInit);

    BENCH_MEM("after-encrypt");

    // ── RLWE -> CKKS ──────────────────────────────────────────────────────
    const std::uint32_t firstLvl =
        ctx_->totalDepth() - (params.levelsAvailableBeforeBootstrap > 0);
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctxtBFV, ctx_->keyPair().publicKey,
        params.BIGQ, numSlotsCKKS, firstLvl);

    // ── Identity EvalMVBNoDecoding (enter slot space) ─────────────────────
    auto identity = activations::Identity(
        params.pInput, params.pOutput, params.hermiteOrder);
    auto coeffIdentity = GetHermiteTrigCoefficients(
        identity.f, identity.pInput.ConvertToInt(), identity.order, params.scaleTHI);

    auto powers = cc->EvalMVBPrecompute(
        ctxt, coeffIdentity, params.pInput.GetMSB() - 1,
        ctx_->elementParams()->GetModulus(), identity.order);
    auto ctxtSlots = cc->EvalMVBNoDecoding(
        powers, coeffIdentity, params.pInput.GetMSB() - 1, identity.order);

    // ── Undo input shift in slot space ────────────────────────────────────
    if (inputShift_ != 0) {
        const std::uint32_t plaintextLvl = ctx_->totalDepth()
                                         - params.lvlb[1]
                                         - params.levelsAvailableAfterBootstrap
                                         - ctx_->levelsComputation();
        std::vector<double> shift_vec(numSlotsCKKS,
                                      -static_cast<double>(inputShift_) / scaleTHI);
        Plaintext pt_shift = cc->MakeCKKSPackedPlaintext(
            shift_vec, 1, plaintextLvl, nullptr, numSlotsCKKS);
        cc->EvalAddInPlace(ctxtSlots, pt_shift);
    }

    return ctxtSlots;
}

std::vector<std::int64_t> Network::DecodeOutput(
    const Ciphertext<DCRTPoly>& ct, int outLength) {
    BENCH_LAYER_SCOPE("OutputDecoder");

    auto&        cc       = ctx_->cc();
    const auto&  params   = ctx_->params();
    const double scaleTHI = static_cast<double>(params.scaleTHI);

    auto ctOut = cc->EvalHomDecoding(ct, scaleTHI, 0);
    auto polys = SchemeletRLWEMP::ConvertCKKSToRLWE(ctOut, params.Q);

    const int reportLen = NextPowerOfTwo(outLength);
    return SchemeletRLWEMP::DecryptCoeff(
        polys, params.Q, params.pOutput,
        ctx_->keyPair().secretKey, ctx_->elementParams(),
        reportLen, reportLen, ctx_->flagBR());
}

std::vector<std::int64_t> Network::Run(const std::vector<std::int64_t>& rawInput, Ciphertext<DCRTPoly> zeroes) {
    if (!compiled_ || !ctx_) {
        throw std::runtime_error("Network::Run called before Compile().");
    }

    BENCH_INFERENCE_SCOPE("Inference (Run)");

    ForwardState         state;

    auto& cc = ctx_->cc();
    const auto&   params       = ctx_->params();
    const std::uint32_t plaintextLvl = ctx_->totalDepth()
                                     - params.lvlb[1]
                                     - params.levelsAvailableAfterBootstrap
                                     - ctx_->levelsComputation();
    const auto    numSlotsCKKS = ctx_->numSlotsCKKS();
    Ciphertext<DCRTPoly> ct;
    if (!zeroes) {
        ct = EncodeInput(rawInput);
    } else {
        const double scaleTHI = static_cast<double>(params.scaleTHI);
        std::vector<double> rawInputDouble(numSlotsCKKS, 0.0);
        const std::size_t lim = std::min<std::size_t>(rawInput.size(), numSlotsCKKS);
        for (std::size_t i = 0; i < lim; ++i) {
            rawInputDouble[i] = static_cast<double>(rawInput[i]) / scaleTHI;
        }
        Plaintext pt_rawInput = cc->MakeCKKSPackedPlaintext(
            rawInputDouble, 1, plaintextLvl, nullptr, numSlotsCKKS);
        ct = cc->EvalAdd(zeroes, pt_rawInput);
    }
    BENCH_MEM("after-input-encode");

    // Per-layer instrumentation lives here (not inside each layer's Apply)
    // so the tag can include the layer's position. Linear layers get
    // "Layer 1", "Layer 2", … in the order they're applied; activation
    // layers get "Bootstrap + Activation" since the block IS the bootstrap.
    int linearN = 1;
    for (auto& layer : layers_) {
        const std::string& nm = layer->Name();
        std::string tag;
        if (nm.rfind("Activation", 0) == 0) {
            tag = "Bootstrap + Activation";
        } else if (nm == "Linear") {
            tag = "Layer " + std::to_string(linearN++);
        } else {
            tag = nm;
        }
        BENCH_LAYER_SCOPE_S(tag);
        ct = layer->Apply(*ctx_, state, ct);
    }
    BENCH_MEM("after-layers");

    const int outLen = trailingOutDim_ > 0 ? trailingOutDim_
                                            : static_cast<int>(rawInput.size());
    auto result = DecodeOutput(ct, outLen);
    BENCH_MEM("after-output-decode");

    if (static_cast<int>(result.size()) > outLen) result.resize(outLen);
    return result;
}

}  // namespace fhednn
