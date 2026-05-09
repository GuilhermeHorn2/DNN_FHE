#ifndef MVB_NETWORK_NETWORK_H
#define MVB_NETWORK_NETWORK_H

#include "network/activation.h"
#include "network/context.h"
#include "network/layer.h"

#include "openfhe.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace fhednn {

// Sequential FHE network builder.
//
// Usage:
//
//   FHEContext ctx;
//   Network    net;
//   net.SetInputShift(1)
//      .Linear(W1, b1)
//      .Activate(activations::Sign(256, ctx.params().pInput))
//      .Linear(W2, b2);
//   net.Compile(ctx);
//   auto scores = net.Run(load_image_bipolar("img.png"));
//
// Compile() walks the layer list, partitions it into "slot-space blocks"
// delimited by ActivationLayers, computes the per-block rescaling cost, takes
// the maximum to size the global `levelsComputation`, and inserts DummyMult
// layers in shorter blocks so that every block is exactly that deep. It then
// hands the deepest activations and the chosen levelsComputation to
// FHEContext::Build to size the CKKS context.
class Network {
public:
    Network() = default;

    // ── Builder API (chainable) ───────────────────────────────────────────

    // Pre-shift applied to the raw input before encryption. The InputEncoder
    // adds `inputShift` to every entry of the raw input, then subtracts the
    // same value (in slot space) right after the identity MVB. Default 0.
    Network& SetInputShift(std::int64_t k);

    // Add a fully-connected Linear layer.
    Network& Linear(std::vector<std::vector<double>> W,
                    std::vector<double>              b);

    // Add an activation. Multiple Activations -> multiple bootstrap blocks.
    Network& Activate(Activation act);

    // Append an arbitrary Layer. Useful for custom layers.
    Network& Add(std::unique_ptr<Layer> layer);

    // ── Compile / Run ─────────────────────────────────────────────────────

    // Computes levelsComputation, inserts DummyMult padding, and builds the
    // FHE context. Must be called exactly once before Run().
    void Compile(FHEContext& ctx);

    // End-to-end inference: encrypt -> RLWE->CKKS -> identity MVB -> shift
    // back -> [layers...] -> HomDecoding -> decrypt.
    // Returns the decrypted output vector (length = trailing Linear's outDim,
    // or rawInput.size() if no Linear is present).
    std::vector<std::int64_t> Run(const std::vector<std::int64_t>& rawInput);

    // ── Introspection ─────────────────────────────────────────────────────
    std::size_t NumLayers() const { return layers_.size(); }
    std::uint32_t LevelsComputation() const { return levelsComputation_; }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::int64_t                        inputShift_         = 0;
    std::uint32_t                       levelsComputation_  = 0;
    bool                                compiled_           = false;
    FHEContext*                         ctx_                = nullptr;

    // Detected at Compile time:
    int  trailingOutDim_ = 0;  // outDim of the last Linear layer (drives final decrypt length)
    int  initialInDim_   = 0;  // inDim of the first Linear layer (drives input size check)

    // Pipeline helpers (called by Run):
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncodeInput(
        const std::vector<std::int64_t>& rawInput);
    std::vector<std::int64_t> DecodeOutput(
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct, int outLength);
};

}  // namespace fhednn

#endif  // MVB_NETWORK_NETWORK_H
