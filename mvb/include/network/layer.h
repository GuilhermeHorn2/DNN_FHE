#ifndef MVB_NETWORK_LAYER_H
#define MVB_NETWORK_LAYER_H

#include "network/activation.h"
#include "network/context.h"

#include "openfhe.h"

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace fhednn {

// Per-Run() mutable state. The Network builds one of these and threads it
// through every Layer::Apply() call. A layer may peek at it (e.g. to know how
// many bootstraps have happened so far) but must never mutate fields it does
// not own.
struct ForwardState {
    std::uint32_t activationIndex = 0;  // # of activations executed so far
    std::uint32_t linearIndex     = 0;  // # of linear layers executed so far
};

class Layer {
public:
    virtual ~Layer() = default;

    // # of rescale-consuming multiplications this layer performs.
    // Used by Network::Compile to size `levelsComputation`.
    virtual std::uint32_t LevelsConsumed() const = 0;

    virtual std::string Name() const = 0;

    virtual lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Apply(
        FHEContext&                                            ctx,
        ForwardState&                                          state,
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&        in) = 0;
};

// Fully-connected layer: out = W * in + b, computed in the CKKS slot space.
class LinearLayer : public Layer {
public:
    LinearLayer(std::vector<std::vector<double>> W,
                std::vector<double>              b,
                int                              inDim,
                int                              outDim);

    std::uint32_t LevelsConsumed() const override { return 2; }
    std::string   Name()           const override { return "Linear"; }

    int InDim()  const { return inDim_; }
    int OutDim() const { return outDim_; }

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Apply(
        FHEContext&                                            ctx,
        ForwardState&                                          state,
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&        in) override;

private:
    std::vector<std::vector<double>> W_;
    std::vector<double>              b_;
    int                              inDim_;
    int                              outDim_;
};

// Activation: shift + HomDecoding + CKKS<->RLWE refresh + EvalMVBNoDecoding.
// This block is the "bootstrap-equivalent" path in the pipeline.
class ActivationLayer : public Layer {
public:
    explicit ActivationLayer(Activation act);

    std::uint32_t LevelsConsumed() const override { return 0; }
    std::string   Name()           const override { return "Activation:" + act_.name; }

    const Activation& Spec() const { return act_; }

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Apply(
        FHEContext&                                            ctx,
        ForwardState&                                          state,
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&        in) override;

private:
    Activation                          act_;
    std::vector<std::complex<double>>   coeffs_;
    bool                                coeffsInitialized_ = false;

    void EnsureCoeffs(const FHEContext& ctx);
};

// Pads a slot-space block to the global `levelsComputation` budget by doing
// `count` fake multiplications by 1.0 + ModReduceInPlace. Inserted by
// Network::Compile when shorter blocks need to align with the deepest block.
class DummyMultLayer : public Layer {
public:
    explicit DummyMultLayer(std::uint32_t count) : count_(count) {}

    std::uint32_t LevelsConsumed() const override { return count_; }
    std::string   Name()           const override { return "DummyMult"; }

    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> Apply(
        FHEContext&                                            ctx,
        ForwardState&                                          state,
        const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&        in) override;

private:
    std::uint32_t count_;
};

}  // namespace fhednn

#endif  // MVB_NETWORK_LAYER_H
