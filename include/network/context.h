#ifndef MVB_NETWORK_CONTEXT_H
#define MVB_NETWORK_CONTEXT_H

#include "network/activation.h"

#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

#include <memory>
#include <vector>

namespace fhednn {

// All tunable crypto parameters in one place. Defaults match the original
// hard-coded values from the monolithic main.cpp.
struct FHEParams {
    lbcrypto::BigInteger qBFVInit = lbcrypto::BigInteger(1) << 60;
    lbcrypto::BigInteger pInput   = lbcrypto::BigInteger(1) << 10;
    lbcrypto::BigInteger pOutput  = lbcrypto::BigInteger(1) << 10;
    lbcrypto::BigInteger Q        = lbcrypto::BigInteger(1) << 47;
    lbcrypto::BigInteger BIGQ     = lbcrypto::BigInteger(1) << 47;

    std::uint32_t numSlots                       = 1024;
    std::uint32_t ringDim                        = 1u << 16;     // 2048
    std::uint64_t scaleTHI                       = 32;
    std::size_t   hermiteOrder                   = 1;
    std::uint32_t levelsAvailableAfterBootstrap  = 0;
    std::uint32_t levelsAvailableBeforeBootstrap = 0;
    std::uint32_t dnum                           = 3;

    std::vector<std::uint32_t>      lvlb           = {3, 3};
    lbcrypto::SecretKeyDist         secretKeyDist  = lbcrypto::SPARSE_TERNARY;
    lbcrypto::SecurityLevel         securityLevel  = lbcrypto::HEStd_NotSet;
};

// Owns the OpenFHE crypto context, key pair, and element parameters used by
// every layer. Built once via `Build()`, then passed to layers by reference.
class FHEContext {
public:
    explicit FHEContext(FHEParams params = {});

    // Builds the CCParams, generates keys, runs EvalFBTSetup using the deepest
    // activation (in terms of FBT depth) found in `activations`, and creates
    // the schemelet element parameters.
    //
    // `levelsComputation` is the global per-block depth budget (see
    // Network::Compile). It is the maximum number of rescaling operations any
    // slot-space block performs.
    void Build(const std::vector<Activation>& activations,
               std::uint32_t                  levelsComputation);

    // ── Accessors (only valid after Build) ────────────────────────────────
    const FHEParams&                      params() const { return params_; }
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc() { return cc_; }
    const lbcrypto::KeyPair<lbcrypto::DCRTPoly>& keyPair() const { return keyPair_; }
    const std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>>&
        elementParams() const { return ep_; }

    std::uint32_t numSlotsCKKS()           const { return numSlotsCKKS_; }
    std::uint32_t totalDepth()             const { return totalDepth_; }
    std::uint32_t fbtDepth()               const { return fbtDepth_; }
    std::uint32_t levelsComputation()      const { return levelsComputation_; }
    bool          flagBR()                 const { return flagBR_; }
    bool          flagSP()                 const { return flagSP_; }
    bool          isBuilt()                const { return built_; }

private:
    FHEParams                                       params_;
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>     cc_;
    lbcrypto::KeyPair<lbcrypto::DCRTPoly>           keyPair_;
    std::shared_ptr<lbcrypto::ILDCRTParams<lbcrypto::DCRTPoly::Integer>> ep_;
    std::uint32_t                                   numSlotsCKKS_     = 0;
    std::uint32_t                                   totalDepth_       = 0;
    std::uint32_t                                   fbtDepth_         = 0;
    std::uint32_t                                   levelsComputation_= 0;
    bool                                            flagBR_           = false;
    bool                                            flagSP_           = true;
    bool                                            built_            = false;
};

}  // namespace fhednn

#endif  // MVB_NETWORK_CONTEXT_H
