#include "network/context.h"

#include "math/hermite.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace fhednn {

using namespace lbcrypto;

namespace {
    const char* SecurityLevelName(SecurityLevel s) {
        switch (s) {
            case HEStd_NotSet:        return "HEStd_NotSet (NO SECURITY GUARANTEE)";
            case HEStd_128_classic:   return "HEStd_128_classic";
            case HEStd_192_classic:   return "HEStd_192_classic";
            case HEStd_256_classic:   return "HEStd_256_classic";
            case HEStd_128_quantum:   return "HEStd_128_quantum";
            case HEStd_192_quantum:   return "HEStd_192_quantum";
            case HEStd_256_quantum:   return "HEStd_256_quantum";
            default:                  return "<unknown SecurityLevel>";
        }
    }

    const char* SecretKeyDistName(SecretKeyDist d) {
        switch (d) {
            case GAUSSIAN:             return "GAUSSIAN";
            case UNIFORM_TERNARY:      return "UNIFORM_TERNARY";
            case SPARSE_TERNARY:       return "SPARSE_TERNARY";
            default:                   return "<unknown SecretKeyDist>";
        }
    }

}

FHEContext::FHEContext(FHEParams params) : params_(std::move(params)) {}

void FHEContext::Build(const std::vector<Activation>& activations,
                       std::uint32_t                  levelsComputation) {
    if (activations.empty()) {
        throw std::runtime_error(
            "FHEContext::Build requires at least one activation to size the FBT depth.");
    }

    flagSP_       = (params_.numSlots <= params_.ringDim / 2);
    numSlotsCKKS_ = flagSP_ ? params_.numSlots : params_.numSlots / 2;

    // Pick the activation with the largest GetFBTDepth for FBT setup.
    const std::uint32_t dcrtBits = params_.BIGQ.GetMSB() - 1;
    const std::uint32_t firstMod = params_.BIGQ.GetMSB() - 1;

    std::vector<std::complex<double>> deepestCoeffs;
    std::uint32_t                     deepestFbtDepth = 0;
    std::string                       deepestName     = "<none>";

    for (const auto& act : activations) {
        auto coeffs = GetHermiteTrigCoefficients(
            act.f, act.pInput.ConvertToInt(), act.order, params_.scaleTHI);
        const std::uint32_t d = FHECKKSRNS::GetFBTDepth(
            params_.lvlb, coeffs, act.pInput, act.order, params_.secretKeyDist);
        if (d > deepestFbtDepth) {
            deepestFbtDepth = d;
            deepestCoeffs   = std::move(coeffs);
            deepestName     = act.name;
        }
    }

    fbtDepth_          = deepestFbtDepth;
    levelsComputation_ = levelsComputation;
    totalDepth_        = params_.levelsAvailableAfterBootstrap
                       + levelsComputation_
                       + fbtDepth_;

    std::cout << "[FHEContext] deepest activation = " << deepestName
              << "  fbtDepth = "         << fbtDepth_
              << "  levelsComputation = " << levelsComputation_
              << "  total depth = "      << totalDepth_ << "\n";

    CCParams<CryptoContextCKKSRNS> p;
    p.SetSecretKeyDist(params_.secretKeyDist);
    p.SetSecurityLevel(params_.securityLevel);
    p.SetScalingModSize(dcrtBits);
    p.SetScalingTechnique(FIXEDMANUAL);
    p.SetFirstModSize(firstMod);
    p.SetNumLargeDigits(params_.dnum);
    p.SetBatchSize(numSlotsCKKS_);

    if (params_.securityLevel == HEStd_NotSet || params_.ringDim != 0) {
        p.SetRingDim(params_.ringDim);
    }

    p.SetMultiplicativeDepth(totalDepth_);

    cc_ = GenCryptoContext(p);
    cc_->Enable(PKE);
    cc_->Enable(KEYSWITCH);
    cc_->Enable(LEVELEDSHE);
    cc_->Enable(ADVANCEDSHE);
    cc_->Enable(FHE);

    std::cout << "[FHEContext] ringDim = " << cc_->GetRingDimension()
              << "  numSlotsCKKS = " << numSlotsCKKS_
              << "  securityLevel = " << SecurityLevelName(params_.securityLevel)
              << "  secretKeyDist = " << SecretKeyDistName(params_.secretKeyDist)
              << "\n";

    keyPair_ = cc_->KeyGen();

    // EvalFBTSetup runs once with the deepest activation's coefficients;
    // the actual MVB calls in each ActivationLayer pass their own coefficients.
    cc_->EvalFBTSetup(deepestCoeffs,
                      numSlotsCKKS_,
                      params_.pInput,
                      params_.pOutput,
                      params_.BIGQ,
                      keyPair_.publicKey,
                      {0, 0},
                      params_.lvlb,
                      params_.levelsAvailableAfterBootstrap,
                      levelsComputation_,
                      params_.hermiteOrder);

    cc_->EvalBootstrapKeyGen(keyPair_.secretKey, numSlotsCKKS_);
    cc_->EvalMultKeyGen(keyPair_.secretKey);
    cc_->EvalSumKeyGen(keyPair_.secretKey);

    flagBR_ = (params_.lvlb[0] != 1 || params_.lvlb[1] != 1);

    ep_ = SchemeletRLWEMP::GetElementParams(
        keyPair_.secretKey,
        totalDepth_ - (params_.levelsAvailableBeforeBootstrap > 0));

    built_ = true;
}

}  // namespace fhednn
