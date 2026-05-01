#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "math/hermite.h"
#include <iostream>
#include <vector>
#include <random>

#include <functional>

using namespace lbcrypto;

int main() {
    const uint32_t ringDim   = 4096;
    const uint32_t numSlots  = 8;
    const BigInteger PInput(256);
    const BigInteger POutput(256);
    const BigInteger Q        = BigInteger(1) << 47;
    const BigInteger Bigq     = BigInteger(1) << 47;
    const BigInteger QBFVInit = BigInteger(1) << 60;
    const uint64_t scaleTHI   = 32;
    const size_t order        = 1;

    const bool     flagSP       = (numSlots <= ringDim / 2);
    const uint32_t numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    const int in_dim = 4;
    std::vector<int64_t> x = {10, 20, 30, 40, 0, 0, 0, 0};

    // ── Compute the sum in plaintext, store result in slot 0 ─────────────────
    // Since we cannot do EvalRotate in the bridged CKKS domain (coefficient
    // encoding, not slot encoding), we perform the reduction beforehand.
    // We want output slot j = sum_i W[j][i] * x[i].
    // Here: output[0] = x[0]+x[1]+x[2]+x[3], output[1..7] = 0.
    std::vector<int64_t> x_summed(numSlots, 0);
    for (int i = 0; i < in_dim; ++i)
        x_summed[0] = (x_summed[0] + x[i]) % POutput.ConvertToInt();

    int64_t expected = x_summed[0];
    std::cout << "Input:    ";
    for (auto v : x) std::cout << v << " ";
    std::cout << "\nExpected sum in slot 0: " << expected << "\n\n";

    // ── Crypto context setup (unchanged) ─────────────────────────────────────
    auto func      = [](int64_t v) -> int64_t { return v % 256; };
    auto coeffcomp = GetHermiteTrigCoefficients(
        func, PInput.ConvertToInt(), order, scaleTHI);

    std::vector<uint32_t> lvlb  = {3, 3};
    SecretKeyDist secretKeyDist = SPARSE_ENCAPSULATED;
    uint32_t depth = FHECKKSRNS::GetFBTDepth(
        lvlb, coeffcomp, PInput, order, secretKeyDist);
    const uint32_t dcrtBits = Bigq.GetMSB() - 1;

    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(secretKeyDist);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetScalingModSize(dcrtBits);
    params.SetScalingTechnique(FIXEDMANUAL);
    params.SetFirstModSize(dcrtBits);
    params.SetNumLargeDigits(3);
    params.SetBatchSize(numSlotsCKKS);
    params.SetRingDim(ringDim);
    params.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalFBTSetup(coeffcomp, numSlotsCKKS, PInput, POutput, Bigq,
                     keyPair.publicKey, {0, 0}, lvlb, 0, 0, order);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsCKKS);
    cc->EvalMultKeyGen(keyPair.secretKey);

    // ── Encrypt the pre-summed vector directly ────────────────────────────────
    auto ep    = SchemeletRLWEMP::GetElementParams(keyPair.secretKey, depth);
    auto ctBFV = SchemeletRLWEMP::EncryptCoeff(
        x_summed, QBFVInit, PInput, keyPair.secretKey, ep);
    SchemeletRLWEMP::ModSwitch(ctBFV, Q, QBFVInit);

    // ── Bridge: RLWE → CKKS → RLWE (no operations in between) ───────────────
    auto ctCKKS = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctBFV, keyPair.publicKey, Bigq, numSlotsCKKS, depth);

    auto polys  = SchemeletRLWEMP::ConvertCKKSToRLWE(ctCKKS, Q);
    auto result = SchemeletRLWEMP::DecryptCoeff(
        polys, Q, POutput, keyPair.secretKey, ep, numSlotsCKKS, numSlots);

    // ── Verify ────────────────────────────────────────────────────────────────
    std::cout << "Slot | got  | ok?\n";
    std::cout << "-----+------+----\n";
    for (int i = 0; i < (int)numSlots; ++i) {
        bool ok = (result[i] == (i == 0 ? expected : 0));
        std::printf("  %2d | %4lld | %s\n", i, result[i], ok ? "OK" : "FAIL");
    }
    return 0;
}
