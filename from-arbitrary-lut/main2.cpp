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
    // ── Parameters ───────────────────────────────────────────────────────────
    const uint32_t ringDim  = 4096;
    const uint32_t numSlots = 8;
    const BigInteger PInput(256);
    const BigInteger Q        = BigInteger(1) << 47;
    const BigInteger Bigq     = BigInteger(1) << 47;
    const BigInteger QBFVInit = BigInteger(1) << 60;
    const uint64_t scaleTHI   = 32;

    const int64_t factorA = 3;
    const int64_t factorB = 5;

    std::vector<int64_t> x = {1, 2, 3, 4, 5, 6, 7, 8};

    // ── Derive depth exactly as the original code does ────────────────────────
    // We need a representative function just to call GetFBTDepth.
    // The depth it returns accounts for the encoding/decoding towers (lvlb)
    // that ConvertRLWEToCKKS consumes internally.
    std::vector<uint32_t> lvlb = {3, 3};
    SecretKeyDist secretKeyDist = SPARSE_ENCAPSULATED;

    auto func = [](int64_t v) -> int64_t { return v % 256; };
    auto coeffcomp = GetHermiteTrigCoefficients(
        func, PInput.ConvertToInt(), /*order=*/1, scaleTHI);

    // levelsAvailableAfterBootstrap=0, levelsComputation=0
    // but GetFBTDepth still returns the minimum bridge depth
    uint32_t depth = FHECKKSRNS::GetFBTDepth(
        lvlb, coeffcomp, PInput, /*order=*/1, secretKeyDist);

    // Add 2 more levels: one for each ct×pt EvalMult + ModReduce
    const uint32_t levelsForMult = 2;
    depth += levelsForMult;

    const uint32_t dcrtBits = Bigq.GetMSB() - 1;

    // ── CKKS context ──────────────────────────────────────────────────────────
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(secretKeyDist);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetScalingModSize(dcrtBits);
    params.SetScalingTechnique(FIXEDMANUAL);
    params.SetFirstModSize(dcrtBits);
    params.SetNumLargeDigits(3);
    params.SetBatchSize(numSlots);
    params.SetRingDim(ringDim);
    params.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);  // needed for ConvertRLWEToCKKS internal path

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);

    // EvalFBTSetup is required because ConvertRLWEToCKKS uses its key material
    cc->EvalFBTSetup(coeffcomp, numSlots, PInput, PInput, Bigq,
                     keyPair.publicKey, {0, 0}, lvlb,
                     /*levelsAfterBootstrap=*/levelsForMult,
                     /*levelsComputation=*/0, /*order=*/1);

    // ── 1. Encrypt in RLWE ───────────────────────────────────────────────────
    // level argument matches original: depth - (levelsAvailableBeforeBootstrap > 0)
    // levelsAvailableBeforeBootstrap=0, so just depth
    auto ep    = SchemeletRLWEMP::GetElementParams(keyPair.secretKey, depth);
    auto ctBFV = SchemeletRLWEMP::EncryptCoeff(x, QBFVInit, PInput, keyPair.secretKey, ep);
    SchemeletRLWEMP::ModSwitch(ctBFV, Q, QBFVInit);

    // ── 2. Bridge: RLWE → CKKS ───────────────────────────────────────────────
    auto ctCKKS = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctBFV, keyPair.publicKey, Bigq, numSlots, depth);

    // ── 3. Two ct×pt multiplications ─────────────────────────────────────────
    // After ConvertRLWEToCKKS the ciphertext is at the level it was handed in.
    // EvalMult(ct, pt) in FIXEDMANUAL raises the scale degree; ModReduce drops
    // one limb to normalise it — this is correct for ct×pt in FIXEDMANUAL.
    // We reserved levelsForMult=2 above exactly for these two reductions.
    auto ptA = cc->MakeCKKSPackedPlaintext(
        std::vector<double>(numSlots, static_cast<double>(factorA)),
        /*scaleDeg=*/1, /*level=*/0, nullptr, numSlots);
    auto ptB = cc->MakeCKKSPackedPlaintext(
        std::vector<double>(numSlots, static_cast<double>(factorB)),
        /*scaleDeg=*/1, /*level=*/0, nullptr, numSlots);

    auto ctA = cc->EvalMult(ctCKKS, ptA);
    cc->ModReduceInPlace(ctA);

    auto ctAB = cc->EvalMult(ctA, ptB);
    cc->ModReduceInPlace(ctAB);

    // ── 4. Bridge back: CKKS → RLWE, decrypt ─────────────────────────────────
    auto polys  = SchemeletRLWEMP::ConvertCKKSToRLWE(ctAB, Q);
    auto result = SchemeletRLWEMP::DecryptCoeff(
        polys, Q, PInput, keyPair.secretKey, ep, numSlots, numSlots);

    // ── 5. Verify ─────────────────────────────────────────────────────────────
    std::cout << "idx | input | expected | got | ok?\n";
    std::cout << "----+-------+----------+-----+----\n";
    int maxErr = 0;
    for (size_t i = 0; i < numSlots; ++i) {
        int64_t expected = (x[i] * factorA * factorB) % PInput.ConvertToInt();
        int64_t got      = result[i];
        int64_t err      = (int64_t)std::abs(expected - got) % PInput.ConvertToInt();
        maxErr = std::max(maxErr, (int)err);
        std::printf("  %zu |   %3lld |      %3lld | %3lld | %s\n",
                    i, x[i], expected, got, err == 0 ? "OK" : "FAIL");
    }
    std::printf("\nMax absolute error: %d\n", maxErr);
    return 0;
}
