//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2025, Example Implementation
//
// Homomorphic Neural Network for MNIST Digit Classification
// Uses OpenFHE CKKS with Functional Bootstrapping (FBT) for ReLU activation.
// Scheme conversion: BFV (coefficient domain) <-> CKKS (slot domain)
//
// Mirrors the ArbitraryLUT pattern from the OpenFHE functional bootstrapping
// examples exactly, to avoid precomputation slot-count mismatches.
//==================================================================================

/*
  Architecture:
    Input  : 784 pixels (28x28 MNIST image), quantized to 8-bit integers [0,255]
    Layer 1: Linear (784 -> NUM_HIDDEN), ReLU via FBT
    Layer 2: Linear (NUM_HIDDEN -> 10), argmax

  Key design choices (matching reference):
    - numSlots    = NUM_HIDDEN (number of BFV-packed values = number of hidden neurons)
    - flagSP      = (numSlots <= ringDim/2)  -> sparse packing when true
    - numSlotsCKKS = numSlots  when sparse,  numSlots/2 when full
    - EvalBootstrapKeyGen(sk, numSlotsCKKS)  <- same value as EvalFBTSetup
    - EvalFBT receives a ciphertext whose BatchSize == numSlotsCKKS

  Fix for "Precomputations for N slots not found":
    The error fires when EvalFBT internally looks up bootstrapping precomputations
    that were registered under a different slot count.  The rule is simple:
      SetBatchSize(numSlotsCKKS)
      EvalFBTSetup(..., numSlotsCKKS, ...)
      EvalBootstrapKeyGen(sk, numSlotsCKKS)   <- ALL THREE must be identical.

  ReLU over Z_256 (8-bit signed):
    relu(x) = x   if x in [0,127]
            = 0   if x in [128,255]  (these are "negative" values mod 256)
*/

#include "math/hermite.h"
#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace lbcrypto;

// ─────────────────────────────────────────────────────────────────────────────
// Global crypto constants  (tuned to match the 8-to-8 bit LUT in the reference)
// ─────────────────────────────────────────────────────────────────────────────
static const BigInteger QBFVINIT(BigInteger(1) << 60);  // large initial BFV modulus
static const BigInteger PINPUT(256);                     // 8-bit input  mod
static const BigInteger POUTPUT(256);                    // 8-bit output mod
static const BigInteger Q_RLWE((BigInteger(1) << 47));   // RLWE working modulus
static const BigInteger BIGQ((BigInteger(1) << 47));     // CKKS first-level modulus
static constexpr uint64_t SCALE_THI = 32;               // Hermite coefficient scale

// Network dimensions
static constexpr uint32_t INPUT_DIM  = 784;
// NUM_HIDDEN must satisfy: NUM_HIDDEN <= RING_DIM / 2  (sparse packing)
// Keep at 8 to exactly mirror the reference "8 slots, ringDim=4096" example.
// To use more neurons, increase RING_DIM accordingly (e.g. 128 neurons -> ringDim=4096 still fine).
static constexpr uint32_t NUM_HIDDEN = 8;
static constexpr uint32_t OUTPUT_DIM = 10;
static constexpr uint32_t RING_DIM   = 4096;

// ─────────────────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::vector<int64_t>> RandomWeights(uint32_t rows, uint32_t cols,
                                                 int64_t scale, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(-scale, scale);
    std::vector<std::vector<int64_t>> W(rows, std::vector<int64_t>(cols));
    for (auto& row : W)
        for (auto& v : row)
            v = dist(rng);
    return W;
}

std::vector<int64_t> RandomBias(uint32_t n, int64_t scale, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int64_t> dist(-scale, scale);
    std::vector<int64_t> b(n);
    for (auto& v : b) v = dist(rng);
    return b;
}

// ReLU in Z_256 (8-bit signed interpretation)
inline int64_t ReLU8(int64_t x) {
    int64_t sx = (x >= 128) ? (x - 256) : x;
    return (sx > 0) ? sx : 0;
}

// Plain forward pass for reference
std::vector<int64_t> PlainForward(
    const std::vector<int64_t>& input,
    const std::vector<std::vector<int64_t>>& W1,
    const std::vector<int64_t>& b1,
    const std::vector<std::vector<int64_t>>& W2,
    const std::vector<int64_t>& b2,
    int64_t wscale)
{
    // Layer 1
    std::vector<int64_t> h1(NUM_HIDDEN);
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j) {
        int64_t acc = b1[j];
        for (uint32_t i = 0; i < INPUT_DIM; ++i)
            acc += W1[j][i] * input[i];
        int64_t z = ((acc / wscale) % 256 + 256) % 256;
        h1[j] = ReLU8(z);
    }
    // Layer 2
    std::vector<int64_t> logits(OUTPUT_DIM);
    for (uint32_t k = 0; k < OUTPUT_DIM; ++k) {
        logits[k] = b2[k];
        for (uint32_t j = 0; j < NUM_HIDDEN; ++j)
            logits[k] += W2[k][j] * h1[j];
    }
    return logits;
}

// ─────────────────────────────────────────────────────────────────────────────
// HE ReLU layer  —  mirrors ArbitraryLUT() from the reference exactly
//
//  numSlots : number of BFV-encoded values (= NUM_HIDDEN here)
//  ringDim  : CKKS ring dimension
//  z        : pre-activation values in Z_256, length == numSlots
//  Returns  : h = ReLU(z), decrypted, length == numSlots
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int64_t> HEReLULayer(
    const std::vector<int64_t>& z,
    uint32_t numSlots,
    uint32_t ringDim)
{
    // ── 1. Sparse vs full packing (same logic as reference) ──────────────────
    // sparse packing: numSlots <= ringDim/2, numSlotsCKKS = numSlots
    // full   packing: numSlots >  ringDim/2, numSlotsCKKS = numSlots/2
    bool     flagSP       = (numSlots <= ringDim / 2);
    uint32_t numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    // ── 2. ReLU function over Z_256 ──────────────────────────────────────────
    auto funcReLU = [](int64_t x) -> int64_t {
        int64_t sx = (x >= 128) ? (x - 256) : x;
        return (sx > 0) ? sx : 0;
    };

    const size_t order = 1;  // first-order THI
    auto coeffReLU = GetHermiteTrigCoefficients(
        funcReLU, PINPUT.ConvertToInt(), order, SCALE_THI);

    // ── 3. Crypto parameters  (identical pattern to reference ArbitraryLUT) ──
    uint32_t dcrtBits = BIGQ.GetMSB() - 1;
    uint32_t firstMod = BIGQ.GetMSB() - 1;

    uint32_t levelsAvailableAfterBootstrap  = 0;
    uint32_t levelsAvailableBeforeBootstrap = 0;
    uint32_t dnum                           = 3;
    SecretKeyDist secretKeyDist             = SPARSE_ENCAPSULATED;
    std::vector<uint32_t> lvlb              = {3, 3};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(dnum);
    // *** Critical: SetBatchSize must equal numSlotsCKKS used in EvalFBTSetup
    //     and EvalBootstrapKeyGen — they all key the precomputation table by
    //     this value.  Mismatch -> "Precomputations for N slots not found". ***
    parameters.SetBatchSize(numSlotsCKKS);
    parameters.SetRingDim(ringDim);

    uint32_t depth = levelsAvailableAfterBootstrap
                   + FHECKKSRNS::GetFBTDepth(lvlb, coeffReLU, PINPUT, order, secretKeyDist);
    parameters.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    std::cout << "    CKKS ring dim = " << cc->GetRingDimension()
              << "  depth = " << depth
              << "  numSlotsCKKS = " << numSlotsCKKS
              << "  sparse = " << (flagSP ? "yes" : "no") << "\n";

    // ── 4. Key generation ────────────────────────────────────────────────────
    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    // *** EvalFBTSetup and EvalBootstrapKeyGen MUST both receive numSlotsCKKS ***
    cc->EvalFBTSetup(coeffReLU, numSlotsCKKS, PINPUT, POUTPUT, BIGQ,
                     keyPair.publicKey, {0, 0}, lvlb,
                     levelsAvailableAfterBootstrap, /*levelsComputation=*/0, order);

    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsCKKS);

    // ── 5. Element params for RLWE encryption ────────────────────────────────
    //   depth - (levelsAvailableBeforeBootstrap > 0)  mirrors the reference exactly
    auto ep = SchemeletRLWEMP::GetElementParams(
        keyPair.secretKey,
        depth - (levelsAvailableBeforeBootstrap > 0));

    // ── 6. Pad z to numSlots and encrypt as BFV coefficients ─────────────────
    std::vector<int64_t> zPadded = z;
    if (zPadded.size() < numSlots)
        zPadded = Fill<int64_t>(zPadded, numSlots);

    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(
        zPadded, QBFVINIT, PINPUT, keyPair.secretKey, ep);

    // Mod-switch from QBFVINIT down to Q_RLWE
    SchemeletRLWEMP::ModSwitch(ctxtBFV, Q_RLWE, QBFVINIT);

    // ── 7. BFV -> CKKS ───────────────────────────────────────────────────────
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctxtBFV, keyPair.publicKey, BIGQ, numSlotsCKKS,
        depth - (levelsAvailableBeforeBootstrap > 0));

    // ── 8. ReLU via EvalFBT ──────────────────────────────────────────────────
    auto ctxtAfterFBT = cc->EvalFBT(
        ctxt, coeffReLU,
        PINPUT.GetMSB() - 1,   // log2(256) = 8
        ep->GetModulus(),
        SCALE_THI,
        /*levelsToDrop=*/0,
        order);

    // ── 9. CKKS -> BFV coefficients ──────────────────────────────────────────
    auto polys = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtAfterFBT, Q_RLWE);

    // ── 10. Decrypt ──────────────────────────────────────────────────────────
    auto result = SchemeletRLWEMP::DecryptCoeff(
        polys, Q_RLWE, POUTPUT,
        keyPair.secretKey, ep,
        numSlotsCKKS, numSlots);

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=============================================================\n";
    std::cout << "  HE Neural Network - MNIST Digit Classification\n";
    std::cout << "  OpenFHE CKKS + Functional Bootstrapping  (BFV <-> CKKS)\n";
    std::cout << "=============================================================\n\n";

    // ── 0. Random weights ────────────────────────────────────────────────────
    const int64_t WSCALE = 16;
    auto W1 = RandomWeights(NUM_HIDDEN, INPUT_DIM,  1, 42);
    auto b1 = RandomBias(NUM_HIDDEN, 4, 43);
    auto W2 = RandomWeights(OUTPUT_DIM, NUM_HIDDEN, 4, 44);
    auto b2 = RandomBias(OUTPUT_DIM, 8, 45);

    // ── 1. Synthetic MNIST-like input ────────────────────────────────────────
    std::mt19937_64 rng(99);
    std::uniform_int_distribution<int64_t> pdist(0, 255);
    std::vector<int64_t> pixels(INPUT_DIM);
    for (auto& p : pixels) p = pdist(rng);

    std::cout << "[Input] First 8 pixel values: [";
    for (int i = 0; i < 8; ++i) std::cout << pixels[i] << (i < 7 ? ", " : "");
    std::cout << " ...]\n\n";

    // ── 2. Plaintext reference ───────────────────────────────────────────────
    auto plain_logits = PlainForward(pixels, W1, b1, W2, b2, WSCALE);
    int  plain_pred   = (int)(std::max_element(plain_logits.begin(), plain_logits.end())
                              - plain_logits.begin());

    std::cout << "[Plain] Logits: [";
    for (int k = 0; k < (int)OUTPUT_DIM; ++k)
        std::cout << plain_logits[k] << (k < (int)OUTPUT_DIM - 1 ? ", " : "");
    std::cout << "]\n";
    std::cout << "[Plain] Predicted digit: " << plain_pred << "\n\n";

    // ── 3. Layer 1 linear (plaintext) -> pre-activations z1 in Z_256 ─────────
    std::vector<int64_t> z1(NUM_HIDDEN);
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j) {
        int64_t acc = b1[j];
        for (uint32_t i = 0; i < INPUT_DIM; ++i)
            acc += W1[j][i] * pixels[i];
        z1[j] = ((acc / WSCALE) % 256 + 256) % 256;
    }

    std::cout << "[Layer 1] Pre-activations z1: [";
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j)
        std::cout << z1[j] << (j < NUM_HIDDEN - 1 ? ", " : "");
    std::cout << "]\n\n";

    // Plain ReLU reference
    std::vector<int64_t> h1_plain(NUM_HIDDEN);
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j)
        h1_plain[j] = ReLU8(z1[j]);

    std::cout << "[Layer 1] Plain ReLU h1:  [";
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j)
        std::cout << h1_plain[j] << (j < NUM_HIDDEN - 1 ? ", " : "");
    std::cout << "]\n\n";

    // ── 4. Homomorphic ReLU via FBT ──────────────────────────────────────────
    std::cout << "[Layer 1] Running HE ReLU (EvalFBT)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    auto h1_he = HEReLULayer(z1, NUM_HIDDEN, RING_DIM);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[Layer 1] HE ReLU h1:     [";
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j)
        std::cout << h1_he[j] << (j < NUM_HIDDEN - 1 ? ", " : "");
    std::cout << "]\n";

    int64_t max_err = 0;
    for (uint32_t j = 0; j < NUM_HIDDEN; ++j) {
        int64_t diff = std::abs(h1_he[j] - h1_plain[j]) % 256;
        max_err = std::max(max_err, diff);
    }
    std::cout << "[Layer 1] Max |HE - plain| error: " << max_err << "\n";
    std::cout << "[Layer 1] Time: " << ms << " ms\n\n";

    // ── 5. Layer 2 (plaintext, using HE activations) ─────────────────────────
    std::vector<int64_t> logits_he(OUTPUT_DIM, 0);
    for (uint32_t k = 0; k < OUTPUT_DIM; ++k) {
        logits_he[k] = b2[k];
        for (uint32_t j = 0; j < NUM_HIDDEN; ++j)
            logits_he[k] += W2[k][j] * h1_he[j];
    }
    int he_pred = (int)(std::max_element(logits_he.begin(), logits_he.end())
                        - logits_he.begin());

    std::cout << "[Layer 2] Logits (HE h1): [";
    for (int k = 0; k < (int)OUTPUT_DIM; ++k)
        std::cout << logits_he[k] << (k < (int)OUTPUT_DIM - 1 ? ", " : "");
    std::cout << "]\n\n";

    // ── 6. Summary ───────────────────────────────────────────────────────────
    std::cout << "=============================================================\n";
    std::cout << "  Summary\n";
    std::cout << "=============================================================\n";
    std::cout << "  Network:          " << INPUT_DIM << " -> " << NUM_HIDDEN
              << " -> " << OUTPUT_DIM << "\n";
    std::cout << "  Activation:       ReLU over Z_256, THI order 1\n";
    std::cout << "  Ring dim:         " << RING_DIM << "\n";
    std::cout << "  Slots (CKKS):     " << NUM_HIDDEN << " (sparse packing)\n";
    std::cout << "  Layer 1 HE time:  " << ms << " ms\n";
    std::cout << "  Max L1 error:     " << max_err << " (mod 256)\n";
    std::cout << "  Plain prediction: " << plain_pred << "\n";
    std::cout << "  HE prediction:    " << he_pred << "\n";
    std::cout << "  Match:            " << (he_pred == plain_pred ? "YES v" : "NO x") << "\n";
    std::cout << "=============================================================\n";

    return 0;
}
