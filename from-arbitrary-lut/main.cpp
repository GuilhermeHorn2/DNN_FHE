//==================================================================================
// BSD 2-Clause License
//
// Copyright (c) 2025, Duality Technologies Inc. and other contributors
//
// All rights reserved.
//
// Author TPOC: contact@openfhe.org
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//==================================================================================

/*
  Examples for functional bootstrapping for RLWE ciphertexts using CKKS.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "math/hermite.h"
#include "openfhe.h"
#include "schemelet/rlwe-mp.h"
#include <iostream>
#include <vector>
#include <random>

#include <functional>

using namespace lbcrypto;

const BigInteger QBFVINIT(BigInteger(1) << 60);
const BigInteger QBFVINITLARGE(BigInteger(1) << 80);
BigInteger PINPUT = BigInteger(1024);
BigInteger POUTPUT = BigInteger(1024);
BigInteger Q = BigInteger(1) << 47;
BigInteger BIGQ = BigInteger(1) << 47;
int64_t INPUT_DIM = 784;

std::vector<int64_t> load_image_bipolar(const char* path) {
    int w, h, ch;
    unsigned char* data = stbi_load(path, &w, &h, &ch, 1);
    if (!data) { std::cerr << "[ERROR] Cannot load: " << path << "\n"; return {}; }
    std::vector<int64_t> out(INPUT_DIM, -1);
    for (int i = 0; i < INPUT_DIM && i < w*h; ++i)
        out[i] = (data[i] > 127) ? 1 : -1;
    stbi_image_free(data);
    return out;
}

int nextPowerOfTwo(int n) {
    if (n <= 1) return 1;

    n--;                     // handle exact powers of two
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

void ArbitraryLUT(std::vector<int64_t> input, BigInteger QBFVInit, BigInteger PInput, BigInteger POutput, BigInteger Q, BigInteger Bigq,
                  uint64_t scaleTHI, size_t order, uint32_t numSlots, uint32_t ringDim,
                  std::function<int64_t(int64_t)> func);
int main(int argc, char* argv[]) {
    std::cerr << "\n*1.* Compute the function (x % PInput - POutput / 2) % POutput." << std::endl << std::endl;
    // LUT with 8-bit input and 4-bit output
    std::cerr << "=====8-to-4 bit LUT order 1 sparsely packed=====" << std::endl << std::endl;
    ArbitraryLUT(load_image_bipolar(argv[1]), QBFVINIT, PINPUT, POUTPUT, Q, BIGQ, 32, 1, 1024,
                 1 << 11, [](int64_t x) { return x; });
    return 0;
}

std::vector<std::vector<int64_t>> createRandomVectorOfVectors(
    int rows, int cols) {
    
    std::vector<std::vector<int64_t>> matrix(rows, std::vector<int64_t>(cols));
    std::random_device rd;
    std::mt19937_64 gen(rd()); // 64-bit generator
    std::uniform_int_distribution<int64_t> dis(-1, 1); // Range: -1, 0, 1

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = 1;
        }
    }
    return matrix;
}

Ciphertext<DCRTPoly> bridge_encrypt(
    const std::vector<int64_t>&          values,
    uint32_t                             enc_depth,
    uint32_t                             numSlotsCKKS,
    CryptoContext<DCRTPoly>&             cc,
    const KeyPair<DCRTPoly>&             kp,
    const std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>>&          ep)
{
    // Pad to numSlotsCKKS
    std::vector<int64_t> v = values;
    // v.resize(numSlotsCKKS, 0);

    auto ctBFV = SchemeletRLWEMP::EncryptCoeff(v, QBFVINIT, PINPUT, kp.secretKey, ep, true);
    SchemeletRLWEMP::ModSwitch(ctBFV, Q, QBFVINIT);
    return SchemeletRLWEMP::ConvertRLWEToCKKS(*cc, ctBFV, kp.publicKey, BIGQ, numSlotsCKKS, enc_depth);
}

Ciphertext<DCRTPoly> linear_layer_test(
    const Ciphertext<DCRTPoly>&              ct_input,
    const std::vector<std::vector<int64_t>>& W,
    const std::vector<int64_t>&              b,
    int                                      in_dim,
    int                                      out_dim,
    uint32_t                                 numSlotsCKKS,
    uint32_t                                 totalDepth,  // context total depth
    CryptoContext<DCRTPoly>&                 cc,
    const KeyPair<DCRTPoly>&                 kp,
    const std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>>&              ep)
{
    // Level accounting (FIXEDMANUAL: ConvertRLWEToCKKS(depth_arg) → level = totalDepth - depth_arg):
    //
    //   ct_input is at level L = ct_input->GetLevel().
    //   ct_w must also be at level L  → enc_depth_w    = totalDepth - L
    //   After EvalMult(ct_input, ct_w) + Rescale: level = L+1
    //   ct_mask must be at level L+1  → enc_depth_mask = totalDepth - L - 1
    //   After EvalMult(ct_sum, ct_mask) + Rescale: level = L+2
    //   Bias added as CKKS plaintext at level L+2 (no ct-ct multiply needed;
    //   bias is model-public data, plaintext is fine and saves one level).
    uint32_t L            = ct_input->GetLevel();
    // uint32_t enc_depth_w  = totalDepth - L;
    // uint32_t enc_depth_m  = (totalDepth > L + 1) ? totalDepth - L - 1 : 1;

    Ciphertext<DCRTPoly> ct_out;

    for (int j = 0; j < 1; ++j) {
        // ── Weight ciphertext at level L (matches ct_input) ──────────────────
        // std::vector<int64_t> w_vec(W[j].begin(), W[j].begin() + in_dim);
        // std::vector<int64_t> w_vec(W[j].begin(), W[j].begin() + 1024);
        auto w_vec = Fill<int64_t>(W[j], 1024);
        // std::cout << "W[j].begin " << w_vec << std::endl;
        // std::cout << "w_vec " << w_vec << std::endl;
        // auto ct_w   = bridge_encrypt(w_vec, enc_depth_w, numSlotsCKKS, cc, kp, ep);
        auto ct_w   = bridge_encrypt(w_vec, totalDepth, numSlotsCKKS, cc, kp, ep);
        // std::cout << "Scaling c_input: " << ct_input->GetScalingFactor() << std::endl;
        // Plaintext ct_w= cc->MakeCKKSPackedPlaintext(
        //         Fill<double>({1}, numSlotsCKKS), 1, ct_input->GetLevel(), nullptr, numSlotsCKKS);
        // std::cout << "Plaintext ones : " << ct_w << std::endl;
        // std::cout << "Scaling ct_w: " << ct_w->GetScalingFactor() << std::endl;
        // auto polys1 = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_w, Q);
        //
        // auto computed1 = SchemeletRLWEMP::DecryptCoeff(polys1, Q, POUTPUT, kp.secretKey, ep, numSlotsCKKS, numSlotsCKKS);
        // std::cout << "Weights:  " << computed1 << std::endl;

        // ── Dot product: EvalMult(ct_input, ct_w) → EvalSum → Rescale ────────
        // std::cout << "Before mult w: "
        //   << ct_w->GetLevel() << std::endl;
        std::cout << "Before mult input: "
          << ct_input->GetLevel() << std::endl;

        auto ct_dot = cc->EvalMult(ct_input, ct_w);
        // std::cout << "After mult dot: "
        //   << ct_dot->GetLevel() << std::endl;
        // ct_dot = cc->EvalSum(ct_dot, numSlotsCKKS);
        std::cout << "After Mult " << j << std::endl;
        cc->RescaleInPlace(ct_dot);       // level L → L+1
        auto polys1 = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_dot, Q);

        auto computed1 = SchemeletRLWEMP::DecryptCoeff(polys1, Q, POUTPUT, kp.secretKey, ep, numSlotsCKKS, numSlotsCKKS, true);
        std::cout << "After Rescale:  " << computed1 << std::endl;
        // cc->ModReduceInPlace(ct_dot);       // level L → L+1
        ct_out = ct_dot;

        // ── Mask ciphertext at level L+1 (matches ct_sum after Rescale) ──────
    //     std::vector<int64_t> mask_vec(numSlotsCKKS, 0);
    //     mask_vec[j] = 1;
    //     auto ct_mask = bridge_encrypt(mask_vec, totalDepth, numSlotsCKKS, cc, kp, ep);
    //
    //     // ── EvalMult(ct_sum, ct_mask) → Rescale ──────────────────────────────
    //     auto ct_n   = cc->EvalMult(ct_sum, ct_mask);
    //     cc->RescaleInPlace(ct_n);         // level L+1 → L+2
    //
    //     // ── Bias as plaintext (bias is public model data, no privacy concern) ─
    //     // MakeCKKSPackedPlaintext at the current level of ct_n (= L+2).
    //     std::vector<double> b_vec(numSlotsCKKS, 0.0);
    //     b_vec[j] = static_cast<double>(b[j]);
    //     ct_b = bridge_encrypt(b_vec, totalDepth, numSlotsCKKS, cc, kp, ep);
    //     // Plaintext pt_b = cc->MakeCKKSPackedPlaintext(b_vec, 1, ct_n->GetLevel());
    //     cc->EvalAddInPlace(ct_n, ct_b);
    //
    //     // ── Accumulate ────────────────────────────────────────────────────────
    //     if (j == 0) ct_out = ct_n;
    //     else        cc->EvalAddInPlace(ct_out, ct_n);
    }
    //
    // std::cout << "  level=" << ct_out->GetLevel()
    //           << "  scale=" << std::log2(ct_out->GetScalingFactor()) << " bits\n";
    return ct_out;
}


Ciphertext<DCRTPoly> linear_layer(
    const Ciphertext<DCRTPoly>&              ct_input,
    const std::vector<std::vector<int64_t>>& W,
    const std::vector<int64_t>&              b,
    int                                      in_dim,
    int                                      out_dim,
    uint32_t                                 numSlotsCKKS,
    uint32_t                                 totalDepth,  // context total depth
    CryptoContext<DCRTPoly>&                 cc,
    const KeyPair<DCRTPoly>&                 kp,
    const std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>>&              ep)
{
    // Level accounting (FIXEDMANUAL: ConvertRLWEToCKKS(depth_arg) → level = totalDepth - depth_arg):
    //
    //   ct_input is at level L = ct_input->GetLevel().
    //   ct_w must also be at level L  → enc_depth_w    = totalDepth - L
    //   After EvalMult(ct_input, ct_w) + Rescale: level = L+1
    //   ct_mask must be at level L+1  → enc_depth_mask = totalDepth - L - 1
    //   After EvalMult(ct_sum, ct_mask) + Rescale: level = L+2
    //   Bias added as CKKS plaintext at level L+2 (no ct-ct multiply needed;
    //   bias is model-public data, plaintext is fine and saves one level).
    // uint32_t L            = ct_input->GetLevel();
    // uint32_t enc_depth_w  = totalDepth - L;
    // uint32_t enc_depth_m  = (totalDepth > L + 1) ? totalDepth - L - 1 : 1;

    Ciphertext<DCRTPoly> ct_out;

    for (int j = 0; j < out_dim; ++j) {
        std::cout << "j " << j << std::endl;
        // ── Weight ciphertext at level L (matches ct_input) ──────────────────
        std::vector<int64_t> w_vec(W[j].begin(), W[j].begin() + nextPowerOfTwo(in_dim));
        // auto ct_w   = bridge_encrypt(w_vec, enc_depth_w, numSlotsCKKS, cc, kp, ep);
        auto ct_w   = bridge_encrypt(w_vec, totalDepth - 1, numSlotsCKKS, cc, kp, ep);

        // ── Dot product: EvalMult(ct_input, ct_w) → EvalSum → Rescale ────────
        auto ct_dot = cc->EvalMult(ct_input, ct_w);
        auto ct_sum = cc->EvalSum(ct_dot, numSlotsCKKS);
        cc->RescaleInPlace(ct_sum);       // level L → L+1
        std::cout << "After dot product " << std::endl;

        // ── Mask ciphertext at level L+1 (matches ct_sum after Rescale) ──────
        std::vector<int64_t> mask_vec(numSlotsCKKS, 0);
        mask_vec[j] = 1;
        auto ct_mask = bridge_encrypt(mask_vec, totalDepth - 1, numSlotsCKKS, cc, kp, ep);

        // ── EvalMult(ct_sum, ct_mask) → Rescale ──────────────────────────────
        auto ct_n   = cc->EvalMult(ct_sum, ct_mask);
        cc->RescaleInPlace(ct_n);         // level L+1 → L+2
        std::cout << "After mult" << std::endl;

        // ── Bias as plaintext (bias is public model data, no privacy concern) ─
        // MakeCKKSPackedPlaintext at the current level of ct_n (= L+2).
        // std::vector<double> b_vec(numSlotsCKKS, 0.0);
        // b_vec[j] = static_cast<double>(b[j]);
        auto ct_b = bridge_encrypt(b, totalDepth, numSlotsCKKS, cc, kp, ep);
        // Plaintext pt_b = cc->MakeCKKSPackedPlaintext(b_vec, 1, ct_n->GetLevel());
        cc->EvalAddInPlace(ct_n, ct_b);
        std::cout << "After add" << std::endl;

        // ── Accumulate ────────────────────────────────────────────────────────
        if (j == 0) ct_out = ct_n;
        else        cc->EvalAddInPlace(ct_out, ct_n);
    }

    std::cout << "  level=" << ct_out->GetLevel()
              << "  scale=" << std::log2(ct_out->GetScalingFactor()) << " bits\n";
    return ct_out;
}


void ArbitraryLUT(std::vector<int64_t> input, BigInteger QBFVInit, BigInteger PInput, BigInteger POutput, BigInteger Q, BigInteger Bigq,
                  uint64_t scaleTHI, size_t order, uint32_t numSlots, uint32_t ringDim,
                  std::function<int64_t(int64_t)> func) {
    /* 1. Figure out whether sparse packing or full packing should be used.
     * numSlots represents the number of values to be encrypted in BFV.
     * If this number is the same as the ring dimension, then the CKKS slots is half.
     */
    bool flagSP       = (numSlots <= ringDim / 2);  // sparse packing
    auto numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    /* 2. Input */
    std::vector<int64_t> x = input;
    x = Fill<int64_t>(x, numSlots);
    /* 3. The case of Boolean LUTs using the first order Trigonometric Hermite Interpolation
     * supports an optimized implementation. */
    std::vector<std::complex<double>> coeffcomp;

    coeffcomp = GetHermiteTrigCoefficients(func, PInput.ConvertToInt(), order, scaleTHI);  

    /* 4. Set up the cryptoparameters. */
    uint32_t dcrtBits                       = Bigq.GetMSB() - 1;
    uint32_t firstMod                       = Bigq.GetMSB() - 1;
    uint32_t levelsAvailableAfterBootstrap  = 0;
    uint32_t levelsAvailableBeforeBootstrap = 1;
    uint32_t dnum                           = 3;
    SecretKeyDist secretKeyDist             = SPARSE_TERNARY;
    std::vector<uint32_t> lvlb              = {3, 3};
    uint32_t levelsComputation = 2;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(dnum);
    parameters.SetBatchSize(numSlotsCKKS);
    parameters.SetRingDim(ringDim);

    uint32_t depth = levelsAvailableAfterBootstrap + levelsComputation;
    depth += FHECKKSRNS::GetFBTDepth(lvlb, coeffcomp, PInput, order, secretKeyDist);
    parameters.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension() << " and a multiplicative depth of "
              << depth << std::endl
              << std::endl;

    /* 5. Compute various moduli and scaling sizes, used for scheme conversions.
     * Then generate the setup parameters and necessary keys.
     */
    auto keyPair = cc->KeyGen();

    cc->EvalFBTSetup(coeffcomp, numSlotsCKKS, PInput, POutput, Bigq, keyPair.publicKey, {0, 0}, lvlb,
                         levelsAvailableAfterBootstrap, levelsComputation, order);

    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsCKKS);
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);


    /* 6. Perform encryption in the RLWE scheme, using a larger initial ciphertext modulus.
     * Switching the modulus to a smaller ciphertext modulus helps offset the encryption error.
     */
    auto ep = SchemeletRLWEMP::GetElementParams(keyPair.secretKey, depth - (levelsAvailableBeforeBootstrap > 0));
    std::cout << "before encrypt " << std::endl;

    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(x, QBFVInit, PInput, keyPair.secretKey, ep, true);
    std::cout << "after encrypt " << std::endl;

    SchemeletRLWEMP::ModSwitch(ctxtBFV, Q, QBFVInit);

    /* 7. Convert from the RLWE ciphertext to a CKKS ciphertext (both use the same secret key).
    */
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(*cc, ctxtBFV, keyPair.publicKey, Bigq, numSlotsCKKS,
                                                   depth - (levelsAvailableBeforeBootstrap > 0));
    std::cout << "ctxt Modulus: " << ctxt->GetElements()[0].GetModulus() << std::endl;
    std::cout << "ep Modulus: " << ep->GetModulus() << std::endl;
    // cc->ModReduceInPlace(ctxt);
    std::cout << "Num Elements: "
          << ctxt->GetElements()[0].GetNumOfElements() << std::endl;


    std::vector<int64_t> ones(numSlotsCKKS, 1);
    std::vector<double> multiplier(numSlotsCKKS, 2.0);
    Plaintext pt_ones = cc->MakeCKKSPackedPlaintext(multiplier, 1, depth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation, nullptr, numSlotsCKKS);
    std::cout << "Num Elements pt_ones: "
          << pt_ones->GetLevel()<< std::endl;
    Plaintext pt_ones1 = cc->MakeCKKSPackedPlaintext(multiplier, 1, depth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation, nullptr, numSlotsCKKS);
    // Plaintext ct_ones = cc->MakeCKKSPackedPlaintext(
    // Fill<double>({1}, numSlotsCKKS),
    // 1,                        // noiseScaleDeg (1 = not pre-scaled)
    // depth - lvlb[1] - levelsAvailableAfterBootstrap - 1,         // level — encodes at correct modulus
    // nullptr,                  // params (nullptr = use context)
    // numSlotsCKKS);
    // std::cout <bridge_encrypt(
    auto ct_ones = bridge_encrypt(
        ones, depth - (levelsAvailableBeforeBootstrap > 0), numSlotsCKKS, cc, keyPair, ep);
    // std::cout << "Num Elements ct_ones: "
    //       << ct_ones->GetElements()[0].GetNumOfElements() << std::endl;
    //
    // ctxt = cc->EvalAdd(ctxt, ct_ones);
    // cc->RescaleInPlace(ctxt);
    // ctxt = cc->EvalAdd(ctxt, ct_ones);
    // cc->RescaleInPlace(ctxt);
    // std::cout << "Level: " << ctxt->GetLevel() << std::endl;
    // std::cout << "Towers: "
    //       << ctxt->GetElements()[0].GetNumOfElements()
    //       << std::endl;
    // ctxt = cc->EvalHomDecoding(ctxt, scaleTHI, 2);
    // std::cout << "Scaling factor " << ct_ones->GetScalingFactor() << std::endl;

    // std::cout << "After Add" << std::endl;
    // cc->ModReduceInPlace(ctxt);
    // std::cout << "Aqui 2" << std::endl;
    // ctxt = cc->EvalMult(ctxt, pt_ones1);
    // cc->RescaleInPlace(ctxt);
    ctxt = cc->EvalMult(ctxt, pt_ones);
    cc->ModReduceInPlace(ctxt);
    // std::cout << "Num Elements: "
    //       << ctxt->GetElements()[0].GetNumOfElements() << std::endl;
    // ctxt = cc->EvalMult(ctxt, pt_ones);
    // cc->ModReduceInPlace(ctxt);
    // cc->ModReduceInPlace(ctxt);
    // std::cout << "Aqui 4" << std::endl;
    // ctxt = cc->EvalMult(ctxt, pt_ones);
    // std::cout << "Num Elements: "
    //       << ctxt->GetElements()[0].GetNumOfElements() << std::endl;
    // cc->RescaleInPlace(ctxt);
    // cc->ModReduceInPlace(ctxt);
    // std::cout << "Aqui 4" << std::endl;
    // ctxt = cc->EvalMult(ctxt, ct_ones);
    // cc->ModReduceInPlace(ctxt);
    // std::cout << "Aqui 4" << std::endl;
    auto polys2 = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxt, Q);

    auto computed2 = SchemeletRLWEMP::DecryptCoeff(polys2, Q, POutput, keyPair.secretKey, ep, numSlotsCKKS, numSlots, true);
    std::cout << "ct_ones:  " << computed2 << std::endl;

    auto W1 = createRandomVectorOfVectors(30, 784); 
    auto b1 = std::vector<int64_t>(Fill<int64_t>({1},numSlotsCKKS));
    std::cout << "After random vector" << std::endl;
    ctxt = linear_layer_test(ctxt, W1, b1, 784, 30, numSlotsCKKS, depth - (levelsAvailableBeforeBootstrap > 0), cc, keyPair, ep);
    std::cout << "Num Elements: "
          << ctxt->GetElements()[0].GetNumOfElements() << std::endl;
    auto polys1 = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxt, Q);

    auto computed1 = SchemeletRLWEMP::DecryptCoeff(polys1, Q, POutput, keyPair.secretKey, ep, numSlotsCKKS, numSlots, true);
    std::cout << "After Linear Layer: " << computed1 << std::endl;
    // std::cout << "Aqui 0" << std::endl;
    // cc->RescaleInPlace(afterLinearLayer1);
    // std::cout << "Aqui 5" << std::endl;

    /* 8. Apply the LUT over the ciphertext.
    */
    std::cout << "ctxt Modulus: " << ctxt->GetElements()[0].GetModulus() << std::endl;
    std::cout << "ep Modulus: " << ep->GetModulus() << std::endl;
    Ciphertext<DCRTPoly> ctxtAfterFBT;
    ctxtAfterFBT = cc->EvalFBT(ctxt, coeffcomp, PInput.GetMSB() - 1, ep->GetModulus(), scaleTHI, 2, order);
    std::cout << "Aqui 1" << std::endl;

    /* 9. Convert the result back to RLWE.
    */
    auto polys = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtAfterFBT, Q);

    auto computed = SchemeletRLWEMP::DecryptCoeff(polys, Q, POutput, keyPair.secretKey, ep, numSlotsCKKS, numSlots);
    computed.resize(784);
    std::transform(input.begin(), input.end(), Fill<int64_t>({1},784).begin(), input.begin(), std::plus<int>());
    bool is_equal = input == computed;
    std::cout << "Are equal? " << is_equal << std::endl;
    std::cout << "Final image: " << computed << std::endl;
}
