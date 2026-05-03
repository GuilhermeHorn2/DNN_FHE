#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "math/hermite.h"
#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>

using namespace lbcrypto;

// ── Global crypto parameters (unchanged from original) ────────────────────────
const BigInteger QBFVINIT(BigInteger(1) << 60);
BigInteger PINPUT  = BigInteger(1) << 10;
BigInteger POUTPUT = BigInteger(1) << 10;
BigInteger Q       = BigInteger(1) << 47;
BigInteger BIGQ    = BigInteger(1) << 47;

// ── Network dimensions ────────────────────────────────────────────────────────
static constexpr int IN_DIM  = 784;
static constexpr int HID_DIM = 30;
static constexpr int OUT_DIM = 10;

// ── CSV / image loaders ───────────────────────────────────────────────────────
static std::vector<double> load_csv_1d(const std::string& path) {
    std::vector<double> v;
    std::ifstream f(path);
    if (!f) { std::cerr << "[ERROR] Cannot open " << path << "\n"; return v; }
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) v.push_back(std::stod(line));
    return v;
}

static std::vector<std::vector<double>> load_csv_2d(const std::string& path,
                                                     int expected_in,
                                                     int expected_out) {
    std::vector<std::vector<double>> raw;
    std::ifstream f(path);
    if (!f) { std::cerr << "[ERROR] Cannot open " << path << "\n"; return raw; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) row.push_back(std::stod(tok));
        if (!row.empty()) raw.push_back(row);
    }
    int rr = (int)raw.size(), rc = (int)raw[0].size();
    std::vector<std::vector<double>> W(expected_out, std::vector<double>(expected_in, 0.0));
    if (rr == expected_out && rc == expected_in) {
        W = raw;
    } else if (rr == expected_in && rc == expected_out) {
        for (int r = 0; r < rr; ++r)
            for (int c = 0; c < rc; ++c)
                W[c][r] = raw[r][c];
    } else {
        std::cerr << "[WARNING] " << path << " shape " << rr << "x" << rc
                  << " does not match expected " << expected_out << "x" << expected_in << "\n";
    }
    return W;
}

static std::vector<int64_t> load_image_bipolar(const char* path) {
    int w, h, ch;
    unsigned char* data = stbi_load(path, &w, &h, &ch, 1);
    if (!data) { std::cerr << "[ERROR] Cannot load: " << path << "\n"; return {}; }
    std::vector<int64_t> out(IN_DIM, -1);
    for (int i = 0; i < IN_DIM && i < w * h; ++i)
        out[i] = (data[i] > 127) ? 1 : -1;
    stbi_image_free(data);
    return out;
}

// ── nextPowerOfTwo (unchanged) ────────────────────────────────────────────────
static int nextPowerOfTwo(int n) {
    if (n <= 1) return 1;
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
}

// ── linear_layer (unchanged from original) ────────────────────────────────────
static Ciphertext<DCRTPoly> linear_layer(
    const Ciphertext<DCRTPoly>&                              ct_input,
    const std::vector<std::vector<double>>&                  W,
    const std::vector<double>&                               b,
    int                                                      in_dim,
    int                                                      out_dim,
    uint32_t                                                 numSlotsCKKS,
    uint32_t                                                 totalDepth,
    CryptoContext<DCRTPoly>&                                 cc,
    const KeyPair<DCRTPoly>&                                 kp,
    const std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>>& ep,
    std::vector<uint32_t>                                    lvlb,
    uint32_t                                                 levelsAvailableAfterBootstrap,
    uint32_t                                                 levelsComputation,
    uint32_t                                                 scaleTHI,
    uint32_t                                                 fbtDepth)
{
    Ciphertext<DCRTPoly> ct_out;

    for (int j = 0; j < out_dim; ++j) {
        std::cout << "  neuron " << j + 1 << "/" << out_dim << "\r" << std::flush;

        std::vector<double> w_vec(W[j].begin(), W[j].end());
        w_vec.resize(nextPowerOfTwo(in_dim));
        Plaintext pt_weights = cc->MakeCKKSPackedPlaintext(
            w_vec, 1,
            totalDepth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation,
            nullptr, numSlotsCKKS);

        auto ct_dot = cc->EvalMult(ct_input, pt_weights);
        auto ct_sum = cc->EvalSum(ct_dot, nextPowerOfTwo(in_dim));
        cc->ModReduceInPlace(ct_sum);

        std::vector<double> mask_vec(nextPowerOfTwo(in_dim), 0);
        mask_vec[j] = 1.0;
        Plaintext pt_mask = cc->MakeCKKSPackedPlaintext(
            mask_vec, 1,
            totalDepth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation,
            nullptr, numSlotsCKKS);

        auto ct_n = cc->EvalMult(ct_sum, pt_mask);
        cc->ModReduceInPlace(ct_n);

        std::vector<double> b_vec(nextPowerOfTwo(in_dim), 0);
        b_vec[j] = b[j]/scaleTHI;

        Plaintext pt_bias = cc->MakeCKKSPackedPlaintext(
            b_vec, 1,
            totalDepth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation,
            nullptr, numSlotsCKKS);
        cc->EvalAddInPlace(ct_n, pt_bias);

        if (j == 0) ct_out = ct_n;
        else        cc->EvalAddInPlace(ct_out, ct_n);
    }
    std::cout << "\n";
    return ct_out;
}

// ── forward declaration ───────────────────────────────────────────────────────
void RunDiNN(std::vector<int64_t>             input,
             std::vector<std::vector<double>> W1,
             std::vector<double>              b1,
             std::vector<std::vector<double>> W2,
             std::vector<double>              b2);

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png>\n"
                  << "       Weights: ../dinn30_W1.csv  ../dinn30_b1.csv\n"
                  << "                ../dinn30_W2.csv  ../dinn30_b2.csv\n";
        return 1;
    }

    std::cout << "Loading weights...\n";
    auto W1 = load_csv_2d("../dinn30_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = load_csv_1d("../dinn30_b1.csv");
    auto W2 = load_csv_2d("../dinn30_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = load_csv_1d("../dinn30_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "Failed to load weights.\n"; return 1;
    }

    std::cout << "Loading image: " << argv[1] << "\n";
    auto pixels = load_image_bipolar(argv[1]);
    if (pixels.empty()) return 1;

    RunDiNN(pixels, W1, b1, W2, b2);
    return 0;
}

// ── RunDiNN: full pipeline ────────────────────────────────────────────────────
void RunDiNN(std::vector<int64_t>             input,
             std::vector<std::vector<double>> W1,
             std::vector<double>              b1,
             std::vector<std::vector<double>> W2,
             std::vector<double>              b2) {

    auto a     = PINPUT.ConvertToInt<int64_t>();
    auto b     = POUTPUT.ConvertToInt<int64_t>();

    const uint32_t numSlots = 1024;
    const uint32_t ringDim  = 1 << 11;   // 2048
    const uint64_t scaleTHI = 32;
    const size_t   order    = 1;

    bool     flagSP       = (numSlots <= ringDim / 2);
    uint32_t numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    // ── Shift input into [0, PINPUT) ─────────────────────────────────────────
    // Pixels are {-1, +1}. FBT requires inputs in [0, PINPUT).
    // We add PINPUT/2 = 512, mapping:
    //   -1 → 511   (in [0, PINPUT/2)   → sign LUT outputs -1)
    //   +1 → 513   (in [PINPUT/2, PINPUT) → sign LUT outputs +1)
    // int64_t half = PINPUT.ConvertToInt<int64_t>() / 2;
    std::vector<int64_t> x_shifted(numSlots, 0);
    for (int i = 0; i < IN_DIM && i < (int)numSlots; ++i)
        x_shifted[i] = input[i] + 1;
    std::cout << "x_shifted: " << x_shifted << std::endl;

    // ── Identity LUT coefficients ─────────────────────────────────────────────
    // Used for the first EvalMVBNoDecoding to enter CKKS slot space.
    // f(x) = x over [0, PINPUT).
    std::function<int64_t(int64_t)> f_identity = [a, b](int64_t x) { return (x % a) % b; };
    auto coeffIdentity = GetHermiteTrigCoefficients(
        f_identity, PINPUT.ConvertToInt(), order, scaleTHI);

    // ── Sign LUT coefficients ─────────────────────────────────────────────────
    // Input is in [0, PINPUT). Compare against PINPUT/2 (the shifted zero):
    //   x ≥ PINPUT/2 → original pre-activation was ≥ 0 → return +1
    //   x <  PINPUT/2 → original pre-activation was  < 0 → return -1
    // FBT can return negative values; it just cannot receive them.
    int64_t pinput_half = PINPUT.ConvertToInt<int64_t>() / 2;
    std::function<int64_t(int64_t)> f_sign = [pinput_half](int64_t x) -> int64_t {
        return (x >= 256) ? 1 : -1;
    };
    auto coeffSign = GetHermiteTrigCoefficients(
        f_sign, PINPUT.ConvertToInt(), order, scaleTHI);

    // ── Crypto context (unchanged from original) ──────────────────────────────
    uint32_t dcrtBits                       = BIGQ.GetMSB() - 1;
    uint32_t firstMod                       = BIGQ.GetMSB() - 1;
    uint32_t levelsAvailableAfterBootstrap  = 0;
    uint32_t levelsAvailableBeforeBootstrap = 0;
    uint32_t dnum                           = 3;
    SecretKeyDist secretKeyDist             = SPARSE_TERNARY;
    std::vector<uint32_t> lvlb              = {3, 3};

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(dnum);
    parameters.SetBatchSize(numSlotsCKKS);
    parameters.SetRingDim(ringDim);

    // Use sign coefficients for depth (deeper than identity)
    uint32_t fbtDepth          = FHECKKSRNS::GetFBTDepth(lvlb, coeffSign, PINPUT, order, secretKeyDist);
    uint32_t levelsComputation = 2;
    uint32_t depth             = levelsAvailableAfterBootstrap + levelsComputation + fbtDepth;
    parameters.SetMultiplicativeDepth(depth);

    std::cout << "fbtDepth=" << fbtDepth
              << "  levelsComputation=" << levelsComputation
              << "  total depth=" << depth << "\n";

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    std::cout << "Ring dim=" << cc->GetRingDimension()
              << "  numSlotsCKKS=" << numSlotsCKKS << "\n";

    auto keyPair = cc->KeyGen();

    // EvalFBTSetup uses sign coefficients (the actual activation used in FBT)
    cc->EvalFBTSetup(coeffSign, numSlotsCKKS, PINPUT, POUTPUT, BIGQ,
                     keyPair.publicKey, {0, 0}, lvlb,
                     levelsAvailableAfterBootstrap, levelsComputation, order);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsCKKS);
    cc->EvalMultKeyGen(keyPair.secretKey);
    cc->EvalSumKeyGen(keyPair.secretKey);

    bool flagBR = (lvlb[0] != 1 || lvlb[1] != 1);

    auto ep = SchemeletRLWEMP::GetElementParams(
        keyPair.secretKey, depth - (levelsAvailableBeforeBootstrap > 0));

    // ── Encrypt shifted input ─────────────────────────────────────────────────
    std::cout << "\nEncrypting input...\n";
    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(
        x_shifted, QBFVINIT, PINPUT, keyPair.secretKey, ep, flagBR);
    SchemeletRLWEMP::ModSwitch(ctxtBFV, Q, QBFVINIT);

    // ── RLWE → CKKS ───────────────────────────────────────────────────────────
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctxtBFV, keyPair.publicKey, BIGQ, numSlotsCKKS,
        depth - (levelsAvailableBeforeBootstrap > 0));

    // ── Enter slot space (identity EvalMVBNoDecoding) ─────────────────────────
    std::cout << "EvalMVBNoDecoding (identity — entering slot space)...\n";
    auto complexExpPowers = cc->EvalMVBPrecompute(
        ctxt, coeffIdentity, PINPUT.GetMSB() - 1, ep->GetModulus(), order);
    auto ctxtSlots = cc->EvalMVBNoDecoding(
        complexExpPowers, coeffIdentity, PINPUT.GetMSB() - 1, order);

    Plaintext pt_ones = cc->MakeCKKSPackedPlaintext(
        Fill<double>({1}, numSlotsCKKS), 1,
        depth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation, nullptr, numSlotsCKKS);

    Plaintext pt_shift = cc->MakeCKKSPackedPlaintext(
        std::vector<double>(numSlotsCKKS, -1.0/scaleTHI), 1,
        depth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation,
        nullptr, numSlotsCKKS);
    cc->EvalAddInPlace(ctxtSlots, pt_shift);

    // ctxtSlots = cc->EvalMult(ctxtSlots, pt_ones);
    // cc->ModReduceInPlace(ctxtSlots);
    // ctxtSlots = cc->EvalMult(ctxtSlots, pt_ones);
    // cc->ModReduceInPlace(ctxtSlots);
    // ctxtSlots = cc->EvalHomDecoding(ctxtSlots, scaleTHI, 0);
    //
    // auto polys1 = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtSlots, Q);
    // auto debug1 = SchemeletRLWEMP::DecryptCoeff(
    //     polys1, Q, POUTPUT, keyPair.secretKey, ep, numSlotsCKKS, numSlotsCKKS, flagBR);
    // std::cout << "after shift " << debug1 << std::endl;

    // ── Layer 1 (in slot space) ───────────────────────────────────────────────
    std::cout << "Layer 1 (" << IN_DIM << " → " << HID_DIM << ")...\n";
    auto ctxtHidPre = linear_layer(
        ctxtSlots, W1, b1,
        IN_DIM, HID_DIM,
        numSlotsCKKS, depth, cc, keyPair, ep,
        lvlb, levelsAvailableAfterBootstrap, levelsComputation,
        scaleTHI, fbtDepth);

    auto vec_ones = std::vector<double>(HID_DIM, 256.0/scaleTHI);
    vec_ones.resize(numSlotsCKKS);
    pt_shift = cc->MakeCKKSPackedPlaintext(
        vec_ones, 1,
        depth - lvlb[1] - levelsAvailableAfterBootstrap - levelsComputation,
        nullptr, numSlotsCKKS);
    cc->EvalAddInPlace(ctxtHidPre, pt_shift);

    // ── Exit slot space, CKKS→RLWE→CKKS refresh, then sign EvalMVBNoDecoding ─
    // This refresh re-encodes the Layer 1 outputs in [0, PINPUT) so that the
    // sign LUT receives valid inputs.
    std::cout << "EvalHomDecoding + CKKS↔RLWE refresh + sign EvalMVBNoDecoding...\n";
    ctxtHidPre = cc->EvalHomDecoding(ctxtHidPre, scaleTHI, 0);
    auto polys = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtHidPre, Q);
    ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, polys, keyPair.publicKey, BIGQ, numSlotsCKKS,
        depth - (levelsAvailableBeforeBootstrap > 0));
    auto debug = SchemeletRLWEMP::DecryptCoeff(
        polys, Q, POUTPUT, keyPair.secretKey, ep, numSlotsCKKS, numSlotsCKKS, flagBR);
    std::cout << "after linear layer " << debug << std::endl;


    complexExpPowers = cc->EvalMVBPrecompute(
        ctxt, coeffSign, PINPUT.GetMSB() - 1, ep->GetModulus(), order);
    auto ctxtHidPost = cc->EvalMVBNoDecoding(
        complexExpPowers, coeffSign, PINPUT.GetMSB() - 1, order);

    // ── Layer 2 (in slot space) ───────────────────────────────────────────────
    std::cout << "Layer 2 (" << HID_DIM << " → " << OUT_DIM << ")...\n";
    auto ctxtScores = linear_layer(
        ctxtHidPost, W2, b2,
        HID_DIM, OUT_DIM,
        numSlotsCKKS, depth, cc, keyPair, ep,
        lvlb, levelsAvailableAfterBootstrap, levelsComputation,
        scaleTHI, fbtDepth);

    // ── Exit slot space and decrypt ───────────────────────────────────────────
    std::cout << "EvalHomDecoding + decrypt...\n";
    ctxtScores = cc->EvalHomDecoding(ctxtScores, scaleTHI, 0);
    polys      = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtScores, Q);
    auto computed = SchemeletRLWEMP::DecryptCoeff(
        polys, Q, POUTPUT, keyPair.secretKey, ep, nextPowerOfTwo(OUT_DIM), nextPowerOfTwo(OUT_DIM), flagBR);

    // ── Scores and prediction ─────────────────────────────────────────────────
    std::cout << "\n--- Final Class Scores ---\n";
    int     predicted = 0;
    int64_t maxScore  = LLONG_MIN;
    for (int j = 0; j < OUT_DIM; ++j) {
        std::printf("  Digit %d: %lld\n", j, computed[j]);
        if (computed[j] > maxScore) { maxScore = computed[j]; predicted = j; }
    }

    // ── Plaintext reference ───────────────────────────────────────────────────
    std::vector<double> ref_hidden(HID_DIM, 0.0);
    for (int j = 0; j < HID_DIM; ++j) {
        double acc = b1[j];
        for (int i = 0; i < IN_DIM; ++i)
            acc += W1[j][i] * (double)input[i];
        ref_hidden[j] = (acc >= 0.0) ? 1.0 : -1.0;
    }
    std::vector<double> ref_scores(OUT_DIM, 0.0);
    for (int j = 0; j < OUT_DIM; ++j) {
        ref_scores[j] = b2[j];
        for (int i = 0; i < HID_DIM; ++i)
            ref_scores[j] += W2[j][i] * ref_hidden[i];
    }
    int ref_pred = (int)(std::max_element(ref_scores.begin(), ref_scores.end())
                         - ref_scores.begin());

    std::cout << "\n================================\n";
    std::printf(" PREDICTED DIGIT  : %d\n", predicted);
    std::printf(" REFERENCE (plain): %d  %s\n", ref_pred,
                predicted == ref_pred ? "(matches)" : "(MISMATCH)");
    std::cout << "================================\n";
}
