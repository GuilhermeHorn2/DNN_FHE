//==================================================================================
// BSD 2-Clause License
// DiNN — Fully Private MNIST Inference
// OpenFHE CKKS + FBT Sign Activation (BFV <-> CKKS bridge throughout)
// Everything encrypted: image, weights, biases, masks — no plaintexts.
// Network: 784 -> [Linear] -> [Sign FBT] -> 30 -> [Linear] -> 10
//==================================================================================
/*
  KEY DESIGN (following ArbitraryLUT + compute_linear_layer from reference):

  All values — input, weights, biases, masks — are encrypted via the BFV bridge:
    1. EncryptCoeff(values, QBFVInit, PInput, sk, ep)
    2. ModSwitch(ctBFV, Q, QBFVInit)
    3. ct = ConvertRLWEToCKKS(*cc, ctBFV, pk, Bigq, numSlotsCKKS, depth - (levBefore > 0))

  Linear layer for neuron j (one level per EvalMult + Rescale, so 2 per neuron):
    ct_sum_j  = Rescale(EvalSum(EvalMult(ct_input, ct_weights_j), numSlotsCKKS))
    ct_neuron = Rescale(EvalMult(ct_sum_j, ct_mask_j)) + ct_bias_j
    ct_out   += ct_neuron

  Because weights are ciphertexts (not plaintexts), EvalMult is ct*ct which
  consumes 1 level + Rescale. Same for mask. Total: 2 levels per linear layer.

  Parameters matching the reference pattern:
    PInput = 8  (matches reference ArbitraryLUT call: PInput=8, POutput=8)
    levelsAvailableBeforeBootstrap = 10  (5 levels per linear layer * 2 layers
                                          — but reference uses 2 per layer = 4,
                                          we reserve 10 to be safe)
    sign function: f(x) = (x > 1) - (x < 1)  — sign over Z_8:
      positive [1,3] → +1, negative [5,7] → -1, zero/mid {0,4} → 0

  Sign function over Z_8 for pre-activations:
    z_j ∈ [-784, 784]. We need sign(z_j).
    z_j mod 8: positives land in [1,3], negatives land in [5,7].
    This only works if |z_j| < 4 — which is NOT true for our large pre-activations.

  CORRECT APPROACH: use PInput large enough to cover the range.
    z_j ∈ [-784, 784]. Use PInput = 2048 so the range fits.
    Negatives land in [1264, 2047], positives in [1, 784].
    sign(x) over Z_2048: (x < 1024) → +1, (x > 1024) → -1.

  But we must stay consistent with the ArbitraryLUT parameter pattern.
  The reference uses PInput=8 with:
    func = (x > 1) - (x < 1)
  This works because the REFERENCE INPUT x is already in Z_8.
  Our pre-activations are NOT in Z_8 — they're large integers.

  SOLUTION: clip/normalize pre-activations into Z_PInput before FBT.
  Since weights are {-1,+1} and inputs are {-1,+1}, pre-activations ∈ [-784,784].
  We can't reduce them mod PInput without decrypting (that would leak info).

  ACTUAL SOLUTION: use a large enough PInput.
  PInput = 2048 works. sign over Z_2048:
    f(x) = +1  if x ∈ [1, 1023]   (positive half)
    f(x) = -1  if x ∈ [1025, 2047] (negative half)
    f(x) =  0  if x ∈ {0, 1024}

  With levelsAvailableBeforeBootstrap = 2 (Layer 1 costs 2 levels),
  the image and weights are encrypted at depth - 1 (since lev > 0 → depth-1).
*/

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "math/hermite.h"
#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace lbcrypto;

// ─────────────────────────────────────────────────────────────────────────────
// Network / crypto constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int     INPUT_DIM  = 784;
static constexpr int     HIDDEN_DIM = 30;
static constexpr int     OUTPUT_DIM = 10;

// PInput = 2048 covers pre-activation range [-784,784].
// Q = Bigq = 2^47 (matching reference 8-to-4 bit LUT scaling).
// QBFVInit >> Q for encryption noise suppression.
// scaleTHI = 32 (matches reference).
// numSlots = 2048, ringDim = 2^13 = 8192
//   → flagSP = (2048 <= 4096) → sparse packing → numSlotsCKKS = 2048
// levelsAvailableBeforeBootstrap = 2 (Layer 1 costs 2 levels).
// levelsAvailableAfterBootstrap  = 2 (Layer 2 costs 2 levels).

static const BigInteger QBFVINIT(BigInteger(1) << 60);
static const BigInteger PINPUT(2048);
static const BigInteger POUTPUT(2048);
static const BigInteger Q(BigInteger(1) << 57);
static const BigInteger BIGQ(BigInteger(1) << 57);
static constexpr uint64_t SCALE_THI = 32;
static constexpr uint32_t NUM_SLOTS  = 2048;   // >= INPUT_DIM=784, pow2
static constexpr uint32_t RING_DIM   = 1 << 13; // 8192
static constexpr uint32_t LEV_BEFORE = 2;       // levels consumed by Layer 1
static constexpr uint32_t LEV_AFTER  = 2;       // levels available for Layer 2

// ─────────────────────────────────────────────────────────────────────────────
// DATA LOADING
// ─────────────────────────────────────────────────────────────────────────────
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

std::vector<int64_t> load_csv_1d(const std::string& path) {
    std::vector<int64_t> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) if (!line.empty()) out.push_back(std::stoll(line));
    return out;
}

std::vector<std::vector<int64_t>> load_csv_2d(const std::string& path, int ei, int eo) {
    std::vector<std::vector<int64_t>> raw;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        std::vector<int64_t> row;
        std::stringstream ss(line); std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) row.push_back(std::stoll(tok));
        if (!row.empty()) raw.push_back(row);
    }
    int nr=(int)raw.size(), nc=(int)raw[0].size();
    std::vector<std::vector<int64_t>> W(eo, std::vector<int64_t>(ei, 0));
    if      (nr==eo && nc==ei) W=raw;
    else if (nr==ei && nc==eo)
        for (int r=0;r<nr;++r) for (int c=0;c<nc;++c) W[c][r]=raw[r][c];
    else std::cerr<<"[ERROR] CSV shape mismatch\n";
    return W;
}

// ─────────────────────────────────────────────────────────────────────────────
// PLAINTEXT REFERENCE
// ─────────────────────────────────────────────────────────────────────────────
int plain_forward(const std::vector<int64_t>& px,
                  const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
                  const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2)
{
    std::vector<int64_t> z1(HIDDEN_DIM,0);
    for (int j=0;j<HIDDEN_DIM;++j) { z1[j]=b1[j]; for (int i=0;i<INPUT_DIM;++i) z1[j]+=W1[j][i]*px[i]; }
    std::vector<int64_t> h1(HIDDEN_DIM);
    for (int j=0;j<HIDDEN_DIM;++j) h1[j]=(z1[j]>0)?1:(z1[j]<0)?-1:0;
    std::vector<int64_t> logits(OUTPUT_DIM,0);
    for (int k=0;k<OUTPUT_DIM;++k) { logits[k]=b2[k]; for (int j=0;j<HIDDEN_DIM;++j) logits[k]+=W2[k][j]*h1[j]; }
    std::cout<<"[Plain] z1(10):  ["; for(int j=0;j<10;++j) std::cout<<z1[j]<<(j<9?", ":""); std::cout<<"]\n";
    std::cout<<"[Plain] h1:      ["; for(int j=0;j<HIDDEN_DIM;++j) std::cout<<h1[j]<<(j<HIDDEN_DIM-1?", ":""); std::cout<<"]\n";
    std::cout<<"[Plain] logits:  ["; for(int k=0;k<OUTPUT_DIM;++k) std::cout<<logits[k]<<(k<9?", ":""); std::cout<<"]\n";
    return (int)(std::max_element(logits.begin(),logits.end())-logits.begin());
}

// ─────────────────────────────────────────────────────────────────────────────
// BFV-BRIDGE HELPER
// Encrypt an integer vector via BFV->CKKS bridge at the given depth.
// ─────────────────────────────────────────────────────────────────────────────
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

    auto ctBFV = SchemeletRLWEMP::EncryptCoeff(v, QBFVINIT, PINPUT, kp.secretKey, ep);
    SchemeletRLWEMP::ModSwitch(ctBFV, Q, QBFVINIT);
    return SchemeletRLWEMP::ConvertRLWEToCKKS(*cc, ctBFV, kp.publicKey, BIGQ, numSlotsCKKS, enc_depth);
}

// ─────────────────────────────────────────────────────────────────────────────
// LINEAR LAYER — all ciphertext operations, weights/masks/biases via BFV bridge
//
// For each output neuron j (0..out_dim-1):
//   ct_w_j    = bridge_encrypt(W[j])       — weight ciphertext (same level as ct_input)
//   ct_mask_j = bridge_encrypt(mask_j)     — one-hot mask for slot j
//   ct_b_j    = bridge_encrypt(bias_j)     — bias in slot j
//
//   ct_dot_j  = EvalMult(ct_input, ct_w_j) → level+1, then Rescale → level stays
//   ct_sum_j  = EvalSum(ct_dot_j, N)        → all slots = dot product
//   ct_sum_j  = Rescale(ct_sum_j)           → level+1
//   ct_n_j    = EvalMult(ct_sum_j, ct_mask_j) → must match level of ct_sum_j
//   ct_n_j    = Rescale(ct_n_j)             → level+1 (= input_level + 2)
//   ct_n_j   += ct_b_j                      — bias add (no level change)
//   ct_out   += ct_n_j
//
// IMPORTANT: ct_w_j, ct_mask_j, ct_b_j are all encrypted at enc_depth so they
// represent the same modulus structure as ct_input. After two Rescales the
// output is at (input_level + 2).
// ─────────────────────────────────────────────────────────────────────────────
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
    uint32_t L            = ct_input->GetLevel();
    uint32_t enc_depth_w  = totalDepth - L;
    uint32_t enc_depth_m  = (totalDepth > L + 1) ? totalDepth - L - 1 : 1;

    std::cout << "enc_depth_w : " << enc_depth_w << std::endl;

    Ciphertext<DCRTPoly> ct_out;

    for (int j = 0; j < out_dim; ++j) {
        // ── Weight ciphertext at level L (matches ct_input) ──────────────────
        std::vector<int64_t> w_vec(W[j].begin(), W[j].begin() + in_dim);
        // auto ct_w   = bridge_encrypt(w_vec, enc_depth_w, numSlotsCKKS, cc, kp, ep);
        auto ct_w   = bridge_encrypt(w_vec, 0, numSlotsCKKS, cc, kp, ep);

        // ── Dot product: EvalMult(ct_input, ct_w) → EvalSum → Rescale ────────
        std::cout << "Before mult tower w: "
          << ct_w->GetElements()[0].GetNumOfElements() << std::endl;
        auto ct_dot = cc->EvalMult(ct_input, ct_w);
        auto ct_sum = cc->EvalSum(ct_dot, numSlotsCKKS);
        std::cout << "Level before: " << ct_sum->GetElements()[0].GetNumOfElements() << std::endl;
        cc->RescaleInPlace(ct_sum);       // level L → L+1
        std::cout << "Aqui " << std::endl;

        // ── Mask ciphertext at level L+1 (matches ct_sum after Rescale) ──────
        std::vector<int64_t> mask_vec(numSlotsCKKS, 0);
        mask_vec[j] = 1;
        auto ct_mask = bridge_encrypt(mask_vec, enc_depth_m, numSlotsCKKS, cc, kp, ep);

        // ── EvalMult(ct_sum, ct_mask) → Rescale ──────────────────────────────
        auto ct_n   = cc->EvalMult(ct_sum, ct_mask);
        cc->RescaleInPlace(ct_n);         // level L+1 → L+2

        // ── Bias as plaintext (bias is public model data, no privacy concern) ─
        // MakeCKKSPackedPlaintext at the current level of ct_n (= L+2).
        std::vector<double> b_vec(numSlotsCKKS, 0.0);
        b_vec[j] = static_cast<double>(b[j]);
        Plaintext pt_b = cc->MakeCKKSPackedPlaintext(b_vec, 1, ct_n->GetLevel());
        cc->EvalAddInPlace(ct_n, pt_b);

        // ── Accumulate ────────────────────────────────────────────────────────
        if (j == 0) ct_out = ct_n;
        else        cc->EvalAddInPlace(ct_out, ct_n);
    }

    std::cout << "  level=" << ct_out->GetLevel()
              << "  scale=" << std::log2(ct_out->GetScalingFactor()) << " bits\n";
    return ct_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: " << argv[0] << " <image>\n"; return 1; }

    std::cout << "=============================================================\n";
    std::cout << "  DiNN — Fully Private HE Inference  |  784->30->10\n";
    std::cout << "  All values encrypted (weights, biases, masks via BFV bridge)\n";
    std::cout << "=============================================================\n\n";

    // ── Load data ─────────────────────────────────────────────────────────────
    auto pixels = load_image_bipolar(argv[1]);
    if (pixels.empty()) return 1;
    auto W1 = load_csv_2d("../dinn30_W1.csv", INPUT_DIM,  HIDDEN_DIM);
    auto b1 = load_csv_1d("../dinn30_b1.csv");
    auto W2 = load_csv_2d("../dinn30_W2.csv", HIDDEN_DIM, OUTPUT_DIM);
    auto b2 = load_csv_1d("../dinn30_b2.csv");
    if (W1.empty()||b1.empty()||W2.empty()||b2.empty()) { std::cerr<<"[ERROR] weights\n"; return 1; }

    int pos=(int)std::count(pixels.begin(),pixels.end(),1L);
    std::cout<<"[Data] "<<pos<<" positive, "<<(784-pos)<<" negative pixels\n\n";

    // ── Plaintext reference ───────────────────────────────────────────────────
    std::cout << "--- Plaintext Reference ---\n";
    int plain_pred = plain_forward(pixels, W1, b1, W2, b2);
    std::cout << "[Plain] Predicted digit: " << plain_pred << "\n\n";

    // ── Crypto setup ──────────────────────────────────────────────────────────
    std::cout << "--- HE Setup ---\n";

    // Sign function over Z_PInput=Z_2048
    // Positive z_j ∈ [1,784]    → mod 2048 ∈ [1,784]     → sign = +1
    // Negative z_j ∈ [-784,-1]  → mod 2048 ∈ [1264,2047] → sign = -1
    // z_j = 0                   → 0                       → sign =  0
    // auto funcSign = [](int64_t x) -> int64_t {
    //     int64_t half = PINPUT.ConvertToInt<int64_t>() / 2; // 1024
    //     if (x == 0 || x == half) return 0;
    //     return (x < half) ? 1 : -1;
    // };
    auto funcSign = [](int64_t x) -> int64_t {
        // return (x > 1) - (x < 1);
        return x;
    };

    bool flagSP       = (NUM_SLOTS <= RING_DIM / 2);
    uint32_t numSlotsCKKS = flagSP ? NUM_SLOTS : NUM_SLOTS / 2;

    const size_t order = 1;
    auto coeffSign = GetHermiteTrigCoefficients(funcSign, PINPUT.ConvertToInt(), order, SCALE_THI);

    uint32_t dcrtBits   = BIGQ.GetMSB() - 1;
    SecretKeyDist skDist = SPARSE_ENCAPSULATED;
    std::vector<uint32_t> lvlb = {3, 3};

    uint32_t fbtDepth = FHECKKSRNS::GetFBTDepth(lvlb, coeffSign, PINPUT, order, skDist);
    // Total: LEV_BEFORE (Layer1) + fbtDepth (FBT) + LEV_AFTER (Layer2)
    uint32_t depth = 11 + LEV_BEFORE + fbtDepth + LEV_AFTER;

    std::cout << "[Setup] fbtDepth=" << fbtDepth << "  totalDepth=" << depth
              << "  numSlotsCKKS=" << numSlotsCKKS << "  sparse=" << flagSP << "\n";

    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(skDist);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetScalingModSize(dcrtBits);
    params.SetScalingTechnique(FIXEDMANUAL);
    params.SetFirstModSize(dcrtBits);
    params.SetNumLargeDigits(3);
    params.SetBatchSize(numSlotsCKKS);
    params.SetRingDim(RING_DIM);
    params.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE); cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE); cc->Enable(ADVANCEDSHE); cc->Enable(FHE);

    auto kp = cc->KeyGen();
    cc->EvalMultKeyGen(kp.secretKey);
    cc->EvalSumKeyGen(kp.secretKey);

    // EvalFBTSetup: levelsAvailableAfterBootstrap = LEV_AFTER (for Layer 2)
    cc->EvalFBTSetup(coeffSign, numSlotsCKKS, PINPUT, POUTPUT, BIGQ,
                     kp.publicKey, {0, 0}, lvlb, LEV_AFTER, 0, order);
    cc->EvalBootstrapKeyGen(kp.secretKey, numSlotsCKKS);

    // ep: element params at the level where the image ciphertext starts.
    // levelsAvailableBeforeBootstrap = LEV_BEFORE > 0 → use depth-1.
    // (Matching reference: ep = GetElementParams(sk, depth - (levBefore > 0)))
    auto ep = SchemeletRLWEMP::GetElementParams(kp.secretKey, depth - 1);

    std::cout << "[Setup] ep modulus bits=" << ep->GetModulus().GetMSB() << "\n";
    std::cout << "[Setup] Generating keys done.\n\n";

    // ── HE Inference ──────────────────────────────────────────────────────────
    std::cout << "--- HE Inference ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Step 1: Encrypt image via BFV bridge ──────────────────────────────────
    std::cout << "\n[Step 1] Encrypt image\n";
    // auto ct_px = bridge_encrypt(pixels, depth, numSlotsCKKS, cc, kp, ep);
    auto ct_px = bridge_encrypt(pixels, 38, numSlotsCKKS, cc, kp, ep);
    std::cout << "Before mult tower: "
          << ct_px->GetElements()[0].GetNumOfElements() << std::endl;
    std::cout << "[Step 1] level=" << ct_px->GetLevel()
              << "  scale=" << std::log2(ct_px->GetScalingFactor()) << " bits\n";

    // ── Step 2: Layer 1 linear (784->30), all ciphertext ops ─────────────────
    std::cout << "\n[Step 2] Layer 1 linear (" << INPUT_DIM << "->" << HIDDEN_DIM << ")\n";
    // auto ct_z1 = linear_layer(ct_px, W1, b1, INPUT_DIM, HIDDEN_DIM,
    //                            numSlotsCKKS, depth, cc, kp, ep);
    auto ct_z1 = linear_layer(ct_px, W1, b1, INPUT_DIM, HIDDEN_DIM,
                               numSlotsCKKS, 38, cc, kp, ep);
    {
        auto ph = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_z1, Q);
        auto hv = SchemeletRLWEMP::DecryptCoeff(ph, Q, POUTPUT, kp.secretKey, ep,
                                                 numSlotsCKKS, (uint32_t)HIDDEN_DIM);
        int64_t half = POUTPUT.ConvertToInt<int64_t>() / 2;
        std::cout << "[Debug] HE z1: [";
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            int64_t v = (hv[j] >= half) ? hv[j] - POUTPUT.ConvertToInt<int64_t>() : hv[j];
            int64_t s = (v > 0) ? 1 : (v < 0) ? -1 : 0;
            std::cout << s << (j<HIDDEN_DIM-1?", ":"");
        }
        std::cout << "]\n";
    }

    // std::vector<int64_t> ones = std::vector<int64_t>(numSlotsCKKS, 1);
    // auto ct_ones = bridge_encrypt(ones, 0, numSlotsCKKS, cc, kp, ep);
    // {
    //     auto ph = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_ones, Q);
    //     auto hv = SchemeletRLWEMP::DecryptCoeff(ph, Q, POUTPUT, kp.secretKey, ep,
    //                                              numSlotsCKKS, (uint32_t)HIDDEN_DIM);
    //     int64_t half = POUTPUT.ConvertToInt<int64_t>() / 2;
    //     std::cout << "[Debug] HE ones: [";
    //     for (int j = 0; j < HIDDEN_DIM; ++j) {
    //         int64_t v = (hv[j] >= half) ? hv[j] - POUTPUT.ConvertToInt<int64_t>() : hv[j];
    //         int64_t s = (v > 0) ? 1 : (v < 0) ? -1 : 0;
    //         std::cout << s << (j<HIDDEN_DIM-1?", ":"");
    //     }
    //     std::cout << "]\n";
    // }
    // ct_z1 = cc->EvalAdd(ct_z1, ct_ones);
    {
        auto ph = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_z1, Q);
        auto hv = SchemeletRLWEMP::DecryptCoeff(ph, Q, POUTPUT, kp.secretKey, ep,
                                                 numSlotsCKKS, (uint32_t)HIDDEN_DIM);
        int64_t half = POUTPUT.ConvertToInt<int64_t>() / 2;
        std::cout << "[Debug] HE z1: [";
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            int64_t v = (hv[j] >= half) ? hv[j] - POUTPUT.ConvertToInt<int64_t>() : hv[j];
            int64_t s = (v > 0) ? 1 : (v < 0) ? -1 : 0;
            std::cout << s << (j<HIDDEN_DIM-1?", ":"");
        }
        std::cout << "]\n";
    }

    // ── Step 3: Sign activation via FBT ───────────────────────────────────────
    std::cout << "\n[Step 3] Sign activation (FBT, Z_" << PINPUT << ")\n";
    std::cout << "[FBT] input level=" << ct_z1->GetLevel() << "\n";

    auto ct_h1 = cc->EvalFBT(ct_z1, coeffSign,
                              PINPUT.GetMSB() - 1,   // log2(2048)=11
                              ep->GetModulus(),
                              SCALE_THI, 0, order);

    std::cout << "[FBT] output level=" << ct_h1->GetLevel()
              << "  scale=" << std::log2(ct_h1->GetScalingFactor()) << " bits\n";

    // ep2: for Layer 2 operands — LEV_AFTER moduli remain after FBT
    auto ep2 = SchemeletRLWEMP::GetElementParams(kp.secretKey, LEV_AFTER);

    // Debug: decrypt ct_h1 to verify sign values
    {
        auto ph = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_h1, Q);
        auto hv = SchemeletRLWEMP::DecryptCoeff(ph, Q, POUTPUT, kp.secretKey, ep2,
                                                 numSlotsCKKS, (uint32_t)HIDDEN_DIM);
        int64_t half = POUTPUT.ConvertToInt<int64_t>() / 2;
        std::cout << "[Debug] HE h1: [";
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            int64_t v = (hv[j] >= half) ? hv[j] - POUTPUT.ConvertToInt<int64_t>() : hv[j];
            int64_t s = (v > 0) ? 1 : (v < 0) ? -1 : 0;
            std::cout << s << (j<HIDDEN_DIM-1?", ":"");
        }
        std::cout << "]\n";
    }

    // ── Step 4: Layer 2 linear (30->10), all ciphertext ops ──────────────────
    std::cout << "\n[Step 4] Layer 2 linear (" << HIDDEN_DIM << "->" << OUTPUT_DIM << ")\n";
    auto ct_logits = linear_layer(ct_h1, W2, b2, HIDDEN_DIM, OUTPUT_DIM,
                                   numSlotsCKKS, depth, cc, kp, ep2);

    // ── Step 5: Decrypt via BFV bridge ────────────────────────────────────────
    std::cout << "\n[Step 5] Decryption\n";

    // ep_out: 1 remaining modulus after Layer 2
    auto ep_out = SchemeletRLWEMP::GetElementParams(kp.secretKey, 1);

    auto polys  = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_logits, Q);
    auto scores = SchemeletRLWEMP::DecryptCoeff(polys, Q, POUTPUT, kp.secretKey, ep_out,
                                                 numSlotsCKKS, (uint32_t)OUTPUT_DIM);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

    int64_t phalf = POUTPUT.ConvertToInt<int64_t>() / 2;
    std::cout << "\n--- Final Class Scores (HE) ---\n";
    int he_pred = 0; int64_t max_s = LLONG_MIN;
    for (int k = 0; k < OUTPUT_DIM; ++k) {
        int64_t s = (scores[k] >= phalf) ? scores[k] - POUTPUT.ConvertToInt<int64_t>() : scores[k];
        std::cout << "  Digit " << k << ": " << s << "\n";
        if (s > max_s) { max_s = s; he_pred = k; }
    }

    std::cout << "\n=============================================================\n";
    std::cout << "  Total HE time:    " << ms << " ms\n";
    std::cout << "  Plain prediction: " << plain_pred << "\n";
    std::cout << "  HE prediction:    " << he_pred << "\n";
    std::cout << "  Match:            " << (he_pred == plain_pred ? "YES v" : "NO x") << "\n";
    std::cout << "=============================================================\n";
    return 0;
}
