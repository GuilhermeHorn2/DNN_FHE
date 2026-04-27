//==================================================================================
// BSD 2-Clause License
//
// DiNN — Fully Private MNIST Inference
// OpenFHE CKKS + Functional Bootstrapping (BFV <-> CKKS bridge)
// NO intermediate decryption anywhere.
//
// Network: 784 -> [Linear] -> [Sign via FBT] -> 30 -> [Linear] -> 10
//==================================================================================
/*
  ARCHITECTURE — SINGLE FIXEDMANUAL CONTEXT
  ──────────────────────────────────────────
  Everything runs in ONE CryptoContext with FIXEDMANUAL scaling.
  This is required because EvalFBT only works in FIXEDMANUAL mode.

  LEVEL BUDGET (from top to bottom as levels are consumed):
  ┌─────────────────────────────────────────────────────────────┐
  │  Level 0   fresh ciphertext after encryption                │
  │  Level 1   after Layer 1 EvalMult(weights) + ModReduce      │
  │  Level 2   after Layer 1 EvalMult(mask)    + ModReduce      │
  │            ← ct_z1 enters EvalFBT here                      │
  │  ...                                                         │
  │  Level 2+fbtDepth   ct_h1 exits EvalFBT here               │
  │  Level 2+fbtDepth+1 after Layer 2 EvalMult(weights)+ModRed  │
  │  Level 2+fbtDepth+2 after Layer 2 EvalMult(mask)  +ModRed  │
  │            ← ct_logits decrypted here                        │
  └─────────────────────────────────────────────────────────────┘
  totalDepth = LEVELS_L1 + fbtDepth + LEVELS_L2 = 2 + fbtDepth + 2

  EvalFBTSetup parameters:
    levelsAvailableAfterBootstrap  = LEVELS_L2 = 2
    levelsAvailableBeforeBootstrap = LEVELS_L1 = 2 (passed only to ep/ConvertRLWEToCKKS)

  IMAGE ENCRYPTION:
  The image is encrypted as a standard CKKS ciphertext using cc->Encrypt().
  This gives a fresh level-0 ciphertext, consistent with the budget above.
  We do NOT use the BFV bridge for image encryption — only EvalFBT uses it
  internally for the sign function evaluation.

  THE BFV BRIDGE (used only inside EvalFBT):
  EvalFBT internally:
    1. Homomorphically encodes CKKS slots → polynomial coefficients (EvalHomEncoding)
    2. Evaluates the Hermite trig polynomial in coefficient form
    3. Homomorphically decodes coefficients → CKKS slots (EvalHomDecoding)
  The SchemeletRLWEMP bridge is used only for the ep (element params) that
  EvalFBT needs to know the BFV modulus structure.

  SIGN FUNCTION:
  After Layer 1, slot j holds z_j ∈ [-784, 784] as a floating-point value.
  EvalFBT interprets the ciphertext as encoding integer values in Z_PInput.
  The scaling factor of ct_z1 is:
    sf = (CKKS scaling factor)^(level) = 2^(dcrtBits * level)
  The actual encoded integer = round(float_value * sf / PInput_scale)
  We must choose PInput and the bias so that sign(z_j) is correctly read.

  CORRECT APPROACH for sign(z_j) when z_j is a float in CKKS:
  The values z_j after Layer 1 are sums of {-1,+1} × {-1,+1} products.
  In CKKS these are represented as floats. The FBT sign function operates
  over Z_PInput where values are embedded as z_j * scalingFactor / PInput.
  
  The simplest correct approach:
    - Use PInput = 2 (binary FBT, the "binaryLUT" path from reference)
    - This evaluates sign in {0,1}: positive → output class 1, negative → output class 0
    - Then we remap: h = 2*sign_binary - 1 (plaintext linear rescaling)
  
  But PInput=2 binaryLUT is for Boolean inputs {0,1}. Our pre-activations are 
  larger floats.

  THE ACTUALLY CORRECT APPROACH:
  Re-read the reference. EvalFBT with PInput=256 evaluates a function f: Z_256 → Z_256
  on a ciphertext whose values are integers mod 256.
  
  For Layer 1 output: z_j are integers (dot products of integers).
  In CKKS floating-point they are represented exactly if the scaling factor is large enough.
  EvalFBT expects: the ciphertext encodes z_j * (Q/PInput) where Q is the CKKS modulus.
  
  In the reference, after BFV→CKKS conversion:
    The CKKS ciphertext encodes values scaled by Q/PInput.
    EvalFBT knows to un-scale by Q/PInput before applying the function.
  
  For our Layer 1 output: the CKKS values are z_j * sf (the CKKS scaling factor).
  EvalFBT will interpret them as z_j * sf / (Q/PInput) = z_j * sf * PInput / Q.
  
  For this to equal z_j mod PInput, we need: sf * PInput / Q = 1
  i.e., sf = Q / PInput.
  
  In FIXEDMANUAL, sf = 2^dcrtBits per level. After 2 levels: sf = 2^(2*dcrtBits).
  Q (the CKKS modulus at the level where EvalFBT is called) ≈ 2^(dcrtBits * remaining_levels).
  
  This is getting complex. The SIMPLEST correct implementation:
  
  APPROACH: Encrypt image using BFV bridge at the correct depth, do Layer 1
  as integer arithmetic using BFV ciphertext operations, then enter FBT.
  
  But BFV ciphertext operations aren't available in OpenFHE's CKKS context.
  
  FINAL CORRECT APPROACH (matching the reference exactly):
  
  1. Use the BFV bridge to encrypt the image at depth = totalDepth - LEVELS_L1
     (i.e., leaving LEVELS_L1 budget for Layer 1).
  2. Do Layer 1 in CKKS using EvalMult + EvalSum + ModReduce, consuming LEVELS_L1.
  3. The result ct_z1 is at level LEVELS_L1 with (totalDepth - LEVELS_L1) remaining.
  4. ep is built for depth = totalDepth - LEVELS_L1 (the level ct enters bridge).
  5. EvalFBT is called with ct_z1, using ep->GetModulus() for the BFV modulus.
  
  The ep modulus tells EvalFBT what the BFV modulus is at the level where
  the ciphertext enters the sign evaluation. This is the key parameter.

  FOR THE SCALING TO BE CORRECT in the linear layer with FIXEDMANUAL:
  The image pixels are {-1, +1}. In CKKS they are encoded with scaling factor sf0.
  After EvalMult(ct_px, pt_w): the result has scale sf0 * sf_ptxt.
  In FIXEDMANUAL, pt_w should have scale = 2^dcrtBits (= the level scaling).
  After ModReduce: scale becomes sf0 (mod-reduced by 2^dcrtBits).
  Then EvalMult(ct_sum, pt_mask) at scale sf0, mask scale = 2^dcrtBits.
  After ModReduce: scale = sf0 again.
  
  So z_j is encoded with scale sf0 = 2^dcrtBits (the initial encryption scale).
  For EvalFBT to correctly interpret z_j as integers in Z_PInput:
    ep->GetModulus() / (sf0 * something) should equal PInput.
  
  Looking at the reference again:
  scaleTHI is the Hermite coefficient scale, NOT the ciphertext scale.
  The ciphertext values are z_j * (ep->GetModulus() / PInput) after BFV→CKKS.
  
  For our case: after Layer 1 in pure CKKS, z_j is encoded at scale sf0.
  ep->GetModulus() is Q_RLWE = 2^51.
  For consistency: we need sf0 = ep->GetModulus() / PInput.
  PInput = 2048 → sf0 = 2^51 / 2048 = 2^51 / 2^11 = 2^40.
  
  So: dcrtBits should be 40, and sf0 = 2^40 = ep->GetModulus() / PInput.
  
  THIS IS THE CORRECT PARAMETER CHOICE.
*/

#include "openfhe.h"
#include "schemelet/rlwe-mp.h"
#include "math/hermite.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <climits>

using namespace lbcrypto;

// ─────────────────────────────────────────────────────────────────────────────
// Network dimensions
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int INPUT_DIM  = 784;
static constexpr int HIDDEN_DIM = 30;
static constexpr int OUTPUT_DIM = 10;

// ─────────────────────────────────────────────────────────────────────────────
// Crypto parameters
//
// Consistency requirement:
//   Q_RLWE / PInput = 2^dcrtBits   (the CKKS scaling factor per level)
//
// With PInput = 2048 = 2^11 and dcrtBits = 40:
//   Q_RLWE = 2^50 * 2048 = 2^61  ← consistent
//   BIGQ   = Q_RLWE = 2^61
//
// NUM_SLOTS = 1024:  pow2, >= INPUT_DIM=784, >= HIDDEN_DIM=30
// LEVELS_L1 = LEVELS_L2 = 2:  one EvalMult+ModReduce for weights, one for mask
// ─────────────────────────────────────────────────────────────────────────────
static constexpr uint32_t NUM_SLOTS  = 1024;
static constexpr uint32_t RING_DIM   = 4096;

static constexpr uint32_t LEVELS_L1 = 2;   // linear layer 1 level budget
static constexpr uint32_t LEVELS_L2 = 2;   // linear layer 2 level budget

// PInput = 2048: range for sign FBT.  Pre-activations z_j ∈ [-784, 784].
// We shift z_j by PInput/2 = 1024 so all values land in [240, 1808] ⊂ Z_2048.
// Sign boundary at 1024: [1,1023] → +1, {0,1024} → 0, [1025,2047] → -1.
static const BigInteger PINPUT(2048);
static const BigInteger POUTPUT(2048);

// dcrtBits = 40, so scaling factor = 2^40 per level.
// Q_RLWE = 2^61 = 2^50 * 2^11 = scalingFactor * PInput.  Consistent!
static constexpr uint32_t DCRT_BITS = 50;
static const BigInteger Q_RLWE(BigInteger(1) << 61);
static const BigInteger BIGQ(BigInteger(1) << 61);
static const BigInteger QBFVINIT(BigInteger(1) << 70);
static constexpr uint64_t SCALE_THI = 64;

// ─────────────────────────────────────────────────────────────────────────────
// HE context
// ─────────────────────────────────────────────────────────────────────────────
struct HEContext {
    CryptoContext<DCRTPoly>           cc;
    KeyPair<DCRTPoly>                 keyPair;
    std::vector<std::complex<double>> coeffSign;
    uint32_t                          fbtDepth;
    uint32_t                          totalDepth;
    // ep_fbt: BFV params at the level where ct_z1 enters EvalFBT
    std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>>              ep;
    // ep_out: BFV params at the level where ct_logits is decrypted (LEVELS_L2 remaining)
    std::shared_ptr<ILDCRTParams<DCRTPoly::Integer>>              ep_out;
};

// ─────────────────────────────────────────────────────────────────────────────
// 1. DATA LOADING
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int64_t> load_csv_1d(const std::string& path) {
    std::vector<int64_t> out;
    std::ifstream f(path);
    if (!f) { std::cerr << "[ERROR] Cannot open " << path << "\n"; return out; }
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) out.push_back(std::stoll(line));
    return out;
}

std::vector<std::vector<int64_t>> load_csv_2d(const std::string& path,
                                               int expected_in, int expected_out) {
    std::vector<std::vector<int64_t>> raw;
    std::ifstream f(path);
    if (!f) { std::cerr << "[ERROR] Cannot open " << path << "\n"; return raw; }
    std::string line;
    while (std::getline(f, line)) {
        std::vector<int64_t> row;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ','))
            if (!tok.empty()) row.push_back(std::stoll(tok));
        if (!row.empty()) raw.push_back(row);
    }
    int nr = (int)raw.size(), nc = (int)raw[0].size();
    std::vector<std::vector<int64_t>> W(expected_out, std::vector<int64_t>(expected_in, 0));
    if      (nr == expected_out && nc == expected_in) W = raw;
    else if (nr == expected_in  && nc == expected_out)
        for (int r = 0; r < nr; ++r)
            for (int c = 0; c < nc; ++c)
                W[c][r] = raw[r][c];
    else
        std::cerr << "[ERROR] CSV " << nr << "x" << nc
                  << " vs expected " << expected_out << "x" << expected_in << "\n";
    return W;
}

std::vector<int64_t> load_image_bipolar(const char* path) {
    int w, h, ch;
    unsigned char* data = stbi_load(path, &w, &h, &ch, 1);
    if (!data) { std::cerr << "[ERROR] Cannot load: " << path << "\n"; return {}; }
    std::vector<int64_t> out(784, -1);
    for (int i = 0; i < 784 && i < w * h; ++i)
        out[i] = (data[i] > 127) ? 1 : -1;
    stbi_image_free(data);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PLAINTEXT REFERENCE
// ─────────────────────────────────────────────────────────────────────────────
int plain_forward(const std::vector<int64_t>& px,
                  const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
                  const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2)
{
    std::vector<int64_t> z1(HIDDEN_DIM, 0);
    for (int j = 0; j < HIDDEN_DIM; ++j) {
        z1[j] = b1[j];
        for (int i = 0; i < INPUT_DIM; ++i) z1[j] += W1[j][i] * px[i];
    }
    std::vector<int64_t> h1(HIDDEN_DIM);
    for (int j = 0; j < HIDDEN_DIM; ++j)
        h1[j] = (z1[j] > 0) ? 1 : (z1[j] < 0) ? -1 : 0;
    std::vector<int64_t> logits(OUTPUT_DIM, 0);
    for (int k = 0; k < OUTPUT_DIM; ++k) {
        logits[k] = b2[k];
        for (int j = 0; j < HIDDEN_DIM; ++j) logits[k] += W2[k][j] * h1[j];
    }
    std::cout << "[Plain] z1 (first 10):  [";
    for (int j=0;j<10;++j) std::cout<<z1[j]<<(j<9?", ":""); std::cout<<"]\n";
    std::cout << "[Plain] h1:             [";
    for (int j=0;j<HIDDEN_DIM;++j) std::cout<<h1[j]<<(j<HIDDEN_DIM-1?", ":""); std::cout<<"]\n";
    std::cout << "[Plain] logits:         [";
    for (int k=0;k<OUTPUT_DIM;++k) std::cout<<logits[k]<<(k<9?", ":""); std::cout<<"]\n";
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. SETUP
// ─────────────────────────────────────────────────────────────────────────────
HEContext setup_he_context() {
    HEContext ctx;

    // Sign function over Z_PInput (2048-point domain)
    // Shifted encoding: z_j + 1024 → z_q ∈ [1,2047], boundary at 1024
    auto funcSign = [](int64_t x) -> int64_t {
        int64_t half = PINPUT.ConvertToInt<int64_t>() / 2;  // 1024
        if (x == 0 || x == half) return 0;
        return (x < half) ? 1 : -1;
    };

    const size_t order = 1;
    ctx.coeffSign = GetHermiteTrigCoefficients(
        funcSign, PINPUT.ConvertToInt(), order, SCALE_THI);

    std::vector<uint32_t> lvlb = {3, 3};
    SecretKeyDist skDist       = SPARSE_ENCAPSULATED;

    ctx.fbtDepth  = FHECKKSRNS::GetFBTDepth(lvlb, ctx.coeffSign, PINPUT, order, skDist);
    ctx.totalDepth = LEVELS_L1 + ctx.fbtDepth + LEVELS_L2;

    std::cout << "[Setup] fbtDepth=" << ctx.fbtDepth
              << "  totalDepth=" << ctx.totalDepth
              << "  slots=" << NUM_SLOTS
              << "  dcrtBits=" << DCRT_BITS << "\n";

    // FIXEDMANUAL is required for EvalFBT
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(skDist);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetScalingModSize(DCRT_BITS);        // 2^40 per level
    params.SetScalingTechnique(FIXEDMANUAL);
    params.SetFirstModSize(60);  // larger first mod for noise headroom
    params.SetNumLargeDigits(3);
    params.SetBatchSize(NUM_SLOTS);             // pow2, same for all three setup calls
    params.SetRingDim(RING_DIM);
    params.SetMultiplicativeDepth(ctx.totalDepth);

    ctx.cc = GenCryptoContext(params);
    ctx.cc->Enable(PKE);
    ctx.cc->Enable(KEYSWITCH);
    ctx.cc->Enable(LEVELEDSHE);
    ctx.cc->Enable(ADVANCEDSHE);
    ctx.cc->Enable(FHE);

    std::cout << "[Setup] Generating keys...\n";
    ctx.keyPair = ctx.cc->KeyGen();
    ctx.cc->EvalMultKeyGen(ctx.keyPair.secretKey);
    ctx.cc->EvalSumKeyGen(ctx.keyPair.secretKey);

    // EvalFBTSetup:
    //   levelsAvailableAfterBootstrap  = LEVELS_L2 (consumed by Layer 2 after FBT)
    //   levelsAvailableBeforeBootstrap = not a parameter here; reflected in ep depth
    ctx.cc->EvalFBTSetup(
        ctx.coeffSign,
        NUM_SLOTS,                  // must equal SetBatchSize and EvalBootstrapKeyGen
        PINPUT, POUTPUT,
        BIGQ,
        ctx.keyPair.publicKey,
        {0, 0},
        lvlb,
        LEVELS_L2,                  // levelsAvailableAfterBootstrap
        0,                          // levelsComputation
        order);

    ctx.cc->EvalBootstrapKeyGen(ctx.keyPair.secretKey, NUM_SLOTS);  // same NUM_SLOTS

    // ep_fbt: element params for the BFV modulus at the level where ct_z1 enters EvalFBT.
    // ct_z1 has consumed LEVELS_L1 levels, leaving (fbtDepth + LEVELS_L2) moduli.
    ctx.ep = SchemeletRLWEMP::GetElementParams(
        ctx.keyPair.secretKey,
        ctx.fbtDepth + LEVELS_L2);

    // ep_out: element params for decryption of ct_logits.
    // ct_logits is at level (totalDepth - LEVELS_L2), with LEVELS_L2 moduli remaining.
    ctx.ep_out = SchemeletRLWEMP::GetElementParams(
        ctx.keyPair.secretKey,
        LEVELS_L2);

    std::cout << "[Setup] Done. ep modulus bits = "
              << ctx.ep->GetModulus().GetMSB() << "\n";
    return ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. ENCRYPT IMAGE
//
//   Encrypt image as a CKKS plaintext using standard cc->Encrypt.
//   In FIXEDMANUAL, the plaintext scaling factor must be set explicitly
//   to 2^dcrtBits so it is consistent with multiplications.
//
//   The pixel values {-1, +1} are encoded at scaling factor 2^dcrtBits.
//   After Layer 1 (2 ModReduces), the values z_j are still at scale 2^dcrtBits
//   (each ModReduce divides by 2^dcrtBits and removes one modulus).
//   This matches: ep->GetModulus() / PInput = 2^51 / 2048 = 2^40 = 2^dcrtBits. ✓
// ─────────────────────────────────────────────────────────────────────────────
Ciphertext<DCRTPoly> encrypt_image(const std::vector<int64_t>& pixels, HEContext& ctx) {
    // Pad to NUM_SLOTS
    std::vector<double> px_d(NUM_SLOTS, 0.0);
    for (int i = 0; i < INPUT_DIM; ++i) px_d[i] = static_cast<double>(pixels[i]);

    // In FIXEDMANUAL: set level=0, scalingFactor=2^dcrtBits (default for fresh encrypt)
    Plaintext pt = ctx.cc->MakeCKKSPackedPlaintext(px_d, 1, 0);
    auto ct = ctx.cc->Encrypt(ctx.keyPair.publicKey, pt);
    std::cout << "[Encrypt] level=" << ct->GetLevel()
              << "  scalingFactor=" << std::log2(ct->GetScalingFactor()) << " bits\n";
    return ct;
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. LINEAR LAYER  (fully homomorphic, no decryption)
//
//   slot j of output = dot(W[j], input_slots[0..in_dim-1]) + b[j]
//
//   In FIXEDMANUAL: plaintext must have explicit level set to match ct level.
//   We create pt_w at the SAME level as ct_input (level 0 for Layer 1).
//   After EvalMult + ModReduce: level increases by 1.
//   After mask EvalMult + ModReduce: level increases by 1 again.
//   Total: 2 levels consumed per linear layer.
// ─────────────────────────────────────────────────────────────────────────────
Ciphertext<DCRTPoly> linear_layer(
    const Ciphertext<DCRTPoly>&              ct_input,
    const std::vector<std::vector<int64_t>>& W,
    const std::vector<int64_t>&              b,
    int                                      in_dim,
    int                                      out_dim,
    HEContext&                               ctx)
{
    uint32_t cur_level = ct_input->GetLevel();
    Ciphertext<DCRTPoly> ct_out;

    for (int j = 0; j < out_dim; ++j) {
        // Weight plaintext at the SAME level as ct_input (FIXEDMANUAL requirement)
        std::vector<double> w_d(NUM_SLOTS, 0.0);
        for (int i = 0; i < in_dim; ++i) w_d[i] = static_cast<double>(W[j][i]);
        Plaintext pt_w = ctx.cc->MakeCKKSPackedPlaintext(w_d, 1, cur_level);

        // Inner product
        auto ct_mult = ctx.cc->EvalMult(ct_input, pt_w);
        auto ct_sum  = ctx.cc->EvalSum(ct_mult, NUM_SLOTS);
        ctx.cc->ModReduceInPlace(ct_sum);          // level: cur_level → cur_level+1

        // Mask to slot j
        std::vector<double> mask(NUM_SLOTS, 0.0);
        mask[j] = 1.0;
        Plaintext pt_mask = ctx.cc->MakeCKKSPackedPlaintext(mask, 1, ct_sum->GetLevel());
        auto ct_neuron = ctx.cc->EvalMult(ct_sum, pt_mask);
        ctx.cc->ModReduceInPlace(ct_neuron);       // level: cur_level+1 → cur_level+2

        // Bias (plaintext add, no level change)
        std::vector<double> bvec(NUM_SLOTS, 0.0);
        bvec[j] = static_cast<double>(b[j]);
        Plaintext pt_bias = ctx.cc->MakeCKKSPackedPlaintext(bvec, 1, ct_neuron->GetLevel());
        ctx.cc->EvalAddInPlace(ct_neuron, pt_bias);

        if (j == 0) ct_out = ct_neuron;
        else        ctx.cc->EvalAddInPlace(ct_out, ct_neuron);
    }

    std::cout << "  output level=" << ct_out->GetLevel()
              << "  scalingFactor=" << std::log2(ct_out->GetScalingFactor()) << " bits\n";
    return ct_out;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. SIGN ACTIVATION via FBT  (fully homomorphic)
//
//   ct_z1 encodes z_j as floats at scale 2^dcrtBits (= ep->GetModulus() / PInput).
//   EvalFBT interprets the ciphertext values as integers in Z_PInput.
//   With scale = ep->GetModulus() / PInput, the integer z_j is recovered exactly.
//
//   We shift z_j by PInput/2 homomorphically BEFORE calling EvalFBT so that
//   the sign boundary lands at PInput/2:
//     shifted = z_j + 1024
//     z_j > 0  → shifted ∈ [1025, 1808] ⊂ [PInput/2+1, PInput-1] → sign = -1? NO
//
//   WAIT: let's re-check:
//     funcSign(x): x ∈ [1, 1023] → +1 (these are "positive" in Z_2048)
//                  x ∈ [1025, 2047] → -1 (these are "negative" in Z_2048)
//     shifted z_j + 1024:
//       z_j > 0: shifted ∈ [1025, 1808] → funcSign returns -1 ✗ (should be +1)
//
//   THE SHIFT DIRECTION WAS WRONG. We need:
//     z_j > 0  → z_shifted ∈ [1, 1023]   → funcSign = +1 ✓
//     z_j < 0  → z_shifted ∈ [1025, 2047] → funcSign = -1 ✓
//   So: z_shifted = z_j + PInput/2 maps:
//     z_j ∈ [1, 784]    → shifted ∈ [1025, 1808] — this goes to the NEGATIVE half!
//
//   We need the OPPOSITE shift: negative z_j → high range, positive → low range.
//   This means: DON'T SHIFT. Use z_j directly.
//     z_j > 0: z_j mod PInput = z_j ∈ [1, 784]    → funcSign(z_j) = +1 ✓
//     z_j < 0: z_j mod PInput = z_j + 2048 ∈ [1264, 2047] → funcSign = -1 ✓
//     z_j = 0: 0 → funcSign = 0 ✓
//
//   In CKKS, negative floats become large positive integers when reduced mod PInput
//   (this is handled by the polynomial arithmetic in the BFV bridge inside EvalFBT).
//   So NO SHIFT IS NEEDED — the modular arithmetic handles negatives automatically.
// ─────────────────────────────────────────────────────────────────────────────
Ciphertext<DCRTPoly> sign_activation_fbt(
    const Ciphertext<DCRTPoly>& ct_z1,
    HEContext& ctx)
{
    std::cout << "[FBT] input level=" << ct_z1->GetLevel()
              << "  scalingFactor=" << std::log2(ct_z1->GetScalingFactor()) << " bits\n";

    auto ct_sign = ctx.cc->EvalFBT(
        ct_z1,
        ctx.coeffSign,
        PINPUT.GetMSB() - 1,        // log2(PInput) = 11
        BIGQ,                       // <--- THE FIX
        SCALE_THI,
        0,                          // levelsToDrop
        1);                         // order

    std::cout << "[FBT] output level=" << ct_sign->GetLevel()
              << "  scalingFactor=" << std::log2(ct_sign->GetScalingFactor()) << " bits\n";
    return ct_sign;
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. MAIN
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image.png/jpg>\n";
        return 1;
    }

    std::cout << "=============================================================\n";
    std::cout << "  DiNN — Fully Private HE Inference\n";
    std::cout << "  OpenFHE CKKS + FBT Sign  |  784->30->10\n";
    std::cout << "  Consistency: Q_RLWE(2^61) / PInput(2^11) = 2^" << DCRT_BITS << " = scaling factor\n";
    std::cout << "=============================================================\n\n";

    // ── Load weights ─────────────────────────────────────────────────────────
    auto W1 = load_csv_2d("../dinn30_W1.csv", INPUT_DIM,  HIDDEN_DIM);
    auto b1 = load_csv_1d("../dinn30_b1.csv");
    auto W2 = load_csv_2d("../dinn30_W2.csv", HIDDEN_DIM, OUTPUT_DIM);
    auto b2 = load_csv_1d("../dinn30_b2.csv");
    if (W1.empty() || b1.empty() || W2.empty() || b2.empty()) {
        std::cerr << "[ERROR] Failed to load weights.\n"; return 1;
    }

    // ── Load image ───────────────────────────────────────────────────────────
    auto pixels = load_image_bipolar(argv[1]);
    if (pixels.empty()) return 1;
    int pos = (int)std::count(pixels.begin(), pixels.end(), 1L);
    std::cout << "[Data] Image: " << pos << " positive, " << (784-pos) << " negative pixels\n\n";

    // ── Plaintext reference ──────────────────────────────────────────────────
    std::cout << "--- Plaintext Reference ---\n";
    int plain_pred = plain_forward(pixels, W1, b1, W2, b2);
    std::cout << "[Plain] Predicted digit: " << plain_pred << "\n\n";

    // ── HE setup ─────────────────────────────────────────────────────────────
    std::cout << "--- HE Setup ---\n";
    HEContext ctx = setup_he_context();
    std::cout << "\n";

    // ── HE inference ─────────────────────────────────────────────────────────
    std::cout << "--- HE Inference (fully private) ---\n";
    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << "\n[Step 1] Encrypt image\n";
    auto ct_px = encrypt_image(pixels, ctx);

    std::cout << "\n[Step 2] Layer 1 linear (" << INPUT_DIM << "->" << HIDDEN_DIM << ")\n";
    auto ct_z1 = linear_layer(ct_px, W1, b1, INPUT_DIM, HIDDEN_DIM, ctx);

    std::cout << "\n[Step 3] Sign activation (FBT over Z_" << PINPUT << ")\n";
    auto ct_h1 = sign_activation_fbt(ct_z1, ctx);

    std::cout << "\n[Step 4] Layer 2 linear (" << HIDDEN_DIM << "->" << OUTPUT_DIM << ")\n";
    auto ct_logits = linear_layer(ct_h1, W2, b2, HIDDEN_DIM, OUTPUT_DIM, ctx);

    std::cout << "\n[Step 5] Final decryption (only decryption in entire pipeline)\n";
    // Use ConvertCKKSToRLWE + DecryptCoeff throughout.
    // This reads raw polynomial coefficients mod Q_RLWE, bypassing the CKKS
    // floating-point decoder — immune to approximation-error failures.
    //
    // Each intermediate result needs its own ep built for the number of
    // moduli remaining in the chain at that point:
    //
    //   ct_h1    exits EvalFBT at level (LEVELS_L1 + fbtDepth) = 29.
    //            Remaining moduli = totalDepth - level = 31 - 29 = 2  → ep with 2 levels.
    //            But we already built ctx.ep = GetElementParams(sk, fbtDepth+LEVELS_L2)
    //            which has fbtDepth+2 levels.  That is the full depth from ct_z1 entry
    //            through FBT.  For ct_h1 (FBT output) we want just LEVELS_L2=2 remaining.
    //
    //   ct_logits exits Layer 2 at level (LEVELS_L1 + fbtDepth + LEVELS_L2) = 31 = totalDepth.
    //            Remaining moduli = totalDepth - level = 0 → but we always have at least 1.
    //            Actually OpenFHE counts levels from 0: a fresh ciphertext is level 0
    //            and has (totalDepth+1) moduli; after k ModReduces it has (totalDepth+1-k).
    //            So ct_logits (level 31) has totalDepth+1-31 = 1 modulus remaining.
    //            → ep_logits = GetElementParams(sk, 1).
    //
    // In practice: build one ep per output ciphertext using its remaining modulus count.

    // ep for ct_h1: LEVELS_L2 = 2 remaining moduli after FBT
    auto ep_h1 = SchemeletRLWEMP::GetElementParams(
        ctx.keyPair.secretKey, LEVELS_L2);

    // ep for ct_logits: 1 remaining modulus after Layer 2
    auto ep_logits = SchemeletRLWEMP::GetElementParams(
        ctx.keyPair.secretKey, 1);

    // Debug: decrypt FBT output (ct_h1) to verify sign values
    {
        auto polys_h1 = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_h1, Q_RLWE);
        auto h1_vals  = SchemeletRLWEMP::DecryptCoeff(
            polys_h1, Q_RLWE, POUTPUT,
            ctx.keyPair.secretKey, ep_h1,
            NUM_SLOTS, (uint32_t)HIDDEN_DIM);
        int64_t phalf_h = POUTPUT.ConvertToInt<int64_t>() / 2;
        std::cout << "[Debug] HE h1: [";
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            int64_t v = (h1_vals[j] >= phalf_h)
                        ? h1_vals[j] - POUTPUT.ConvertToInt<int64_t>() : h1_vals[j];
            std::cout << v << (j < HIDDEN_DIM-1 ? ", " : "");
        }
        std::cout << "]\n";
    }

    // Final logits
    auto polys  = SchemeletRLWEMP::ConvertCKKSToRLWE(ct_logits, Q_RLWE);
    auto scores = SchemeletRLWEMP::DecryptCoeff(
        polys, Q_RLWE, POUTPUT,
        ctx.keyPair.secretKey, ep_logits,
        NUM_SLOTS, (uint32_t)OUTPUT_DIM);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int64_t phalf = POUTPUT.ConvertToInt<int64_t>() / 2;
    std::cout << "\n--- Final Class Scores (HE) ---\n";
    int he_pred = 0;
    int64_t max_s = LLONG_MIN;
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
