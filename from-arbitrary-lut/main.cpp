#include "openfhe.h"
#include "schemelet/rlwe-mp.h"
#include "math/hermite.h"

using namespace lbcrypto;

using RLWECt = std::vector<Poly>;

// ── Helper: scalar multiplication (mod Q) ────────────────────────────────────
RLWECt ScalarMult(const RLWECt& ct, int64_t scalar, const BigInteger& Q) {
    RLWECt result = ct;
    BigInteger s  = (scalar < 0) ? Q - BigInteger(static_cast<uint64_t>(-scalar))
                                 : BigInteger(static_cast<uint64_t>(scalar));
    for (uint32_t i = 0; i < ct[0].GetLength(); ++i) {
        result[0][i] = ct[0][i].ModMul(s, Q);
        result[1][i] = ct[1][i].ModMul(s, Q);
    }
    return result;
}

// ── Helper: ciphertext addition (mod Q) ──────────────────────────────────────
RLWECt Add(const RLWECt& ct1, const RLWECt& ct2, const BigInteger& Q) {
    RLWECt result = ct1;
    for (uint32_t i = 0; i < ct1[0].GetLength(); ++i) {
        result[0][i] = ct1[0][i].ModAdd(ct2[0][i], Q);
        result[1][i] = ct1[1][i].ModAdd(ct2[1][i], Q);
    }
    return result;
}

// ── Helper: plaintext addition — add a constant to every slot ────────────────
// The constant c is added to slot j by adding delta*c to position j*gap.
RLWECt AddPlaintext(const RLWECt& ct, const std::vector<int64_t>& constants,
                    const BigInteger& Q, const BigInteger& p,
                    uint32_t numSlots) {
    RLWECt result   = ct;
    BigInteger delta = Q / p;
    uint32_t gap     = ct[0].GetLength() / (2 * numSlots);
    gap              = (gap == 0) ? 1 : gap;

    for (uint32_t i = 0; i < numSlots && i < constants.size(); ++i) {
        int64_t c      = constants[i] % p.ConvertToInt<int64_t>();
        BigInteger val = (c < 0) ? Q - delta * BigInteger(static_cast<uint64_t>(-c))
                                 : delta * BigInteger(static_cast<uint64_t>(c));
        result[0][i * gap] = result[0][i * gap].ModAdd(val, Q);
        // Also update the mirrored position (gap > 1 case, matches EncryptCoeff)
        if (gap > 1)
            result[0][(i + numSlots) * gap] =
                result[0][(i + numSlots) * gap].ModAdd(val, Q);
    }
    return result;
}

// ── Helper: slot sum — sum slots 0..inDim-1 into slot 0 ──────────────────────
// In coefficient space, slot i lives at position i*gap.
// Adding ct rotated by i*gap positions (monomial x^{i*gap}) brings
// slot i's contribution into slot 0.
// Polynomial rotation by k: coeff[j] -> coeff[(j+k) mod N] with sign flip
// for the wrapped-around part (negacyclic ring). We implement this directly.
RLWECt SlotSum(const RLWECt& ct, uint32_t inDim, uint32_t numSlots,
               const BigInteger& Q) {
    uint32_t N   = ct[0].GetLength();
    uint32_t gap = N / (2 * numSlots);
    gap          = (gap == 0) ? 1 : gap;

    RLWECt result = ct;  // accumulator, starts with slot 0 contribution

    for (uint32_t s = 1; s < inDim; ++s) {
        uint32_t shift = s * gap;  // rotate by this many coefficients

        // Rotate both polynomials of ct by 'shift' in the negacyclic ring:
        // p_rot[j] = p[(j + shift) mod N], with negation for wrap-around
        RLWECt rotated = ct;
        for (uint32_t poly = 0; poly < 2; ++poly) {
            for (uint32_t j = 0; j < N; ++j) {
                uint32_t src = (j + shift) % N;
                if (j + shift >= N) {
                    // Wrap-around in negacyclic ring: negate
                    BigInteger val = ct[poly][src];
                    rotated[poly][j] = (val == BigInteger(0)) ? BigInteger(0)
                                                              : Q - val;
                } else {
                    rotated[poly][j] = ct[poly][src];
                }
            }
        }
        result = Add(result, rotated, Q);
    }
    return result;
}

int main() {
    // ── Parameters ────────────────────────────────────────────────────────────
    const uint32_t ringDim  = 4096;
    const uint32_t numSlots = 8;
    const BigInteger p(256);
    const BigInteger Q        = BigInteger(1) << 47;
    const BigInteger QBFVInit = BigInteger(1) << 60;
    size_t order = 1;
    uint64_t scaleTHI = 32;
    SecretKeyDist secretKeyDist             = SPARSE_ENCAPSULATED;

    std::function<int64_t(int64_t)> func = [](int64_t x) { return x; };

    std::vector<int64_t> x = {10, 20, 30, 40, 5, 6, 7, 8};
    std::vector<int64_t> y = { 1,  2,  3,  4, 1, 2, 3, 4};

    bool flagSP       = (numSlots <= ringDim / 2);  // sparse packing
    auto numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    std::vector<std::complex<double>> coeffcomp = GetHermiteTrigCoefficients(func, p.ConvertToInt(), order, scaleTHI);


    // ── Context and keys (minimal — no FHE, no bootstrapping) ─────────────────
    const uint32_t dcrtBits = Q.GetMSB() - 1;

    std::vector<uint32_t> lvlb              = {3, 3};
    CCParams<CryptoContextCKKSRNS> params;
    params.SetSecretKeyDist(secretKeyDist);
    params.SetSecurityLevel(HEStd_NotSet);
    params.SetScalingModSize(dcrtBits);
    params.SetScalingTechnique(FIXEDMANUAL);
    params.SetFirstModSize(dcrtBits);
    params.SetNumLargeDigits(3);
    params.SetBatchSize(numSlots);
    params.SetRingDim(ringDim);
    uint32_t depth = FHECKKSRNS::GetFBTDepth(lvlb, coeffcomp, p, order, secretKeyDist);
    params.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(params);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);
    auto keyPair = cc->KeyGen();
    auto ep      = SchemeletRLWEMP::GetElementParams(keyPair.secretKey, depth);

    cc->EvalFBTSetup(coeffcomp, numSlots, p, p, Q, keyPair.publicKey, {0, 0}, lvlb,
                         0, 0, order);
    cc->EvalBootstrapKeyGen(keyPair.secretKey, numSlotsCKKS);
    cc->EvalMultKeyGen(keyPair.secretKey);

    // ── Encrypt x and y ───────────────────────────────────────────────────────
    auto ctX = SchemeletRLWEMP::EncryptCoeff(x, QBFVInit, p, keyPair.secretKey, ep);
    SchemeletRLWEMP::ModSwitch(ctX, Q, QBFVInit);

    auto ctY = SchemeletRLWEMP::EncryptCoeff(y, QBFVInit, p, keyPair.secretKey, ep);
    SchemeletRLWEMP::ModSwitch(ctY, Q, QBFVInit);

    // ── Operation 1: scalar multiplication ct = x * 3 ────────────────────────
    auto ctMul = ScalarMult(ctX, 3, Q);

    auto rMul = SchemeletRLWEMP::DecryptCoeff(ctMul, Q, p, keyPair.secretKey, ep, numSlots, numSlots);
    std::cout << "=== x * 3 ===\n";
    std::cout << "idx | input | expected | got  | ok?\n";
    for (size_t i = 0; i < numSlots; ++i) {
        int64_t exp = (x[i] * 3) % p.ConvertToInt();
        std::printf("  %zu |   %3lld |      %3lld | %3lld | %s\n",
                    i, x[i], exp, rMul[i], rMul[i] == exp ? "OK" : "FAIL");
    }

    // ── Operation 2: ciphertext addition ct = x + y ──────────────────────────
    auto ctAdd = Add(ctX, ctY, Q);

    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(*cc, ctAdd, keyPair.publicKey, Q, numSlots, depth);

    Ciphertext<DCRTPoly> ctxtAfterFBT = cc->EvalFBT(ctxt, coeffcomp, p.GetMSB() - 1, ep->GetModulus(), scaleTHI, 0, order);

    ctAdd = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtAfterFBT, Q);

    auto rAdd = SchemeletRLWEMP::DecryptCoeff(ctAdd, Q, p, keyPair.secretKey, ep, numSlots, numSlots);
    std::cout << "\n=== x + y ===\n";
    std::cout << "idx | x  | y  | expected | got  | ok?\n";
    for (size_t i = 0; i < numSlots; ++i) {
        int64_t exp = (x[i] + y[i]) % p.ConvertToInt();
        std::printf("  %zu | %3lld| %3lld|      %3lld | %3lld | %s\n",
                    i, x[i], y[i], exp, rAdd[i], rAdd[i] == exp ? "OK" : "FAIL");
    }

    // ── Operation 3: plaintext constant addition ct = x + [100,0,0,...] ───────
    std::vector<int64_t> bias(numSlots, 0);
    bias[0] = 100;
    auto ctBias = AddPlaintext(ctX, bias, Q, p, numSlots);

    auto rBias = SchemeletRLWEMP::DecryptCoeff(ctBias, Q, p, keyPair.secretKey, ep, numSlots, numSlots);
    std::cout << "\n=== x + bias[100,0,...] ===\n";
    std::cout << "idx | input | expected | got  | ok?\n";
    for (size_t i = 0; i < numSlots; ++i) {
        int64_t exp = (x[i] + bias[i]) % p.ConvertToInt();
        std::printf("  %zu |   %3lld |      %3lld | %3lld | %s\n",
                    i, x[i], exp, rBias[i], rBias[i] == exp ? "OK" : "FAIL");
    }

    // ── Operation 4: chained scalar mults ct = x * 3 * 5 ─────────────────────
    auto ctChain = ScalarMult(ScalarMult(ctX, 3, Q), 5, Q);

    auto rChain = SchemeletRLWEMP::DecryptCoeff(ctChain, Q, p, keyPair.secretKey, ep, numSlots, numSlots);
    std::cout << "\n=== x * 3 * 5 ===\n";
    std::cout << "idx | input | expected | got  | ok?\n";
    for (size_t i = 0; i < numSlots; ++i) {
        int64_t exp = (x[i] * 15) % p.ConvertToInt();
        std::printf("  %zu |   %3lld |      %3lld | %3lld | %s\n",
                    i, x[i], exp, rChain[i], rChain[i] == exp ? "OK" : "FAIL");
    }

    // ── Operation 5: slot sum — sum first 4 slots into slot 0 ────────────────
    // x = {10,20,30,40,...} → slot0 should become 10+20+30+40 = 100
    const uint32_t inDim = 4;
    auto ctSum = SlotSum(ctX, inDim, numSlots, Q);

    auto rSum = SchemeletRLWEMP::DecryptCoeff(ctSum, Q, p, keyPair.secretKey, ep, numSlots, numSlots);
    int64_t expSum = 0;
    for (uint32_t i = 0; i < inDim; ++i) expSum += x[i];
    expSum %= p.ConvertToInt();
    std::cout << "\n=== sum of first " << inDim << " slots of x ===\n";
    std::cout << "Expected in slot 0: " << expSum << "\n";
    std::cout << "idx | got  | ok?\n";
    for (size_t i = 0; i < numSlots; ++i) {
        bool ok = (i == 0) ? (rSum[i] == expSum) : true;
        std::printf("  %zu | %3lld | %s\n", i, rSum[i], ok ? "OK" : "(other slot)");
    }

    // ── Operation 6: combined — dot product W·x + bias for one output neuron ──
    // W[0] = {2, 3, 1, 4, 0, 0, 0, 0}, bias[0] = 7
    // result = 2*10 + 3*20 + 1*30 + 4*40 + 7 = 20+60+30+160+7 = 277 mod 256 = 21
    std::vector<int64_t> W0 = {2, 3, 1, 4, 0, 0, 0, 0};
    int64_t b0 = 7;

    // Encrypt each x[i] separately, scale by W0[i], sum them up
    RLWECt ctDot;
    bool first = true;
    for (uint32_t i = 0; i < inDim; ++i) {
        if (W0[i] == 0) continue;
        // Encrypt just x[i] in slot i
        std::vector<int64_t> xi(numSlots, 0);
        xi[i] = x[i];
        auto ctxi = SchemeletRLWEMP::EncryptCoeff(xi, QBFVInit, p, keyPair.secretKey, ep);
        SchemeletRLWEMP::ModSwitch(ctxi, Q, QBFVInit);
        auto ctWx = ScalarMult(ctxi, W0[i], Q);
        if (first) { ctDot = ctWx; first = false; }
        else        ctDot  = Add(ctDot, ctWx, Q);
    }
    // Add bias
    std::vector<int64_t> biasVec(numSlots, 0);
    biasVec[0] = b0;
    ctDot = AddPlaintext(ctDot, biasVec, Q, p, numSlots);

    auto rDot = SchemeletRLWEMP::DecryptCoeff(ctDot, Q, p, keyPair.secretKey, ep, numSlots, numSlots);
    int64_t expDot = 0;
    for (uint32_t i = 0; i < inDim; ++i) expDot += W0[i] * x[i];
    expDot = (expDot + b0) % p.ConvertToInt();
    std::cout << "\n=== dot product W·x + bias (slot 0) ===\n";
    std::printf("Expected: %lld  Got: %lld  %s\n",
                expDot, rDot[0], rDot[0] == expDot ? "OK" : "FAIL");

    return 0;
}
