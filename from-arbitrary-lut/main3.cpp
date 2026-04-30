#include "openfhe.h"
#include "schemelet/rlwe-mp.h"

using namespace lbcrypto;

int main() {
    // === Parameters ===
    const BigInteger QBFVInit(BigInteger(1) << 60);
    const BigInteger Q(BigInteger(1) << 33);
    const BigInteger Bigq(BigInteger(1) << 33);
    const BigInteger PInput(256);       // plaintext modulus (8-bit input)
    const uint32_t   numSlots  = 8;
    const uint32_t   ringDim   = 4096;
    const uint32_t   dcrtBits  = Bigq.GetMSB() - 1;
    const uint32_t   firstMod  = Bigq.GetMSB() - 1;
    const uint32_t   depth     = 1;    // one multiplication level
    const uint32_t   dnum      = 0;

    bool flagSP       = (numSlots <= ringDim / 2);   // sparse packing
    auto numSlotsCKKS = flagSP ? numSlots : numSlots / 2;

    // === Input vector ===
    std::vector<int64_t> x = {0, 1, 2, 3, 4, 5, 6, 7};

    // Plaintext multiplier (real-valued, same slot count)
    std::vector<double> multiplier = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};

    // === CKKS crypto context ===
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(SPARSE_ENCAPSULATED);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(dnum);
    parameters.SetBatchSize(numSlotsCKKS);
    parameters.SetRingDim(ringDim);
    parameters.SetMultiplicativeDepth(depth);

    auto cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keyPair = cc->KeyGen();
    cc->EvalMultKeyGen(keyPair.secretKey);

    // === Encrypt in RLWE (BFV-style coefficient encoding) ===
    //   GetElementParams picks element params matching the secret key at the given level
    auto ep = SchemeletRLWEMP::GetElementParams(keyPair.secretKey, depth);

    auto ctxtBFV = SchemeletRLWEMP::EncryptCoeff(x, QBFVInit, PInput, keyPair.secretKey, ep);

    // Mod-switch from QBFVInit down to Q to reduce encryption error
    SchemeletRLWEMP::ModSwitch(ctxtBFV, Q, QBFVInit);

    // === Convert RLWE → CKKS ===
    auto ctxt = SchemeletRLWEMP::ConvertRLWEToCKKS(
        *cc, ctxtBFV, keyPair.publicKey, Bigq, numSlotsCKKS, depth);

    // === Multiply by plaintext ===
    Plaintext ptxtMult = cc->MakeCKKSPackedPlaintext(multiplier, 1, 0, nullptr, numSlotsCKKS);
    auto ctxtMult      = cc->EvalMult(ctxt, ptxtMult);
    // cc->ModReduceInPlace(ctxtMult);

    // === Convert CKKS → RLWE and decrypt ===
    auto polys    = SchemeletRLWEMP::ConvertCKKSToRLWE(ctxtMult, Q);
    auto computed = SchemeletRLWEMP::DecryptCoeff(
        polys, Q, PInput, keyPair.secretKey, ep, numSlotsCKKS, numSlots);

    // === Check correctness ===
    std::cout << "Input:    ";
    for (auto v : x)        std::cout << v << " ";
    std::cout << "\nExpected: ";
    for (auto v : x)        std::cout << v * 2 << " ";
    std::cout << "\nObtained: ";
    for (auto v : computed) std::cout << v << " ";

    int64_t maxErr = 0;
    for (size_t i = 0; i < x.size(); ++i)
        maxErr = std::max(maxErr, std::abs(computed[i] - x[i] * 2));
    std::cout << "\nMax absolute error: " << maxErr << std::endl;

    return 0;
}
