#include "openfhe.h"
#include "schemelet/rlwe-mp.h"
#include "math/hermite.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace lbcrypto;

// ====================================================================
// 1. DATA LOADING
// ====================================================================
std::vector<int64_t> load_csv_1d(const std::string& filename) {
    std::vector<int64_t> result;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        result.push_back(std::stoll(line));
    }
    return result;
}

std::vector<std::vector<int64_t>> load_csv_2d(const std::string& filename, int expected_in, int expected_out) {
    std::vector<std::vector<int64_t>> raw_data;
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        std::vector<int64_t> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            if (!val.empty()) row.push_back(std::stoll(val));
        }
        if (!row.empty()) raw_data.push_back(row);
    }

    int read_rows = raw_data.size();
    int read_cols = raw_data[0].size();
    
    std::vector<std::vector<int64_t>> formatted_W(expected_out, std::vector<int64_t>(expected_in));
    
    if (read_rows == expected_out && read_cols == expected_in) {
        formatted_W = raw_data;
    } else if (read_rows == expected_in && read_cols == expected_out) {
        for (int r = 0; r < read_rows; ++r) {
            for (int c = 0; c < read_cols; ++c) {
                formatted_W[c][r] = raw_data[r][c];
            }
        }
    } else {
        std::cerr << "CRITICAL ERROR: CSV shape " << read_rows << "x" << read_cols 
                  << " does not match expected " << expected_in << "x" << expected_out << std::endl;
    }
    return formatted_W;
}

// ====================================================================
// 2. FBT CONFIGURATION STRUCT & SETUP
// ====================================================================
struct FBTConfig {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    BigInteger Q;       
    BigInteger Bigq;    
    BigInteger PInput;
    BigInteger POutput;
    
    uint32_t numSlotsCKKS; 
    
    size_t order;
    uint32_t scaleTHI;
    std::vector<std::complex<double>> coeffcomp;
};

FBTConfig setup_fbt_environment(uint32_t num_neurons) {
    std::cout << "Setting up FBT Environment (High RAM / Mathematically Secure Mode)..." << std::endl;
    FBTConfig config;
    
    // STRICT FIX: OpenFHE's evaluation tree order must be 1, 2, or 3.
    config.order = 3; 
    config.scaleTHI = 32;
    config.PInput = BigInteger(4096); 
    config.POutput = BigInteger(16);
    config.Q = BigInteger(1) << 48;
    config.Bigq = BigInteger(1) << 40;
    
    config.numSlotsCKKS = 1024;

    auto funcStep = [](int64_t x) -> int64_t {
        return (x < 2048) ? 1 : 0; 
    };

    config.coeffcomp = GetHermiteTrigCoefficients(funcStep, config.PInput.ConvertToInt(), config.order, config.scaleTHI);

    // Standard cryptographic CKKS parameters to prevent modulus mismatch/noise overflow
    uint32_t dcrtBits = 50;
    uint32_t firstMod = 60;
    
    // Balanced Baby-Step Giant-Step tree budget compatible with order=3
    std::vector<uint32_t> lvlb = {3, 3}; 
    SecretKeyDist secretKeyDist = SPARSE_ENCAPSULATED;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(secretKeyDist);
    
    // OpenFHE will now dynamically choose a massive, secure Ring Dimension (e.g., 32768 or 65536)
    parameters.SetSecurityLevel(HEStd_128_classic); 
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetScalingTechnique(FIXEDMANUAL);
    parameters.SetFirstModSize(firstMod);
    parameters.SetNumLargeDigits(3);
    parameters.SetBatchSize(config.numSlotsCKKS); 

    // Calculate exact depth required by FBT
    uint32_t depth = FHECKKSRNS::GetFBTDepth(lvlb, config.coeffcomp, config.PInput, config.order, secretKeyDist);
    
    // Request exact depth needed for the entire network: 2 (Layer 1) + FBT Depth + 2 (Layer 2)
    parameters.SetMultiplicativeDepth(depth + 4);

    config.cc = GenCryptoContext(parameters);
    config.cc->Enable(PKE);
    config.cc->Enable(KEYSWITCH);
    config.cc->Enable(LEVELEDSHE);
    config.cc->Enable(ADVANCEDSHE);
    config.cc->Enable(FHE);

    std::cout << "Generating keys (this will safely consume significant RAM)..." << std::endl;
    config.keyPair = config.cc->KeyGen();
    config.cc->EvalMultKeyGen(config.keyPair.secretKey);
    config.cc->EvalSumKeyGen(config.keyPair.secretKey);

    config.cc->EvalFBTSetup(
        config.coeffcomp, 
        config.numSlotsCKKS, 
        config.PInput,    
        config.POutput,   
        config.Bigq, 
        config.keyPair.publicKey, 
        {0, 0},           
        lvlb,             
        config.scaleTHI, 
        0, 
        config.order
    );

    std::cout << "Generating FBT Evaluation Keys..." << std::endl;
    config.cc->EvalBootstrapKeyGen(config.keyPair.secretKey, config.numSlotsCKKS);
    std::cout << "FBT Environment Setup Complete." << std::endl;

    return config;
}

// ====================================================================
// 3. NEURAL NETWORK LAYERS
// ====================================================================
Ciphertext<DCRTPoly> compute_linear_layer(
    Ciphertext<DCRTPoly> ct_input, 
    const std::vector<std::vector<int64_t>>& W, 
    const std::vector<int64_t>& b,              
    FBTConfig& config,
    int num_neurons) 
{
    Ciphertext<DCRTPoly> ct_layer_out;
    
    for (int i = 0; i < num_neurons; ++i) {
        std::vector<double> w_double(W[i].begin(), W[i].end());
        w_double.resize(config.numSlotsCKKS, 0.0); 
        Plaintext pt_weights = config.cc->MakeCKKSPackedPlaintext(w_double);
        
        // Multiplication + Rescale consumes 1 multiplicative level
        auto ct_mult = config.cc->EvalMult(ct_input, pt_weights);
        auto ct_sum = config.cc->EvalSum(ct_mult, config.numSlotsCKKS); 
        auto ct_rescaled = config.cc->Rescale(ct_sum);
        
        std::vector<double> mask(config.numSlotsCKKS, 0.0);
        mask[i] = 1.0; 
        Plaintext pt_mask = config.cc->MakeCKKSPackedPlaintext(mask, 1, ct_rescaled->GetLevel());
        
        // Masking + Rescale consumes 1 multiplicative level
        auto ct_masked = config.cc->EvalMult(ct_rescaled, pt_mask);
        auto ct_neuron = config.cc->Rescale(ct_masked);
        
        std::vector<double> b_vec(config.numSlotsCKKS, 0.0);
        b_vec[i] = static_cast<double>(b[i]);
        Plaintext pt_bias = config.cc->MakeCKKSPackedPlaintext(b_vec, 1, ct_neuron->GetLevel());
        ct_neuron = config.cc->EvalAdd(ct_neuron, pt_bias);
        
        if (i == 0) {
            ct_layer_out = ct_neuron;
        } else {
            ct_layer_out = config.cc->EvalAdd(ct_layer_out, ct_neuron);
        }
    }
    
    return ct_layer_out;
}

Ciphertext<DCRTPoly> apply_sign_activation_fbt(Ciphertext<DCRTPoly> ct_input, FBTConfig& config) {
    std::cout << "Applying FBT Sign Activation..." << std::endl;

    // We requested `depth + 4` total levels. 
    // Layer 1 consumes exactly 2 levels.
    // FBT consumes `depth` levels. 
    // Therefore, we tell EvalFBT to leave us with exactly 2 excess levels for Layer 2.
    uint32_t excess_levels = 2; 

    return config.cc->EvalFBT(
        ct_input, config.coeffcomp, config.PInput.GetMSB() - 1, 
        config.Bigq, config.scaleTHI, excess_levels, config.order
    );
}

// ====================================================================
// 4. MAIN EXECUTION
// ====================================================================
int main(int argc, char* argv[]) {
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image.png>" << std::endl;
        return 1;
    }
    const char* image_path = argv[1];

    std::cout << "--- DiNN OpenFHE Inference ---" << std::endl;

    std::cout << "Loading weights..." << std::endl;
    auto W1 = load_csv_2d("../dinn30_W1.csv", 784, 30);
    auto b1 = load_csv_1d("../dinn30_b1.csv");
    auto W2 = load_csv_2d("../dinn30_W2.csv", 30, 10);
    auto b2 = load_csv_1d("../dinn30_b2.csv");
    
    FBTConfig config = setup_fbt_environment(30);

    int width, height, channels;
    unsigned char *img_data = stbi_load(image_path, &width, &height, &channels, 1); 
    
    if (img_data == NULL) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return 1;
    }

    std::vector<double> real_image(784, 0.0);
    for(int i = 0; i < 784; ++i) {
        double pixel = img_data[i] / 255.0; 
        // Bipolar data representation
        real_image[i] = (pixel > 0.5) ? 1.0 : -1.0; 
    }
    stbi_image_free(img_data);

    Plaintext pt_image = config.cc->MakeCKKSPackedPlaintext(real_image);
    auto ct_image = config.cc->Encrypt(config.keyPair.publicKey, pt_image);

    std::cout << "Evaluating Layer 1..." << std::endl;
    auto ct_hidden_pre_act = compute_linear_layer(ct_image, W1, b1, config, 30);
    
    auto ct_hidden_post_act = apply_sign_activation_fbt(ct_hidden_pre_act, config);

    std::cout << "Evaluating Layer 2..." << std::endl;
    auto ct_scores = compute_linear_layer(ct_hidden_post_act, W2, b2, config, 10);

    std::cout << "Decrypting..." << std::endl;
    Plaintext pt_result;
    config.cc->Decrypt(config.keyPair.secretKey, ct_scores, &pt_result);
    
    pt_result->SetLength(10);
    auto computed = pt_result->GetRealPackedValue();

    std::cout << "\n--- Final Class Scores ---" << std::endl;
    int predicted_digit = 0;
    double max_score = computed[0];

    for (int i = 0; i < 10; ++i) {
        std::cout << "Digit " << i << ": " << computed[i] << std::endl;
        if (computed[i] > max_score) {
            max_score = computed[i];
            predicted_digit = i;
        }
    }

    std::cout << "\n===============================" << std::endl;
    std::cout << " PREDICTED DIGIT: " << predicted_digit << std::endl;
    std::cout << "===============================" << std::endl;

    return 0;
}
