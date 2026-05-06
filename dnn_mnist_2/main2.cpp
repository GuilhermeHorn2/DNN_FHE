#include "openfhe.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace lbcrypto;


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


struct FHEConfig {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keyPair;
    uint32_t numSlotsCKKS; 
};

FHEConfig setup_fhe_environment() {
    std::cout << "Setting up FHE Environment (Low RAM / Polynomial Approximation Mode)..." << std::endl;
    FHEConfig config;
    config.numSlotsCKKS = 1024;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_128_classic); 
    
    
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetBatchSize(config.numSlotsCKKS); 

    
    parameters.SetMultiplicativeDepth(12);

    config.cc = GenCryptoContext(parameters);
    config.cc->Enable(PKE);
    config.cc->Enable(KEYSWITCH);
    config.cc->Enable(LEVELEDSHE);
    config.cc->Enable(ADVANCEDSHE);

    std::cout << "Generating keys (Fast & Low RAM)..." << std::endl;
    config.keyPair = config.cc->KeyGen();
    config.cc->EvalMultKeyGen(config.keyPair.secretKey);
    config.cc->EvalSumKeyGen(config.keyPair.secretKey);
    
    std::cout << "FHE Environment Setup Complete." << std::endl;

    return config;
}


Ciphertext<DCRTPoly> compute_linear_layer(
    Ciphertext<DCRTPoly> ct_input, 
    const std::vector<std::vector<int64_t>>& W, 
    const std::vector<int64_t>& b,              
    FHEConfig& config,
    int num_neurons) 
{
    Ciphertext<DCRTPoly> ct_layer_out;
    
    for (int i = 0; i < num_neurons; ++i) {
        std::vector<double> w_double(W[i].begin(), W[i].end());
        w_double.resize(config.numSlotsCKKS, 0.0); 
        Plaintext pt_weights = config.cc->MakeCKKSPackedPlaintext(w_double);
        
        // Level -1
        auto ct_mult = config.cc->EvalMult(ct_input, pt_weights);
        auto ct_sum = config.cc->EvalSum(ct_mult, config.numSlotsCKKS); 
        auto ct_rescaled = config.cc->Rescale(ct_sum);
        
        std::vector<double> mask(config.numSlotsCKKS, 0.0);
        mask[i] = 1.0; 
        Plaintext pt_mask = config.cc->MakeCKKSPackedPlaintext(mask, 1, ct_rescaled->GetLevel());
        
        // Level -2
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

Ciphertext<DCRTPoly> apply_approx_activation(Ciphertext<DCRTPoly> ct_input, FHEConfig& config) {
    std::cout << "Applying Chebyshev Polynomial Approximation (Degree 7)..." << std::endl;
    

    std::vector<double> scale_vec(config.numSlotsCKKS, 0.01);
    Plaintext pt_scale = config.cc->MakeCKKSPackedPlaintext(scale_vec);
    
    auto ct_scaled = config.cc->EvalMult(ct_input, pt_scale);
    ct_scaled = config.cc->Rescale(ct_scaled); // Consumes 1 level

    double lowerBound = -8.0;
    double upperBound = 8.0;
    uint32_t degree = 7;
    
    auto tanh_lambda = [](double x) -> double { return std::tanh(x); };
    
    return config.cc->EvalChebyshevFunction(tanh_lambda, ct_scaled, lowerBound, upperBound, degree);
}


int main(int argc, char* argv[]) {
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image.png>" << std::endl;
        return 1;
    }
    const char* image_path = argv[1];

    std::cout << "--- DiNN OpenFHE Inference (Polynomial Mode) ---" << std::endl;

    std::cout << "Loading weights..." << std::endl;
    auto W1 = load_csv_2d("../signal100_W1.csv", 784, 100);
    auto b1 = load_csv_1d("../signal100_b1.csv");
    auto W2 = load_csv_2d("../signal100_W2.csv", 100, 10);
    auto b2 = load_csv_1d("../signal100_b2.csv");
    
    FHEConfig config = setup_fhe_environment();

    int width, height, channels;
    unsigned char *img_data = stbi_load(image_path, &width, &height, &channels, 1); 
    
    if (img_data == NULL) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return 1;
    }

    std::vector<double> real_image(784, 0.0);
    for(int i = 0; i < 784; ++i) {
        double pixel = img_data[i] / 255.0; 
        real_image[i] = (pixel > 0.5) ? 1.0 : -1.0; 
    }
    stbi_image_free(img_data);

    Plaintext pt_image = config.cc->MakeCKKSPackedPlaintext(real_image);
    auto ct_image = config.cc->Encrypt(config.keyPair.publicKey, pt_image);

    std::cout << "Evaluating Layer 1..." << std::endl;
    auto ct_hidden_pre_act = compute_linear_layer(ct_image, W1, b1, config, 100);
    
    auto ct_hidden_post_act = apply_approx_activation(ct_hidden_pre_act, config);

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