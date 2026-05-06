#include "openfhe.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <climits>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace lbcrypto;

// CIFAR-10 DiNN: 32 x 32 x 3 RGB → flat 3072 → 30 → 10 (same topology as DiNN30,
// just a much fatter first-layer matrix).
static constexpr int IN_DIM  = 3072;
static constexpr int HID_DIM = 30;
static constexpr int OUT_DIM = 10;


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
    std::cout << "Setting up FHE Environment (Polynomial Approximation Mode, CIFAR-sized)..." << std::endl;
    FHEConfig config;
    // 3072 inputs need at least 3072 slots; round up to the next power of two.
    // The ring dim is then forced to >= 2 * numSlots = 8192 by HEStd_128_classic.
    config.numSlotsCKKS = 4096;

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

    std::cout << "  numSlots = " << config.numSlotsCKKS
              << ", ringDim = " << config.cc->GetRingDimension() << std::endl;

    std::cout << "Generating keys..." << std::endl;
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

Ciphertext<DCRTPoly> apply_approx_activation(Ciphertext<DCRTPoly> ct_input, FHEConfig& config, double pre_scale) {
    std::cout << "Applying Chebyshev Polynomial Approximation (Degree 7), pre_scale = "
              << pre_scale << "..." << std::endl;
    
    std::vector<double> scale_vec(config.numSlotsCKKS, pre_scale);
    Plaintext pt_scale = config.cc->MakeCKKSPackedPlaintext(scale_vec);
    
    auto ct_scaled = config.cc->EvalMult(ct_input, pt_scale);
    ct_scaled = config.cc->Rescale(ct_scaled); // Consumes 1 level

    double lowerBound = -8.0;
    double upperBound = 8.0;
    uint32_t degree = 7;
    
    auto tanh_lambda = [](double x) -> double { return std::tanh(x); };
    
    return config.cc->EvalChebyshevFunction(tanh_lambda, ct_scaled, lowerBound, upperBound, degree);
}

// Pick a Chebyshev pre-scale so that the worst-case |layer-1 pre-activation|
// maps into roughly half of [-8, 8]. Worst case happens when every input
// pixel aligns sign-wise with W1[j][i], giving sum_i |W1[j][i]| + |b1[j]|.
double compute_pre_scale(const std::vector<std::vector<int64_t>>& W1,
                         const std::vector<int64_t>& b1) {
    int64_t max_abs_pre = 1;
    for (size_t j = 0; j < W1.size(); ++j) {
        int64_t row_l1 = std::abs(b1[j]);
        for (auto w : W1[j]) row_l1 += std::abs(w);
        if (row_l1 > max_abs_pre) max_abs_pre = row_l1;
    }
    // 4.0 (not 8.0) leaves headroom so Chebyshev edge effects don't dominate.
    return 4.0 / static_cast<double>(max_abs_pre);
}


int main(int argc, char* argv[]) {
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_image.png>" << std::endl;
        return 1;
    }
    const char* image_path = argv[1];

    std::cout << "--- CIFAR-10 OpenFHE Inference (Polynomial Mode) ---" << std::endl;

    std::cout << "Loading weights..." << std::endl;
    auto W1 = load_csv_2d("../signal30_W1.csv", IN_DIM,  HID_DIM);
    auto b1 = load_csv_1d("../signal30_b1.csv");
    auto W2 = load_csv_2d("../signal30_W2.csv", HID_DIM, OUT_DIM);
    auto b2 = load_csv_1d("../signal30_b2.csv");

    const double cheb_pre_scale = compute_pre_scale(W1, b1);
    std::cout << "Auto-tuned Chebyshev pre-scale = " << cheb_pre_scale
              << "  (1 / " << (1.0 / cheb_pre_scale) << ")" << std::endl;

    FHEConfig config = setup_fhe_environment();

    int width, height, channels;
    // CIFAR PNGs are 3-channel RGB; force stb_image to deliver 3072 = 32*32*3 bytes.
    unsigned char *img_data = stbi_load(image_path, &width, &height, &channels, 3); 
    
    if (img_data == NULL) {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return 1;
    }

    std::vector<double> real_image(IN_DIM, 0.0);
    for (int i = 0; i < IN_DIM; ++i) {
        double pixel = img_data[i] / 255.0; 
        real_image[i] = (pixel > 0.5) ? 1.0 : -1.0; 
    }
    stbi_image_free(img_data);

    Plaintext pt_image = config.cc->MakeCKKSPackedPlaintext(real_image);
    auto ct_image = config.cc->Encrypt(config.keyPair.publicKey, pt_image);

    std::cout << "Evaluating Layer 1..." << std::endl;
    auto ct_hidden_pre_act = compute_linear_layer(ct_image, W1, b1, config, HID_DIM);
    
    auto ct_hidden_post_act = apply_approx_activation(ct_hidden_pre_act, config, cheb_pre_scale);

    std::cout << "Evaluating Layer 2..." << std::endl;
    auto ct_scores = compute_linear_layer(ct_hidden_post_act, W2, b2, config, OUT_DIM);

    std::cout << "Decrypting..." << std::endl;
    Plaintext pt_result;
    config.cc->Decrypt(config.keyPair.secretKey, ct_scores, &pt_result);
    
    pt_result->SetLength(OUT_DIM);
    auto computed = pt_result->GetRealPackedValue();

    std::cout << "\n--- Final Class Scores ---" << std::endl;
    int predicted_class = 0;
    double max_score = computed[0];

    for (int i = 0; i < OUT_DIM; ++i) {
        std::cout << "Class " << i << ": " << computed[i] << std::endl;
        if (computed[i] > max_score) {
            max_score = computed[i];
            predicted_class = i;
        }
    }

    // ── Plaintext reference (sign activation, exact) ──────────────────────
    std::vector<double> hidden(HID_DIM, 0.0);
    for (int j = 0; j < HID_DIM; ++j) {
        double acc = static_cast<double>(b1[j]);
        for (int i = 0; i < IN_DIM; ++i) acc += static_cast<double>(W1[j][i]) * real_image[i];
        hidden[j] = (acc >= 0.0) ? 1.0 : -1.0;
    }
    std::vector<double> ref_scores(OUT_DIM, 0.0);
    for (int j = 0; j < OUT_DIM; ++j) {
        ref_scores[j] = static_cast<double>(b2[j]);
        for (int i = 0; i < HID_DIM; ++i) ref_scores[j] += static_cast<double>(W2[j][i]) * hidden[i];
    }
    const int ref_pred = static_cast<int>(
        std::max_element(ref_scores.begin(), ref_scores.end()) - ref_scores.begin());

    std::cout << "\n===============================" << std::endl;
    std::cout << " PREDICTED CLASS  : " << predicted_class << std::endl;
    std::cout << " REFERENCE (plain): " << ref_pred
              << (predicted_class == ref_pred ? "  (matches)" : "  (MISMATCH)")
              << std::endl;
    std::cout << "===============================" << std::endl;

    return 0;
}
