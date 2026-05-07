#include "openfhe.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono> 
#include <sys/resource.h>
#include <filesystem> 
#include <string>
#include <algorithm> 
#include <unistd.h> 

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace lbcrypto;
namespace fs = std::filesystem;

// --- Funções de Memória ---
long get_current_memory_kb() {
    std::ifstream stat_stream("/proc/self/status", std::ios_base::in);
    std::string line;
    long ram = 0;
    while (std::getline(stat_stream, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::istringstream iss(line.substr(6));
            iss >> ram;
            break;
        }
    }
    return ram;
}

long get_peak_memory_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

// --- Funções de Leitura de CSV ---
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

// --- Setup com Bootstrapping Habilitado e Sem Segurança ---
FHEConfig setup_fhe_environment() {
    std::cout << "Setting up FHE Environment (Security: HEStd_NotSet)..." << std::endl;
    FHEConfig config;

    CCParams<CryptoContextCKKSRNS> parameters;
    
    // 1. Desabilita a trava de segurança do OpenFHE
    parameters.SetSecurityLevel(HEStd_NotSet); 
    
    // 2. Força um anel muito menor (16384 ou 32768) para economizar RAM. 
    // Como a profundidade é 17, 32768 é o mais seguro para não dar erro matemático interno,
    // mas 16384 pode rodar dependendo dos tamanhos exatos dos módulos.
    parameters.SetRingDim(32768); 
    
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(17);
    
    // 3. Limita o número de slots gerados para poupar RAM
    uint32_t desired_slots = 1024;
    parameters.SetBatchSize(desired_slots); 

    config.cc = GenCryptoContext(parameters);
    
    uint32_t n = config.cc->GetRingDimension();
    config.numSlotsCKKS = desired_slots; 
    
    std::cout << "Ring Dimension (n): " << n << std::endl;
    std::cout << "Configured Slots: " << config.numSlotsCKKS << std::endl;

    config.cc->Enable(PKE);
    config.cc->Enable(KEYSWITCH);
    config.cc->Enable(LEVELEDSHE);
    config.cc->Enable(ADVANCEDSHE);
    config.cc->Enable(FHE); 

    std::cout << "Generating keys..." << std::endl;
    config.keyPair = config.cc->KeyGen();
    config.cc->EvalMultKeyGen(config.keyPair.secretKey);
    config.cc->EvalSumKeyGen(config.keyPair.secretKey);
    
    std::cout << "Setting up Bootstrapping Keys..." << std::endl;
    std::vector<uint32_t> levelBudget = {4, 4};
    std::vector<uint32_t> bsgsDim = {0, 0};
    
    config.cc->EvalBootstrapSetup(levelBudget, bsgsDim, config.numSlotsCKKS);
    config.cc->EvalBootstrapKeyGen(config.keyPair.secretKey, config.numSlotsCKKS);
    std::cout << "Bootstrapping keys generated successfully." << std::endl;

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
        
        auto ct_mult = config.cc->EvalMult(ct_input, pt_weights);
        auto ct_sum = config.cc->EvalSum(ct_mult, config.numSlotsCKKS); 
        auto ct_rescaled = config.cc->Rescale(ct_sum);
        
        std::vector<double> mask(config.numSlotsCKKS, 0.0);
        mask[i] = 1.0; 
        Plaintext pt_mask = config.cc->MakeCKKSPackedPlaintext(mask, 1, ct_rescaled->GetLevel());
        
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
    std::vector<double> scale_vec(config.numSlotsCKKS, 0.01);
    Plaintext pt_scale = config.cc->MakeCKKSPackedPlaintext(scale_vec);
    
    auto ct_scaled = config.cc->EvalMult(ct_input, pt_scale);
    ct_scaled = config.cc->Rescale(ct_scaled); 

    double lowerBound = -8.0;
    double upperBound = 8.0;
    uint32_t degree = 7;
    
    auto tanh_lambda = [](double x) -> double { return std::tanh(x); };
    
    return config.cc->EvalChebyshevFunction(tanh_lambda, ct_scaled, lowerBound, upperBound, degree);
}

int plaintext_inference(
    const std::vector<double>& image, 
    const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
    const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2,
    double& time_taken) 
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> hidden(30, 0.0);
    for (int i = 0; i < 30; ++i) {
        double sum = static_cast<double>(b1[i]);
        for (int j = 0; j < 784; ++j) {
            sum += W1[i][j] * image[j];
        }
        hidden[i] = std::tanh(sum * 0.01); 
    }

    std::vector<double> output(10, 0.0);
    int predicted_class = 0;
    double max_score = -1e9;

    for (int i = 0; i < 10; ++i) {
        double sum = static_cast<double>(b2[i]);
        for (int j = 0; j < 30; ++j) {
            sum += W2[i][j] * hidden[j];
        }
        output[i] = sum;
        
        if (sum > max_score) {
            max_score = sum;
            predicted_class = i;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    time_taken = duration.count();

    return predicted_class;
}

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "./dinn_inference <image_path>\n\n";
        std::cerr << "Example:\n";
        std::cerr << "./dinn_inference /home/horn/Downloads/dnn_crypto/mnist/testSet/3.png\n";
        return 1;
    }

    std::string image_path = argv[1];

    if (!fs::exists(image_path)) {
        std::cerr << "Error: Image does not exist -> " << image_path << std::endl;
        return 1;
    }

    std::cout << "--- DiNN OpenFHE Inference (Single Image Mode) ---" << std::endl;

    long mem_before_fhe = get_current_memory_kb();

    std::cout << "Loading weights..." << std::endl;

    auto W1 = load_csv_2d("../dinn30_W1.csv", 784, 30);
    auto b1 = load_csv_1d("../dinn30_b1.csv");

    auto W2 = load_csv_2d("../dinn30_W2.csv", 30, 10);
    auto b2 = load_csv_1d("../dinn30_b2.csv");

    FHEConfig config = setup_fhe_environment();

    long mem_after_fhe_setup = get_current_memory_kb();
    long fhe_base_memory = mem_after_fhe_setup - mem_before_fhe;

    std::cout << "-> FHE Base Memory: "
              << fhe_base_memory / 1024
              << " MB\n" << std::endl;

    std::cout << "Loading image: " << image_path << std::endl;

    int width, height, channels;

    unsigned char* img_data =
        stbi_load(image_path.c_str(),
                  &width,
                  &height,
                  &channels,
                  1);

    if (img_data == NULL) {
        std::cerr << "Error loading image." << std::endl;
        return 1;
    }

    std::vector<double> real_image(784, 0.0);

    for (int p = 0; p < 784; ++p) {
        double pixel = img_data[p] / 255.0;
        real_image[p] = (pixel > 0.5) ? 1.0 : -1.0;
    }

    stbi_image_free(img_data);

    double pt_time = 0.0;

    int plaintext_pred = plaintext_inference(
        real_image,
        W1,
        b1,
        W2,
        b2,
        pt_time
    );

    auto start_inference = std::chrono::high_resolution_clock::now();

    Plaintext pt_image =
        config.cc->MakeCKKSPackedPlaintext(real_image);

    auto ct_image =
        config.cc->Encrypt(config.keyPair.publicKey, pt_image);

    long mem_before_inference = get_current_memory_kb();

    auto start_eval = std::chrono::high_resolution_clock::now();

    auto ct_hidden_pre_act =
        compute_linear_layer(
            ct_image,
            W1,
            b1,
            config,
            30
        );

    auto start_act_bs =
        std::chrono::high_resolution_clock::now();

    auto ct_hidden_post_act =
        apply_approx_activation(
            ct_hidden_pre_act,
            config
        );

    auto ct_bootstrapped =
        config.cc->EvalBootstrap(ct_hidden_post_act);

    auto end_act_bs =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> act_bs_duration =
        end_act_bs - start_act_bs;

    auto ct_scores =
        compute_linear_layer(
            ct_bootstrapped,
            W2,
            b2,
            config,
            10
        );

    auto end_eval =
        std::chrono::high_resolution_clock::now();

    long current_peak = get_peak_memory_kb();
    long active_inference_cost =
        current_peak - mem_after_fhe_setup;

    Plaintext pt_result;

    config.cc->Decrypt(
        config.keyPair.secretKey,
        ct_scores,
        &pt_result
    );

    pt_result->SetLength(10);

    auto computed =
        pt_result->GetRealPackedValue();

    auto end_inference =
        std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> eval_duration =
        end_eval - start_eval;

    std::chrono::duration<double> total_duration =
        end_inference - start_inference;

    int fhe_pred = 0;
    double max_score = computed[0];

    std::cout << "\n--- Scores ---\n";

    for (int d = 0; d < 10; ++d) {

        std::cout << "Digit "
                  << d
                  << ": "
                  << computed[d]
                  << std::endl;

        if (computed[d] > max_score) {
            max_score = computed[d];
            fhe_pred = d;
        }
    }

    std::cout << "\n=========================================\n";

    if (fhe_pred == plaintext_pred) {
        std::cout << "MATCH!\n";
    } else {
        std::cout << "DIVERGENCE!\n";
    }

    std::cout << "Plaintext Prediction: "
              << plaintext_pred
              << std::endl;

    std::cout << "FHE Prediction: "
              << fhe_pred
              << std::endl;

    std::cout << "\n--- Timing ---\n";

    std::cout << "Plaintext Time: "
              << pt_time
              << " s\n";

    std::cout << "FHE Eval Time: "
              << eval_duration.count()
              << " s\n";

    std::cout << "Activation + Bootstrap: "
              << act_bs_duration.count()
              << " s\n";

    std::cout << "Total FHE Time: "
              << total_duration.count()
              << " s\n";

    std::cout << "\n--- Memory ---\n";

    std::cout << "FHE Base Memory: ~"
              << fhe_base_memory / 1024
              << " MB\n";

    std::cout << "Inference Memory: ~"
              << active_inference_cost / 1024
              << " MB\n";

    std::cout << "Global Peak Memory: "
              << get_peak_memory_kb() / 1024
              << " MB\n";

    std::cout << "=========================================\n";

    return 0;
}
