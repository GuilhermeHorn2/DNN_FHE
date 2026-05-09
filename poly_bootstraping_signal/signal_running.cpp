#include "openfhe.h"
#include "bench/bench.h"

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
#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace lbcrypto;
namespace fs = std::filesystem;

// --- Funções de Memória ---
// Kept around because main() reports the legacy "-> FHE Base Memory" line and
// we want the binary to still print something useful when BENCH_MEMORY is OFF.
// When BENCH_MEMORY is ON, the [BENCH][MEM] markers cover everything these do.
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
    std::cout << "Setting up FHE Environment (Security: HEStd_NotSet, Bootstrap)..." << std::endl;
    FHEConfig config;

    CCParams<CryptoContextCKKSRNS> parameters;

    // CKKS bootstrap REQUIRES a ternary secret key distribution.
    // Without this line, OpenFHE defaults to GAUSSIAN and EvalBootstrapKeyGen
    // produces heap corruption ("munmap_chunk(): invalid pointer").
    SecretKeyDist secretKeyDist = UNIFORM_TERNARY;
    parameters.SetSecretKeyDist(secretKeyDist);

    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(1 << 16);

    ScalingTechnique rescaleTech = FLEXIBLEAUTO;
    uint32_t dcrtBits = 50;
    uint32_t firstMod = 51;
    parameters.SetScalingModSize(dcrtBits);
    parameters.SetFirstModSize(firstMod);
    parameters.SetScalingTechnique(rescaleTech);

    std::vector<uint32_t> levelBudget = {4, 4};
    uint32_t levelsAvailableAfterBootstrap = 7;
    uint32_t depth = levelsAvailableAfterBootstrap +
                     FHECKKSRNS::GetBootstrapDepth(levelBudget, secretKeyDist);
    parameters.SetMultiplicativeDepth(depth);

    uint32_t desired_slots = 1024;
    parameters.SetBatchSize(desired_slots);

    config.cc = GenCryptoContext(parameters);

    config.cc->Enable(PKE);
    config.cc->Enable(KEYSWITCH);
    config.cc->Enable(LEVELEDSHE);
    config.cc->Enable(ADVANCEDSHE);
    config.cc->Enable(FHE);

    uint32_t n = config.cc->GetRingDimension();
    config.numSlotsCKKS = desired_slots;

    std::cout << "Ring Dimension (n): " << n << std::endl;
    std::cout << "Configured Slots: " << config.numSlotsCKKS << std::endl;
    std::cout << "Multiplicative Depth: " << depth << std::endl;

    // EvalBootstrapSetup MUST be called BEFORE KeyGen.
    std::cout << "Setting up Bootstrapping (precomputations)..." << std::endl;
    std::vector<uint32_t> bsgsDim = {0, 0};
    config.cc->EvalBootstrapSetup(levelBudget, bsgsDim, config.numSlotsCKKS);

    std::cout << "Generating keys..." << std::endl;
    config.keyPair = config.cc->KeyGen();
    config.cc->EvalMultKeyGen(config.keyPair.secretKey);
    config.cc->EvalSumKeyGen(config.keyPair.secretKey);

    std::cout << "Generating Bootstrapping Keys..." << std::endl;
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
    uint32_t degree = 3;

    auto tanh_lambda = [](double x) -> double { return std::tanh(x); };

    return config.cc->EvalChebyshevFunction(tanh_lambda, ct_scaled, lowerBound, upperBound, degree);
}

int plaintext_inference(
    const std::vector<double>& image,
    const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
    const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2,
    int hidden_size,
    double& time_taken)
{
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<double> hidden(hidden_size, 0.0);
    for (int i = 0; i < hidden_size; ++i) {
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
        for (int j = 0; j < hidden_size; ++j) {
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

// --- Per-image inference helpers (shared by single-image and folder modes) ---

// Phase timing/memory used to live here as struct fields. They were only used
// for printf, which moved to the bench:: infrastructure (BENCH_LAYER_SCOPE,
// BENCH_MEM, BENCH_INFERENCE_SCOPE). The struct now only carries the
// information the caller actually consumes: the predicted class and the
// raw decrypted scores.
struct InferenceResult {
    int                 fhe_pred;
    std::vector<double> scores;          // 10
};

// Loads `path` as 28x28 grayscale (784 pixels) and binarizes to {-1, +1}
// (Sign-trained network expects bipolar inputs).
// Returns an empty vector on failure so callers can `continue` in folder mode.
std::vector<double> load_image_to_input(const std::string& path) {
    int width, height, channels;
    unsigned char* img_data = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (img_data == NULL) {
        return {};
    }
    std::vector<double> real_image(784, 0.0);
    for (int p = 0; p < 784; ++p) {
        double pixel = img_data[p] / 255.0;
        real_image[p] = (pixel > 0.5) ? 1.0 : -1.0;
    }
    stbi_image_free(img_data);
    return real_image;
}

// Encrypt -> linear1 -> activation -> bootstrap -> linear2 -> decrypt for one
// image. All instrumentation goes through bench:: macros so the per-image
// `[BENCH] <tag> ms (peak RSS …)` lines and `[BENCH][MEM] <tag> …` markers
// match the layout produced by MNIST_signal/main.cpp.
InferenceResult run_fhe_inference(
    const std::vector<double>& real_image,
    const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
    const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2,
    int hidden_size,
    FHEConfig& config)
{
    BENCH_INFERENCE_SCOPE("Inference (Run)");

    InferenceResult r;

    Ciphertext<DCRTPoly> ct_image;
    {
        BENCH_LAYER_SCOPE("InputEncoder");
        Plaintext pt_image = config.cc->MakeCKKSPackedPlaintext(real_image);
        ct_image           = config.cc->Encrypt(config.keyPair.publicKey, pt_image);
        BENCH_MEM("after-encrypt");
    }
    BENCH_MEM("after-input-encode");

    Ciphertext<DCRTPoly> ct_hidden_pre_act;
    {
        BENCH_LAYER_SCOPE("Layer 1");
        ct_hidden_pre_act = compute_linear_layer(ct_image, W1, b1, config, hidden_size);
    }

    Ciphertext<DCRTPoly> ct_bootstrapped;
    {
        BENCH_LAYER_SCOPE("Bootstrap + Activation");
        auto ct_hidden_post_act = apply_approx_activation(ct_hidden_pre_act, config);
        ct_bootstrapped         = config.cc->EvalBootstrap(ct_hidden_post_act);
    }

    Ciphertext<DCRTPoly> ct_scores;
    {
        BENCH_LAYER_SCOPE("Layer 2");
        ct_scores = compute_linear_layer(ct_bootstrapped, W2, b2, config, 10);
    }
    BENCH_MEM("after-layers");

    {
        BENCH_LAYER_SCOPE("OutputDecoder");
        Plaintext pt_result;
        config.cc->Decrypt(config.keyPair.secretKey, ct_scores, &pt_result);
        pt_result->SetLength(10);
        auto computed = pt_result->GetRealPackedValue();
        r.scores.assign(computed.begin(), computed.begin() + 10);
    }
    BENCH_MEM("after-output-decode");

    int    fhe_pred  = 0;
    double max_score = r.scores[0];
    for (int d = 1; d < 10; ++d) {
        if (r.scores[d] > max_score) {
            max_score = r.scores[d];
            fhe_pred  = d;
        }
    }
    r.fhe_pred = fhe_pred;

    return r;
}

// --- Mode runners ---

// Single-image mode: per-image diagnostics. The scope timers in
// run_fhe_inference already produce '[BENCH] InputEncoder/Layer 1/...'
// lines on stdout (when BENCH_*=ON), so this runner only adds the
// human-readable score listing and the FHE-vs-plaintext sanity check.
int run_single_image_mode(
    const std::string& image_path,
    int hidden_size,
    const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
    const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2,
    FHEConfig& config)
{
    std::cout << "Loading image: " << image_path << std::endl;

    auto real_image = load_image_to_input(image_path);
    if (real_image.empty()) {
        std::cerr << "Error loading image." << std::endl;
        return 1;
    }

    double pt_time = 0.0;
    int plaintext_pred = plaintext_inference(
        real_image, W1, b1, W2, b2, hidden_size, pt_time);

    auto r = run_fhe_inference(
        real_image, W1, b1, W2, b2, hidden_size, config);

    std::cout << "\n--- Scores ---\n";
    for (int d = 0; d < 10; ++d) {
        std::cout << "Digit " << d << ": " << r.scores[d] << std::endl;
    }

    std::cout << "\n=========================================\n";
    std::cout << (r.fhe_pred == plaintext_pred ? "MATCH!\n" : "DIVERGENCE!\n");
    std::cout << "Plaintext Prediction: " << plaintext_pred << std::endl;
    std::cout << "FHE Prediction:       " << r.fhe_pred     << std::endl;
    std::cout << "Plaintext Time:       " << pt_time        << " s\n";
    std::cout << "=========================================\n";

    bench::PrintSummary("Single");
    return 0;
}

// Folder mode: iterate <root>/<label>/*.{png,jpg,jpeg}, run FHE inference per
// image, then print the [BENCH SUMMARY Batch] table + accuracy summary +
// confusion matrix. Output layout mirrors io::PrintAccuracySummary / the
// example in Untitled-2 so all driver outputs in this repo line up.
int run_folder_mode(
    const std::string& test_root,
    int hidden_size,
    const std::vector<std::vector<int64_t>>& W1, const std::vector<int64_t>& b1,
    const std::vector<std::vector<int64_t>>& W2, const std::vector<int64_t>& b2,
    FHEConfig& config)
{
    constexpr int OUT_DIM = 10;

    std::cout << "Iterating over " << test_root << ":" << std::endl;

    int total           = 0;
    int correct         = 0;
    int correctPlain    = 0;
    int fheMatchesPlain = 0;
    std::vector<std::vector<int>> confusion(OUT_DIM, std::vector<int>(OUT_DIM, 0));

    for (int label = 0; label < OUT_DIM; ++label) {
        const fs::path classDir = fs::path(test_root) / std::to_string(label);
        if (!fs::exists(classDir)) continue;

        for (const auto& entry : fs::directory_iterator(classDir)) {
            if (!entry.is_regular_file()) continue;
            const auto& path = entry.path();
            const auto  ext  = path.extension().string();
            if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

            auto real_image = load_image_to_input(path.string());
            if (real_image.empty()) continue;

            double pt_time = 0.0;
            int plainPred = plaintext_inference(
                real_image, W1, b1, W2, b2, hidden_size, pt_time);

            auto r = run_fhe_inference(
                real_image, W1, b1, W2, b2, hidden_size, config);

            ++total;
            if (r.fhe_pred == label)     ++correct;
            if (plainPred == label)      ++correctPlain;
            if (r.fhe_pred == plainPred) ++fheMatchesPlain;
            confusion[label][r.fhe_pred] += 1;

            std::printf("[%4d] %s -> pred=%d  plain=%d  truth=%d  %s\n",
                        total, path.filename().c_str(),
                        r.fhe_pred, plainPred, label,
                        r.fhe_pred == label ? "OK" : "MISS");
            std::fflush(stdout);
        }
    }

    bench::PrintSummary("Batch");

    auto pct = [&](int n) {
        return total > 0 ? 100.0 * n / total : 0.0;
    };

    std::printf("\n=== FHE vs PlainText ===\n");
    std::printf("Hit: %d / %d  (%.2f%%)\n", fheMatchesPlain, total, pct(fheMatchesPlain));

    std::printf("\n=== PlainText Acc ===\n");
    std::printf("Correct: %d / %d  (%.2f%%)\n", correctPlain, total, pct(correctPlain));

    std::printf("\n=== Accuracy ===\n");
    std::printf("Correct: %d / %d  (%.2f%%)\n", correct, total, pct(correct));

    std::printf("\nConfusion matrix (row=truth, col=pred):\n     ");
    for (int j = 0; j < OUT_DIM; ++j) std::printf(" %4d", j);
    std::printf("\n");
    for (int i = 0; i < OUT_DIM; ++i) {
        std::printf("  %d :", i);
        for (int j = 0; j < OUT_DIM; ++j) std::printf(" %4d", confusion[i][j]);
        std::printf("\n");
    }

    return 0;
}

int main(int argc, char* argv[]) {
    BENCH_TOTAL_SCOPE("Total program");
    BENCH_MEM("startup");

    if (argc < 2) {
        std::cerr << "Usage:\n";
        std::cerr << "./main <image_or_folder_path> [hidden_size]\n\n";
        std::cerr << "  <image_or_folder_path>:\n";
        std::cerr << "      - a regular file -> single-image mode (verbose output)\n";
        std::cerr << "      - a directory    -> folder mode, expects layout\n";
        std::cerr << "                          <root>/0/*.{png,jpg,jpeg}\n";
        std::cerr << "                          <root>/1/*.{png,jpg,jpeg}\n";
        std::cerr << "                          ...\n";
        std::cerr << "                          <root>/9/*.{png,jpg,jpeg}\n\n";
        std::cerr << "  hidden_size: number of neurons in the hidden layer (default: 30).\n";
        std::cerr << "               Supported values: 30, 100.\n\n";
        std::cerr << "Examples:\n";
        std::cerr << "./main ../img_1.jpg\n";
        std::cerr << "./main ../img_1.jpg 100\n";
        std::cerr << "./main /path/to/test_root\n";
        std::cerr << "./main /path/to/test_root 100\n";
        return 1;
    }

    std::string input_path = argv[1];

    int hidden_size = 30;
    if (argc >= 3) {
        try {
            hidden_size = std::stoi(argv[2]);
        } catch (const std::exception& e) {
            std::cerr << "Error: hidden_size must be an integer (got '"
                      << argv[2] << "')." << std::endl;
            return 1;
        }
    }

    if (hidden_size != 30 && hidden_size != 100) {
        std::cerr << "Error: unsupported hidden_size " << hidden_size
                  << ". Supported values: 30, 100." << std::endl;
        return 1;
    }

    if (!fs::exists(input_path)) {
        std::cerr << "Error: path does not exist -> " << input_path << std::endl;
        return 1;
    }

    const bool folder_mode = fs::is_directory(input_path);
    if (!folder_mode && !fs::is_regular_file(input_path)) {
        std::cerr << "Error: path is neither a regular file nor a directory: "
                  << input_path << std::endl;
        return 1;
    }

    std::cout << "--- DiNN OpenFHE Inference ("
              << (folder_mode ? "Folder Mode" : "Single Image Mode")
              << ") ---" << std::endl;
    std::cout << "Hidden layer size: " << hidden_size << std::endl;

    long mem_before_fhe = get_current_memory_kb();

    std::cout << "Loading weights..." << std::endl;

    const std::string tag       = "signal" + std::to_string(hidden_size);
    const std::string W1_path   = "../" + tag + "_W1.csv";
    const std::string b1_path   = "../" + tag + "_b1.csv";
    const std::string W2_path   = "../" + tag + "_W2.csv";
    const std::string b2_path   = "../" + tag + "_b2.csv";

    auto W1 = load_csv_2d(W1_path, 784, hidden_size);
    auto b1 = load_csv_1d(b1_path);

    auto W2 = load_csv_2d(W2_path, hidden_size, 10);
    auto b2 = load_csv_1d(b2_path);

    FHEConfig config = setup_fhe_environment();
    BENCH_MEM("after-compile");

    long mem_after_fhe_setup = get_current_memory_kb();
    long fhe_base_memory = mem_after_fhe_setup - mem_before_fhe;

    std::cout << "-> FHE Base Memory: "
              << fhe_base_memory / 1024
              << " MB\n" << std::endl;

    int rc = folder_mode
           ? run_folder_mode(input_path, hidden_size, W1, b1, W2, b2, config)
           : run_single_image_mode(input_path, hidden_size, W1, b1, W2, b2, config);

    BENCH_MEM("end");
    return rc;
}
