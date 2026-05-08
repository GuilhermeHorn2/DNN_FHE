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
#include <unistd.h> // Para sysconf

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace lbcrypto;
namespace fs = std::filesystem;

// --- Função para ler a Memória RAM Atual (VmRSS) no Linux ---
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

// --- Função para ler o Pico de Memória (ru_maxrss) ---
long get_peak_memory_kb() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

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

// ================================================================
// HELPERS
// ================================================================

// Proxima potencia de 2 >= x
static int next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// Calcula todos os índices de rotação necessários para a camada

std::vector<int32_t> simd_layer_rotation_indices(
    int input_size, int num_neurons, int num_slots)
{
    std::set<int32_t> s;
    const int B  = next_pow2(input_size);
    const int Nc = num_slots / B;
    const int Nb = (num_neurons + Nc - 1) / Nc;

    // (A) Soma em arvore: +1, +2, +4 … +B/2  →  log2(B) chaves
    for (int r = 1; r < B; r <<= 1) s.insert(r);

    // (B) Replicacao: -B, -2B, -4B …  →  log2(numSlots/B) chaves
    for (int r = B; r < num_slots; r <<= 1) s.insert(-r);

    // (C) Compactação BSGS: O(sqrt(Nc)) chaves em vez de O(Nc)
    
    {
        const int max_neurons_per_batch = std::min(Nc, num_neurons);
        const int g = (int)std::ceil(std::sqrt((double)max_neurons_per_batch));
        const int step = B - 1; // = 1023 para B=1024

        // Baby steps: 1*step, 2*step, ..., (g-1)*step
        for (int j = 1; j < g; ++j)
            s.insert(j * step);

        // Giant steps: g*step, 2g*step, 3g*step, ...
        for (int k = 1; k * g < max_neurons_per_batch; ++k)
            s.insert(k * g * step);
    }

    // (D) Posicionamento de batch: -(k*Nc) para k=1..Nb-1
    for (int k = 1; k < Nb; ++k) s.insert(-(k * Nc));

    return {s.begin(), s.end()};
}

// ================================================================
// SETUP — recebe a arquitetura da rede para pré-gerar as chaves
//         de rotação exatas de cada camada linear SIMD.
// ================================================================
FHEConfig setup_fhe_environment(
    int l1_in, int l1_out,   // ex: 784, 30
    int l2_in, int l2_out)   // ex:  30, 10
{
    std::cout << "Setting up FHE Environment (Max Slots Mode)..." << std::endl;
    FHEConfig config;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecurityLevel(HEStd_128_classic);
    parameters.SetScalingModSize(50);
    parameters.SetFirstModSize(60);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetMultiplicativeDepth(12);
    parameters.SetBatchSize(0); // Max slots

    config.cc = GenCryptoContext(parameters);
    uint32_t n = config.cc->GetRingDimension();
    config.numSlotsCKKS = n / 2;

    std::cout << "Ring Dimension: " << n
              << " | Slots: " << config.numSlotsCKKS << "\n";

    config.cc->Enable(PKE);
    config.cc->Enable(KEYSWITCH);
    config.cc->Enable(LEVELEDSHE);
    config.cc->Enable(ADVANCEDSHE);

    std::cout << "Generating keys..." << std::endl;
    config.keyPair = config.cc->KeyGen();
    config.cc->EvalMultKeyGen(config.keyPair.secretKey);
    config.cc->EvalSumKeyGen(config.keyPair.secretKey);

    // Unifica os indices necessários para AMBAS as camadas da rede
    std::set<int32_t> rot_set;
    for (auto r : simd_layer_rotation_indices(l1_in, l1_out, config.numSlotsCKKS))
        rot_set.insert(r);
    /*for (auto r : simd_layer_rotation_indices(l2_in, l2_out, config.numSlotsCKKS))
        rot_set.insert(r);*/

    std::vector<int32_t> rot_vec(rot_set.begin(), rot_set.end());
    config.cc->EvalRotateKeyGen(config.keyPair.secretKey, rot_vec);

    std::cout << "Total de chaves de rotação geradas: " << rot_vec.size() << "\n";
    return config;
}

// ================================================================
// COMPUTE_LINEAR_LAYER_SIMD — versão genérica e com batching
// ================================================================
Ciphertext<DCRTPoly> compute_linear_layer_simd(
    Ciphertext<DCRTPoly> ct_input,
    const std::vector<std::vector<int64_t>>& W,
    const std::vector<int64_t>& b,
    FHEConfig& config,
    int num_neurons,
    int input_size)           
{

    const int B  = next_pow2(input_size);                   
    const int Nc = (int)config.numSlotsCKKS / B;            
    const int Nb = (num_neurons + Nc - 1) / Nc;             

    if (B > (int)config.numSlotsCKKS)
        throw std::runtime_error(
            "input_size ultrapassa o número de slots: impossível usar SIMD.");

    std::cout << "[SIMD Layer] input=" << input_size
              << " | block=" << B
              << " | neurons_per_batch=" << Nc
              << " | num_batches=" << Nb << "\n";


    auto ct_rep = ct_input;
    for (int stride = B; stride < (int)config.numSlotsCKKS; stride <<= 1)
        ct_rep = config.cc->EvalAdd(
            ct_rep, config.cc->EvalRotate(ct_rep, -stride));


    Ciphertext<DCRTPoly> ct_out;

    for (int batch = 0; batch < Nb; ++batch) {

        const int n0 = batch * Nc;                           // primeiro neuronio
        const int n1 = std::min(n0 + Nc, num_neurons);      // último + 1
        const int Bs = n1 - n0;                              // tamanho real do batch

        // ---- Passo 2: Aglomerar pesos ---------------------------------
        std::vector<double> flat_W(config.numSlotsCKKS, 0.0);
        for (int i = 0; i < Bs; ++i)
            for (int j = 0; j < input_size; ++j)
                flat_W[i * B + j] = static_cast<double>(W[n0 + i][j]);

        auto pt_W  = config.cc->MakeCKKSPackedPlaintext(flat_W);
        auto ct_mw = config.cc->EvalMult(ct_rep, pt_W);

        // ---- Passo 3: Soma em arvore (dentro de cada bloco) ----------
        auto ct_sum = ct_mw;
        for (int r = 1; r < B; r <<= 1)
            ct_sum = config.cc->EvalAdd(
                ct_sum, config.cc->EvalRotate(ct_sum, r));
        auto ct_rs = config.cc->Rescale(ct_sum);

        // ---- Passo 4: Adicao do bias ----------------------------------
        std::vector<double> bvec(config.numSlotsCKKS, 0.0);
        for (int i = 0; i < Bs; ++i)
            bvec[i * B] = static_cast<double>(b[n0 + i]);

        auto pt_b  = config.cc->MakeCKKSPackedPlaintext(bvec, 1, ct_rs->GetLevel());
        auto ct_sp = config.cc->EvalAdd(ct_rs, pt_b);

 
        {
            const int step = B - 1;                          // 1023 para B=1024
            const int g    = (int)std::ceil(std::sqrt((double)Bs));
            Ciphertext<DCRTPoly> ct_cmp;
            bool first = true;
            
            for (int kg = 0; kg * g < Bs; ++kg) {
                // Rotação giant: kg * g * step (0 no primeiro grupo)
                Ciphertext<DCRTPoly> ct_giant = ct_sp;
                if (kg > 0)
                    ct_giant = config.cc->EvalRotate(ct_giant, kg * g * step);
                    
                // Dentro deste grupo: aplica baby step para cada neuronio
                for (int jb = 0; jb < g; ++jb) {
                    int i = kg * g + jb;        
                    if (i >= Bs) break;
                    
                    // Isola slot[i*B] 
                    std::vector<double> mask(config.numSlotsCKKS, 0.0);
                    mask[i * B] = 1.0;
                    auto pt_mask = config.cc->MakeCKKSPackedPlaintext(
                                       mask, 1, ct_sp->GetLevel());
                                       
                    auto ct_m = config.cc->Rescale(
                                    config.cc->EvalMult(ct_sp, pt_mask));
                                    
                    // Rotação baby: jb * step
                    if (kg > 0)
                        ct_m = config.cc->EvalRotate(ct_m, kg * g * step);
                    if (jb > 0)
                        ct_m = config.cc->EvalRotate(ct_m, jb * step);
                        
                    ct_cmp = first ? ct_m : config.cc->EvalAdd(ct_cmp, ct_m);
                    first = false;
                }
            }
            
            // ct_cmp agora tem resultados em slots 0..Bs-1
            
            // ---- Passo 6: Posicionamento do batch ----------------------
            if (batch > 0)
                ct_cmp = config.cc->EvalRotate(ct_cmp, -(batch * Nc));
                
            // ---- Passo 7: Acumular todos os batches ----------------------    
            ct_out = (batch == 0) ? ct_cmp : config.cc->EvalAdd(ct_out, ct_cmp);
        }
    }

    return ct_out;
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
    std::string directory_path = "/home/horn/Downloads/dnn_crypto/mnist/testSet";
    int n_images_to_test = 20;

    if (argc >= 2) n_images_to_test = std::stoi(argv[1]);
    if (argc >= 3) directory_path = argv[2];

    if (!fs::exists(directory_path) || !fs::is_directory(directory_path)) {
        std::cerr << "Erro: Diretorio invalido -> " << directory_path << std::endl;
        return 1;
    }

    std::cout << "--- DiNN OpenFHE Inference (FHE vs Plaintext Mode) ---" << std::endl;
    
    // [MEMÓRIA] Medindo RAM antes de carregar o OpenFHE
    long mem_before_fhe = get_current_memory_kb();

    std::cout << "Loading weights..." << std::endl;
    auto W1 = load_csv_2d("../dinn30_W1.csv", 784, 30);
    auto b1 = load_csv_1d("../dinn30_b1.csv");
    auto W2 = load_csv_2d("../dinn30_W2.csv", 30, 10);
    auto b2 = load_csv_1d("../dinn30_b2.csv");
    
    ///////
    //FHEConfig config = setup_fhe_environment();
    FHEConfig config = setup_fhe_environment(784, 30, 30, 10);
    ///////
    long mem_after_fhe_setup = get_current_memory_kb();
    long fhe_base_memory = mem_after_fhe_setup - mem_before_fhe;

    std::cout << "-> Memória Base FHE (Chaves + Contexto): " << fhe_base_memory / 1024 << " MB\n" << std::endl;

    std::vector<std::string> valid_files;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find(".png") != std::string::npos || filename.find(".jpg") != std::string::npos) {
                valid_files.push_back(entry.path().string());
            }
        }
    }
    
    std::sort(valid_files.begin(), valid_files.end());
    int files_to_process = std::min(n_images_to_test, static_cast<int>(valid_files.size()));

    int matches_with_plaintext = 0;
    double total_fhe_eval_time = 0.0;
    double total_fhe_end_to_end_time = 0.0;
    double total_plaintext_time = 0.0;
    
    long total_inference_memory_kb = 0; 
    for (int i = 0; i < files_to_process; ++i) {
        std::string filepath = valid_files[i];
        std::string filename = fs::path(filepath).filename().string();

        std::cout << "[" << i + 1 << "/" << files_to_process << "] Processando: " << filename << std::endl;

        int width, height, channels;
        unsigned char *img_data = stbi_load(filepath.c_str(), &width, &height, &channels, 1); 
        
        if (img_data == NULL) continue;

        std::vector<double> real_image(784, 0.0);
        for(int p = 0; p < 784; ++p) {
            double pixel = img_data[p] / 255.0; 
            real_image[p] = (pixel > 0.5) ? 1.0 : -1.0; 
        }
        stbi_image_free(img_data);

        double pt_time = 0.0;
        int plaintext_pred = plaintext_inference(real_image, W1, b1, W2, b2, pt_time);
        total_plaintext_time += pt_time;

        auto start_inference = std::chrono::high_resolution_clock::now();

        Plaintext pt_image = config.cc->MakeCKKSPackedPlaintext(real_image);
        auto ct_image = config.cc->Encrypt(config.keyPair.publicKey, pt_image);

       
        long mem_before_inference = get_current_memory_kb();

        auto start_eval = std::chrono::high_resolution_clock::now();

        // 1
        //auto ct_hidden_pre_act = compute_linear_layer_simd(ct_image, W1, b1, config, 30);
        auto ct_hidden_pre_act = compute_linear_layer_simd(
            ct_image, W1, b1, config, 30, 784);
        ////////////////

        // 2
        auto ct_hidden_post_act = apply_approx_activation(ct_hidden_pre_act, config);
        
        // 3
        auto ct_scores = compute_linear_layer(ct_hidden_post_act, W2, b2, config, 10);
        /*auto ct_scores = compute_linear_layer_simd(
            ct_hidden_post_act, W2, b2, config, 10, 30);*/
        ////////////////////////////////////////////////////////////

        auto end_eval = std::chrono::high_resolution_clock::now();
    
        long mem_after_inference = get_current_memory_kb();
        long current_inference_mem = mem_after_inference - mem_before_inference;
        if(current_inference_mem < 0) current_inference_mem = 0; // Fallback de segurança 
        
     
        long current_peak = get_peak_memory_kb();
        long active_inference_cost = current_peak - mem_after_fhe_setup;

        total_inference_memory_kb += active_inference_cost;

        std::chrono::duration<double> eval_duration = end_eval - start_eval;
        total_fhe_eval_time += eval_duration.count();

        Plaintext pt_result;
        config.cc->Decrypt(config.keyPair.secretKey, ct_scores, &pt_result);
        pt_result->SetLength(10);
        auto computed = pt_result->GetRealPackedValue();

        auto end_inference = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> inference_duration = end_inference - start_inference;
        total_fhe_end_to_end_time += inference_duration.count();

        int fhe_pred = 0;
        double max_score = computed[0];

        for (int d = 0; d < 10; ++d) {
            if (computed[d] > max_score) {
                max_score = computed[d];
                fhe_pred = d;
            }
        }

        if (fhe_pred == plaintext_pred) {
            matches_with_plaintext++;
            std::cout << " -> OK! Predição: " << fhe_pred;
        } else {
            std::cout << " -> DIVERGÊNCIA! FHE: " << fhe_pred << " | PT: " << plaintext_pred;
        }
        std::cout << " | Tempo Eval: " << eval_duration.count() << "s | RAM da Inferencia: " << active_inference_cost / 1024 << " MB\n";
    }

    std::cout << "\n=========================================" << std::endl;
    std::cout << "             RELATÓRIO FINAL             " << std::endl;
    std::cout << "=========================================" << std::endl;
    
    if (files_to_process > 0) {
        double fidelity = (static_cast<double>(matches_with_plaintext) / files_to_process) * 100.0;
        std::cout << "Fidelidade FHE vs Plaintext: " << fidelity << "% (" << matches_with_plaintext << "/" << files_to_process << ")" << std::endl;
        
        std::cout << "\n--- Tempos Médios ---" << std::endl;
        std::cout << "Plaintext (CPU): " << (total_plaintext_time / files_to_process) << " s" << std::endl;
        std::cout << "FHE (Eval Only): " << (total_fhe_eval_time / files_to_process) << " s" << std::endl;
        std::cout << "FHE (I/O + Cripto): " << (total_fhe_end_to_end_time / files_to_process) << " s" << std::endl;

        std::cout << "\n--- Consumo de Memória ---" << std::endl;
        std::cout << "Ambiente FHE (Chaves): ~" << fhe_base_memory / 1024 << " MB (Permanente)" << std::endl;
        std::cout << "Custo por Inferência: ~" << (total_inference_memory_kb / files_to_process) / 1024 << " MB (Dinâmico)" << std::endl;
        std::cout << "Pico Global do Processo: " << get_peak_memory_kb() / 1024 << " MB" << std::endl;
    }
    std::cout << "=========================================" << std::endl;

    return 0;
}
