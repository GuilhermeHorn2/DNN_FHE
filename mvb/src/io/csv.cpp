#include "io/csv.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace io {

std::vector<double> LoadCsv1D(const std::string& path) {
    std::vector<double> v;
    std::ifstream       f(path);
    if (!f) {
        std::cerr << "[ERROR] Cannot open " << path << "\n";
        return v;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) v.push_back(std::stod(line));
    }
    return v;
}

std::vector<std::vector<double>> LoadCsv2D(const std::string& path,
                                           int                expected_in,
                                           int                expected_out) {
    std::vector<std::vector<double>> raw;
    std::ifstream                    f(path);
    if (!f) {
        std::cerr << "[ERROR] Cannot open " << path << "\n";
        return raw;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream   ss(line);
        std::string         tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) row.push_back(std::stod(tok));
        }
        if (!row.empty()) raw.push_back(row);
    }
    if (raw.empty()) return raw;

    const int rr = static_cast<int>(raw.size());
    const int rc = static_cast<int>(raw[0].size());

    std::vector<std::vector<double>> W(expected_out,
                                       std::vector<double>(expected_in, 0.0));

    if (rr == expected_out && rc == expected_in) {
        W = raw;
    } else if (rr == expected_in && rc == expected_out) {
        for (int r = 0; r < rr; ++r)
            for (int c = 0; c < rc; ++c) W[c][r] = raw[r][c];
    } else {
        std::cerr << "[WARNING] " << path << " shape " << rr << "x" << rc
                  << " does not match expected " << expected_out << "x"
                  << expected_in << "\n";
    }
    return W;
}

}  // namespace io
