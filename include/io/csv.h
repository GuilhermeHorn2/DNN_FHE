#ifndef MVB_IO_CSV_H
#define MVB_IO_CSV_H

#include <string>
#include <vector>

namespace io {

// Reads one floating-point value per non-empty line.
std::vector<double> LoadCsv1D(const std::string& path);

// Reads a comma-separated 2-D matrix and reshapes it to (out_rows x in_cols).
// Accepts files in either orientation: (out x in) or (in x out); transposes
// automatically if needed. Logs a warning if the file shape does not match.
std::vector<std::vector<double>> LoadCsv2D(const std::string& path,
                                           int                expected_in,
                                           int                expected_out);

}  // namespace io

#endif  // MVB_IO_CSV_H
