#ifndef MVB_NETWORK_ACTIVATION_H
#define MVB_NETWORK_ACTIVATION_H

#include "openfhe.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace fhednn {

// A parameterized activation function realized as a Hermite-trigonometric LUT
// over the domain [0, pInput).
//
// Convention (matches the original code):
//   * Before HomDecoding, the network adds `preShift / scaleTHI` to the slot
//     ciphertext. This guarantees the input to `f` is non-negative.
//   * `f` therefore receives values shifted by `preShift`, and must compare
//     against the shifted zero (e.g. sign(x) becomes "x >= preShift ? +1 : -1").
//   * `pInput` defines the LUT period and must be a power of two.
//   * `order` is the Hermite trigonometric order (1 by default, matching the
//     original code).
struct Activation {
    std::string                              name;
    std::function<std::int64_t(std::int64_t)> f;
    std::int64_t                             preShift = 0;
    lbcrypto::BigInteger                     pInput   = lbcrypto::BigInteger(1) << 10;
    std::size_t                              order    = 1;
};

namespace activations {

// f(x) = sign(x - preShift), returning {-1, +1}.
// Default preShift = pInput/2 maps a slot-space value of 0 to the threshold.
Activation Sign(std::int64_t        preShift,
                lbcrypto::BigInteger pInput,
                std::size_t          order = 1);

// f(x) = H(x - preShift), the Heaviside step, returning {0, 1}.
// Like Sign() but with output range {0, 1} instead of {-1, +1}.
Activation Heaviside(std::int64_t        preShift,
                     lbcrypto::BigInteger pInput,
                     std::size_t          order = 1);

// f(x) = max(0, x - preShift), the rectified linear unit.
// `preShift` plays the same role as in Sign/Heaviside: it is the post-shift
// threshold that maps a slot-space value of 0 to the ReLU "knee", so the
// LUT receives values shifted by `preShift` and must subtract it back when
// the input is positive. Output range: [0, pInput - preShift).
Activation ReLU(std::int64_t        preShift,
                lbcrypto::BigInteger pInput,
                std::size_t          order = 1);

// f(x) = x mod pOutput. Used by the InputEncoder to enter slot space.
Activation Identity(lbcrypto::BigInteger pInput,
                    lbcrypto::BigInteger pOutput,
                    std::size_t          order = 1);

// f(x) = (x >= threshold) ? high : low.
Activation Step(std::int64_t        threshold,
                std::int64_t        low,
                std::int64_t        high,
                lbcrypto::BigInteger pInput,
                std::size_t          order = 1);

// Custom activation. Caller supplies the LUT lambda directly.
Activation Custom(std::string                                name,
                  std::function<std::int64_t(std::int64_t)>  f,
                  std::int64_t                               preShift,
                  lbcrypto::BigInteger                       pInput,
                  std::size_t                                order = 1);

}  // namespace activations

}  // namespace fhednn

#endif  // MVB_NETWORK_ACTIVATION_H
