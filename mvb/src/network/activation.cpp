#include "network/activation.h"

namespace fhednn {
namespace activations {

Activation Sign(std::int64_t         preShift,
                lbcrypto::BigInteger pInput,
                std::size_t          order) {
    Activation a;
    a.name     = "Sign";
    a.preShift = preShift;
    a.pInput   = pInput;
    a.order    = order;
    const std::int64_t threshold = preShift;
    a.f = [threshold](std::int64_t x) -> std::int64_t {
        return (x >= threshold) ? 1 : -1;
    };
    return a;
}

Activation Identity(lbcrypto::BigInteger pInput,
                    lbcrypto::BigInteger pOutput,
                    std::size_t          order) {
    Activation         a;
    const std::int64_t a_int = pInput.ConvertToInt<std::int64_t>();
    const std::int64_t b_int = pOutput.ConvertToInt<std::int64_t>();
    a.name                   = "Identity";
    a.preShift               = 0;
    a.pInput                 = pInput;
    a.order                  = order;
    a.f = [a_int, b_int](std::int64_t x) -> std::int64_t {
        return (x % a_int) % b_int;
    };
    return a;
}

Activation Step(std::int64_t         threshold,
                std::int64_t         low,
                std::int64_t         high,
                lbcrypto::BigInteger pInput,
                std::size_t          order) {
    Activation a;
    a.name     = "Step";
    a.preShift = threshold;
    a.pInput   = pInput;
    a.order    = order;
    a.f = [threshold, low, high](std::int64_t x) -> std::int64_t {
        return (x >= threshold) ? high : low;
    };
    return a;
}

Activation Custom(std::string                                name,
                  std::function<std::int64_t(std::int64_t)>  f,
                  std::int64_t                               preShift,
                  lbcrypto::BigInteger                       pInput,
                  std::size_t                                order) {
    Activation a;
    a.name     = std::move(name);
    a.f        = std::move(f);
    a.preShift = preShift;
    a.pInput   = pInput;
    a.order    = order;
    return a;
}

}  // namespace activations
}  // namespace fhednn
