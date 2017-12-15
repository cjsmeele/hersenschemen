#pragma once

#define NDEBUG 1
#define MATRIX_WANT_STREAMOPS 1

#include <cstdint>
#include <cmath>

// Uncomment to round all floats before printing.
//
// template<typename S>
// S &operator<<(S& s, const std::vector<double> &v) {
//     s << "[";
//     for (const auto &vv : v) {
//         s << ' ';
//         s << round(vv);
//     }
//     s << " ]";
//     return s;
// }
