#pragma once

#include <vector>
#include <cstdint>

template<typename S, typename T>
S &operator<<(S& s, const std::vector<T> &v) {
    s << "[";
    for (const auto &vv : v) {
        s << ' ';
        s << vv;
    }
    s << " ]";
    return s;
}

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
