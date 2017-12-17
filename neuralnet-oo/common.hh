/* neuralnet-oo - Object oriented neural network
 * Copyright (C) 2017, Chris Smeele and Jan Halsema.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
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
