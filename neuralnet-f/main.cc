/* neuralnet-f - Algebraic (vectorized) neural network
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
#include "common.hh"
#include "nn.hh"
#include "matrix.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <tuple>

int main() {
    srand(time(NULL));
    std::cout.precision(2);

    Matrixd<4,3> X {
            0, 0, 1,
            0, 1, 1,
            1, 0, 1,
            1, 1, 1
    };

    Matrixd<4,1> y {
            0,
            1,
            1,
            0
    };

    auto net = nn::make_net<double,3,1,1,4>{};
    char *name = abi::__cxa_demangle(typeid(net).name(),
                                     nullptr,
                                     nullptr,
                                     nullptr);
    std::cout << "🦄🦄🦄🦄🦄🦄🦄: " << name << "\n";


    auto rd = [](auto) {return (double)rand()/RAND_MAX*2 - 1;};
    std::apply([&rd](auto& ...x){(x.mip(rd) , ...);}, net);
    //auto net = std::apply([&](auto... x){ return std::make_tuple(x.map(rd)...);}, net);

    //std::apply([&](const auto&... x){ ((std::cout << x), ...);}, net);

    for (int _ = 0; _ < 1000000; ++_) {
        std::apply([&](auto&...x) { nn::train(X, y, x...); }, net);
    }

    auto A = std::apply([&](auto&...x) { return nn::forwards(X, x...); }, net);
    std::cout << A;
    std::cout << "MSE(all): " << nn::get_mse(A, y) << "\n";

    return 0;
}
