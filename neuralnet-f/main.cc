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
#include "matrix.hh"
#include "nn.hh"
#include "idx.hh"

void run_mnist() {
    auto result = idx::run<28,28,10,100,4,30>("../../mnist/train-images.idx3-ubyte",
                                              "../../mnist/train-labels.idx1-ubyte",
                                              "../../mnist/t10k-images.idx3-ubyte",
                                              "../../mnist/t10k-labels.idx1-ubyte",
                                              4);
}

int main() {
    srand(time(NULL));
    std::cout.precision(2);

    run_mnist();

    return 0;
}
