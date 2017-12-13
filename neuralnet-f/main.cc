#include "common.hh"
#include "nn.hh"
#include "matrix.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace nn;

int main() {
    srand(time(NULL));
    std::cout.precision(2);

    Matrixd<2,3> A { 1, 2, 3,
                     4, 5, 6 };
    Matrixd<3,2> B { 0, 5,
                     1, 2,
                     3, 0 };

    auto C = B.map([](auto x) { return 1 / (1 + exp(-x)); });
    B.mip([](auto x) { return 1 / (1 + exp(-x)); });

    std::cout << (B * A) << "\n";
    std::cout << (C * A) << "\n";

    // [[  2.0,  9.0 ],
    //  [  5.0, 30.0 ]]

    // [[ 20.0, 25.0, 30.0 ],
    //  [  9.0, 12.0, 15.0 ],
    //  [  3.0,  6.0,  9.0 ]]

    #ifdef NDEBUG
    std::cout << "NDEBUG\n";
    #endif

    return 0;
}
