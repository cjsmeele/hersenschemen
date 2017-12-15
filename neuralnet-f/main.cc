#include "common.hh"
#include "nn.hh"
#include "matrix.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace nn;

constexpr auto g = [](auto z)  { return 1 / (1 + exp(-z)); };
constexpr auto g_ = [](auto z) { return g(z) * (1 - g(z)); };

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

    auto rd = [](auto) {return (double)rand()/RAND_MAX*2 - 1;};
    Matrixd<3,4> w1 {};
    w1.mip(rd);
    Matrixd<4,1> w2 {};
    w2.mip(rd);

    constexpr auto eta = 0.1;

    for (int _ = 0; _ < 60000; ++_) {
        auto L1 = dot(X,  w1).map(g); // 4 * 4
        auto L2 = dot(L1, w2).map(g); // 4 * 1
        auto L2D = (y - L2) * L2.map(g_);
        auto L1D = dot(L2D, w2.T()) * L1.map(g_);
        w2 += eta * dot(L1.T(), L2D);
        w1 += eta * dot( X.T(), L1D);
    }

    auto L1 = dot(X,  w1).map(g);
    auto L2 = dot(L1, w2).map(g);
    std::cout << L2;

    #ifdef NDEBUG
    std::cout << "NDEBUG\n";
    #endif

    return 0;
}
