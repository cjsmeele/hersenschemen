#include "common.hh"
#include "nn.hh"
#include "matrix.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>

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

    auto aoeu = nn::make_net<double,3,1,0,4>{};
    char *name = abi::__cxa_demangle(typeid(aoeu).name(),
                                     nullptr,
                                     nullptr,
                                     nullptr);
    std::cout << "🦄🦄🦄🦄🦄🦄🦄: " << name << "\n";

    auto rd = [](auto) {return (double)rand()/RAND_MAX*2 - 1;};
    Matrixd<3,4> w0 {};
    w0.mip(rd);
    Matrixd<4,1> w1 {};
    w1.mip(rd);

    for (int _ = 0; _ < 10000; ++_) {
        // auto L1 = dot(X,  w0).map(g); // 4 * 4
        // auto L2 = dot(L1, w1).map(g); // 4 * 1
        // auto L2D = (y - L2) * L2.map(g_);
        // auto L1D = dot(L2D, w1.T()) * L1.map(g_);
        // w1 += eta * dot(L1.T(), L2D);
        // w0 += eta * dot( X.T(), L1D);

        // backwards(y, forwards(X, w0, w1), L1, w1, L0, w0);
        // auto x = nn::forward(X, w0, w1);
        //auto x = nn::train(X, y, w0, w1);
        nn::train(X, y, w0, w1);
        //auto A = nn::forwards(X, w0, w1);
        //std::cout << "a:\n" << A;
        //std::cout << "y:\n" << y;
        //std::cout << "MSE(  1): " << nn::get_mse(A(1), y(1)) << "\n";
        //std::cout << "MSE(all): " << nn::get_mse(A, y) << "\n";

        //char *name = abi::__cxa_demangle(typeid(x).name(),
        //                                 nullptr,
        //                                 nullptr,
        //                                 nullptr);
        //std::cout << "🦄: " << name << "\n";

        //std::cout << "a:\n" << std::get<0>(x);
        //std::cout << "y:\n" << y;
        //std::cout << "MSE(  1): " << nn::get_mse(std::get<0>(x)(1), y(1)) << "\n";
        //std::cout << "MSE(all): " << nn::get_mse(std::get<0>(x), y) << "\n";
    }

    auto A = nn::forwards(X, w0, w1);
    std::cout << A;
    std::cout << "MSE(all): " << nn::get_mse(A, y) << "\n";


    //auto L1 = dot(X,  w0).map(nn::g);
    //auto L2 = dot(L1, w1).map(nn::g);
    //std::cout << L2;

    #ifdef NDEBUG
    std::cout << "NDEBUG\n";
    #endif

    return 0;
}
