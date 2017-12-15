#include "common.hh"
#include "nn.hh"
#include "matrix.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <tuple>
#include <typeinfo>
#include <cxxabi.h>

using namespace nn;

constexpr auto g  = [](auto z) { return 1 / (1 + exp(-z)); };
constexpr auto g_ = [](auto z) { return g(z) * (1 - g(z)); };
//constexpr auto g_ = [](auto z) { return z * (1 - z); };

constexpr inline auto eta = 0.1;

// template<typename LM, typename W, typename... Ws>
// constexpr auto forwards(LM matrix, W weights, Ws... rest) {
//     auto A = dot(matrix, weights).map(g);
//     if constexpr (sizeof...(Ws))
//         return forwards(A, rest...);
//     else
//         return A;
// }

using std::tuple;

// The machine spirits are willing.

template<typename...>
struct forward_impl;

template<typename... LsT>
struct forward_impl<tuple<LsT...>,tuple<>> {
    constexpr auto operator()(LsT... Ls) {
        return tuple<LsT...>{ Ls... };
    }
};

template<typename... LsT, typename W1T, typename A1T>
struct forward_impl<tuple<LsT...>,tuple<W1T>,A1T> {
    constexpr auto operator()(A1T A1, W1T W1, LsT... Ls) {

        auto A = dot(A1,W1).map(g);
        return forward_impl<tuple<decltype(A),A1T,LsT...>,
                            tuple<>>{}
            (A, A1, Ls...);
    }
};

template<typename... LsT, typename W1T, typename W2T, typename... WsT, typename A1T>
struct forward_impl<tuple<LsT...>,tuple<W1T,W2T,WsT...>,A1T> {
    constexpr auto operator()(A1T A1, W1T W1, W2T W2, WsT... Ws, LsT... Ls) {

        auto A = dot(A1,W1).map(g);
        return forward_impl<tuple<A1T,LsT...>,
                            tuple<W2T,WsT...>,
                            decltype(A)>{}
            (A, W2, Ws..., A1, Ls...);
    }
};

template<typename A1T, typename... WsT>
constexpr auto forward(A1T A1, WsT... Ws) {
    return forward_impl<tuple<>,tuple<WsT...>,A1T>{}(A1, Ws...);
}

// template<typename L2DT, typename L1T, typename WT, typename... LWs>
// constexpr void backwards_impl(L2DT L2D, L1T L1, WT W, LWs... rest) {
//     auto D = dot(L2D, W.T()).map(g_);
//     W += eta * dot(L1, L2D);
//     if constexpr (sizeof...(LWs))
//         backwards(D, rest...);
// }

// template<typename YT, typename LT1, typename WT, typename... LWs>
// constexpr void backwards(YT Y, LT1 L1, WT W, LWs... rest) {
//     auto D = (Y - L1).map(g_);
//     if constexpr (sizeof...(LWs))
//         backwards_impl(D, rest...);
// }

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
    Matrixd<3,4> w0 {};
    w0.mip(rd);
    Matrixd<4,1> w1 {};
    w1.mip(rd);

    // for (int _ = 0; _ < 60000; ++_) {
    for (int _ = 0; _ < 3; ++_) {
        // auto L1 = dot(X,  w0).map(g); // 4 * 4
        // auto L2 = dot(L1, w1).map(g); // 4 * 1
        // auto L2D = (y - L2) * L2.map(g_);
        // auto L1D = dot(L2D, w1.T()) * L1.map(g_);
        // w1 += eta * dot(L1.T(), L2D);
        // w0 += eta * dot( X.T(), L1D);

        // backwards(y, forwards(X, w0, w1), L1, w1, L0, w0);
        auto x = forward(X, w0, w1);

        char *name = abi::__cxa_demangle(typeid(x).name(),
                                         nullptr,
                                         nullptr,
                                         nullptr);
        std::cout << "🦄: " << name << "\n";

        std::cout << std::get<0>(x) << "\n";
    }

    auto L1 = dot(X,  w0).map(g);
    auto L2 = dot(L1, w1).map(g);
    std::cout << L2;

    #ifdef NDEBUG
    std::cout << "NDEBUG\n";
    #endif

    return 0;
}
