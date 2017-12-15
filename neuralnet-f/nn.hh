#pragma once

#include <tuple>
#include "common.hh"
#include "matrix.hh"

namespace nn {

    constexpr auto sigma  = [](auto ws) { return 1 / (1 + exp(-ws)); };
    constexpr auto sigma_ = [](auto a)  { return a * (1 - a);       };
    // constexpr auto sigma_ = [](auto a)  { return sigma(a) * (1 - sigma(a)); };

    constexpr auto g  = sigma;
    constexpr auto g_ = sigma_;

    constexpr auto eta = 0.1;

    // Cheap forward that doesn't keep track of activations,
    // for when you care only about the output layer.
    //
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

    template<typename AT, typename WT, typename F = decltype(g)>
    constexpr auto forward_one(const AT &A, const WT &W, F g = g) {
        return dot(A,W).map(g);
    }

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

            auto A = forward_one(A1,W1);
            return forward_impl<tuple<decltype(A),A1T,LsT...>,
                                tuple<>>{}
            (A, A1, Ls...);
        }
    };

    template<typename... LsT, typename W1T, typename W2T, typename... WsT, typename A1T>
    struct forward_impl<tuple<LsT...>,tuple<W1T,W2T,WsT...>,A1T> {
        constexpr auto operator()(A1T A1, W1T W1, W2T W2, WsT... Ws, LsT... Ls) {

            auto A = forward_one(A1,W1);
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

// template<typename A1T, typename... WsT>
// constexpr auto train(A1T A1, WsT... &Ws) {
// }

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

}
