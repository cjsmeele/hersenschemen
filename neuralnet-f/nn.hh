#pragma once

#include "common.hh"
#include "matrix.hh"
#include <tuple>

namespace nn {

    constexpr auto sigma  = [](auto ws) { return 1 / (1 + exp(-ws)); };
    constexpr auto sigma_ = [](auto a)  { return a * (1 - a);        };
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

    template<typename...>
    struct tuple {};

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
            return std::tuple<LsT...>{ Ls... };
        }
    };

    template<typename... LsT, typename W1T, typename... WsT>
    struct forward_impl<tuple<LsT...>,tuple<W1T,WsT...>> {
        template<typename A1T>
        constexpr auto operator()(A1T A1, W1T W1, WsT... Ws, LsT... Ls) {

            auto A = forward_one(A1,W1);
            if constexpr (sizeof...(WsT)) {
                return forward_impl<tuple<A1T,LsT...>,
                                    tuple<WsT...>>{}
                (A, Ws..., A1, Ls...);
            } else {
                return forward_impl<tuple<decltype(A),A1T,LsT...>,
                                    tuple<>>{}
                (A, A1, Ls...);
            }
        }
    };

    template<typename A1T, typename... WsT>
    constexpr auto forward(A1T A1, WsT... Ws) {
        return forward_impl<tuple<>,tuple<WsT...>>{}(A1, Ws...);
    }

    // template<typename L2DT, typename L1T, typename WT, typename... LWs>
    // constexpr void backwards_impl(L2DT L2D, L1T L1, WT W, LWs... rest) {
    //     auto D = dot(L2D, W.T()).map(g_);
    //     W += eta * dot(L1, L2D);
    //     if constexpr (sizeof...(LWs))
    //                      backwards(D, rest...);
    // }

    // template<typename YT, typename LT1, typename WT, typename... LWs>
    // constexpr void backwards(YT Y, LT1 L1, WT W, LWs... rest) {
    //     auto D = (Y - L1).map(g_);
    //     if constexpr (sizeof...(LWs))
    //         backwards_impl(D, rest...);
    // }

    template<typename T1, uint rows, uint cols>
    struct get_mse_impl {
        using MT = Matrix<T1,rows,cols>;
        constexpr static double f(const MT &A, const MT &Y) {
            double sum = 0;
            for (uint i = 1; i <= MT::nrows; ++i)
                sum += get_mse_impl<T1,1,MT::ncols>::f(A(i), Y(i));
            return sum / MT::nrows;
        }
    };

    template<typename T1, uint cols>
    struct get_mse_impl<T1,1,cols> {
        using MT = Matrix<T1,1,cols>;
        constexpr static double f(const MT &A, const MT &Y) {
            double sum = 0;
            auto SE = (Y - A).map([](auto x) { return x*x; });
            for (uint c = 1; c <= SE.ncols; ++c)
                sum += SE(c,1);
            return sum / (2*A.ncols);
        }
    };

    template<typename T1, uint rows, uint cols>
    constexpr double get_mse(const Matrix<T1,rows,cols> &A, const Matrix<T1,rows,cols> &Y) {
        return get_mse_impl<T1,rows,cols>::f(A, Y);
    }



    // template<typename...>
    // struct train_impl;

    // template<typename A1T, typename YT, typename... LsT, typename... WsT>
    // struct train_impl<A1T,YT,tuple<LsT...>,tuple<WsT...>> {
    //     constexpr auto operator()(LsT... Ls) {
    //         return tuple<LsT...>{ Ls... };
    //     }
    // };
    // constexpr auto train(A1T A1, YT Y, WsT... &Ws,) {
    // }


}
