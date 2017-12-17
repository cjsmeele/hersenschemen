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
#pragma once

#include "common.hh"
#include "matrix.hh"
#include <tuple>

namespace nn {

    constexpr auto sigma  = [](auto ws) { return 1 / (1 + exp(-ws)); };
    // Input for sigma' is an activation value (the sigma of the weighted sum of inputs).
    constexpr auto sigma_ = [](auto a)  { return a * (1 - a);        };

    constexpr auto g  = sigma;
    constexpr auto g_ = sigma_;

    constexpr auto eta = 0.1;

    template<typename...>
    struct list {};

    namespace detail {
        template<template<typename...> typename C, typename Acc, typename X, uint I>
        struct repeat;
        template<template<typename...> typename C, typename... Acc, typename X>
        struct repeat<C,list<Acc...>,X,0> {
            using type = C<Acc...>;
        };
        template<template<typename...> typename C, typename... Acc, typename X, uint I>
        struct repeat<C,list<Acc...>,X,I> {
            using type = typename repeat<C,list<X,Acc...>,X,I-1>::type;
        };
    }

    template<template<typename...> typename C, typename X, uint I>
    using repeat = typename detail::repeat<C,list<>,X,I>::type;

    // The machine spirits are willing.

    template<typename AT, typename WT, typename F = decltype(g)>
    constexpr auto forward_one(const AT &A, const WT &W, F f = g) {
        return dot(A,W).map(f);
    }

    /* Cheap forward that doesn't keep track of activations,
     * for when you care only about the output layer.
     */
    template<typename A1T, typename WT, typename... WsT>
    constexpr auto forwards(const A1T &A1, const WT &W, const WsT&... Ws) {
        auto A = forward_one(A1,W);
        if constexpr (sizeof...(WsT) > 0)
            return forwards(A, Ws...);
        else
            return A;
    }

    namespace detail {

        template<typename...>
        struct train_backward;

        template<typename L1T, typename... LsT, typename WT, typename... WsT>
        struct train_backward<list<L1T,LsT...>,list<WT,WsT...>> {
            template<typename L2DT>
            constexpr static auto f(L2DT L2D, L1T L1, LsT... Ls, WT &W, WsT&... Ws) {
                auto D = dot(L2D, W.T()) * L1.map(g_);
                W += eta * dot(L1.T(), L2D);
                if constexpr (sizeof...(WsT) > 0)
                    train_backward<list<LsT...>,list<WsT...>>
                        ::f(D, Ls..., Ws...);
            }
        };

        template<typename LAT, typename... LsT, typename... WsT, typename YT>
        struct train_backward<list<LAT,LsT...>,list<WsT...>,YT> {
            constexpr static auto f(LAT LA, LsT... Ls, WsT&... Ws, const YT &Y) {
                auto D = (Y - LA) * LA.map(g_);
                if constexpr (sizeof...(LsT) > 0)
                    train_backward<list<LsT...>,list<WsT...>>
                        ::f(D, Ls..., Ws...);
            }
        };


        template<typename...>
        struct train_forward;

        template<typename... LsT,
                 typename... WsCT>
        struct train_forward<list<LsT...>,
                             list<>,
                             list<WsCT...>> {
            template<typename YT>
            constexpr static auto f(LsT... Ls, WsCT&... Ws, const YT &Y) {
                return train_backward<list<LsT...>,list<WsCT...>,YT>::f(Ls..., Ws..., Y);
            }
        };

        template<typename... LsT,
                 typename W1T,
                 typename... WsT,
                 typename... WsCT>
        struct train_forward<list<LsT...>,
                             list<W1T,WsT...>,
                             list<WsCT...>> {
            template<typename A1T, typename YT>
            constexpr static auto f(const A1T &A1,
                                    W1T &W1,
                                    WsT&... Ws,
                                    WsCT&... WsC,
                                    LsT... Ls,
                                    const YT &Y) {

                auto A = forward_one(A1,W1);
                if constexpr (sizeof...(WsT) > 0) {
                    return train_forward<list<A1T,LsT...>,
                                         list<WsT...>,
                                         list<W1T,WsCT...>>
                           ::f(A, Ws..., W1, WsC..., A1, Ls..., Y);
                } else {
                    return train_forward<list<decltype(A),A1T,LsT...>,
                                         list<>,
                                         list<W1T,WsCT...>>
                           ::f(A, A1, Ls..., W1, WsC..., Y);
                }
            }
        };
    }

    namespace detail {

        template<typename T1, uint rows, uint cols>
        struct get_mse {
            using MT = Matrix<T1,rows,cols>;
            constexpr static double f(const MT &A, const MT &Y) {
                double sum = 0;
                for (uint i = 1; i <= MT::nrows; ++i)
                    sum += get_mse<T1,1,MT::ncols>::f(A(i), Y(i));
                return sum / MT::nrows;
            }
        };

        template<typename T1, uint cols>
        struct get_mse<T1,1,cols> {
            using MT = Matrix<T1,1,cols>;
            constexpr static double f(const MT &A, const MT &Y) {
                double sum = 0;
                auto SE = (Y - A).map([](auto x) { return x*x; });
                for (uint c = 1; c <= SE.ncols; ++c)
                    sum += SE(1,c);
                return sum / (2*A.ncols);
            }
        };
    }

    template<typename T1, uint rows, uint cols>
    constexpr double get_mse(const Matrix<T1,rows,cols> &A, const Matrix<T1,rows,cols> &Y) {
        return detail::get_mse<T1,rows,cols>::f(A, Y);
    }

    template<typename AT, typename YT, typename... WsT>
    constexpr auto train(const AT &A, const YT &Y, WsT&... Ws) {
        return detail::train_forward<list<>,list<WsT...>,list<>>::f(A, Ws..., Y);
    }

    namespace detail2 {
        template<typename T, uint InCount, uint OutCount>
        using Layer = Matrix<T, InCount, OutCount>;

        template<typename T, uint I, uint O, uint N>
        struct netcat {
            template<typename... HT>
            struct c {
                using type = std::tuple<Layer<T,I,N>,
                                        HT...,
                                        Layer<T,N,O>>;
            };
        };

        template<typename T, uint I, uint O, uint H, uint N>
        struct make_net {
            using type = typename repeat<netcat<T,I,O,N>::template c,Layer<T,N,N>,H-1>::type;
        };

        template<typename T, uint I, uint O, uint N>
        struct make_net<T,I,O,0,N> {
            using type = std::tuple<Layer<T,I,O>>;
        };
    }

    template<typename T,
             uint Inputs,
             uint Outputs,
             uint HiddenLayers,
             uint NeuronsPerLayer>
    using make_net = typename detail2::make_net<T,
                                                Inputs,
                                                Outputs,
                                                HiddenLayers,
                                                NeuronsPerLayer>::type;
}
