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

    template<typename...>
    struct list {};

    // The machine spirits are willing.

    template<typename AT, typename WT, typename F = decltype(g)>
    constexpr auto forward_one(const AT &A, const WT &W, F g = g) {
        return dot(A,W).map(g);
    }

    /* Cheap forward that doesn't keep track of activations,
     * for when you care only about the output layer.
     */
    template<typename A1T, typename WT, typename... WsT>
    constexpr auto forwards(const A1T &A1, const WT &W, const WsT&... Ws) {
        auto A = forward_one(A1,W);
        if constexpr (sizeof...(WsT))
            return forwards(A, Ws...);
        else
            return A;
    }

    // namespace detail {

    //     template<typename...>
    //     struct forward;

    //     template<typename... LsT>
    //     struct forward<list<LsT...>,list<>> {
    //         constexpr static auto f(LsT... Ls) {
    //             return std::tuple { Ls... };
    //         }
    //     };

    //     template<typename... LsT, typename W1T, typename... WsT>
    //     struct forward<list<LsT...>,list<W1T,WsT...>> {
    //         template<typename A1T>
    //         constexpr static auto f(const A1T &A1,
    //                                 W1T &W1,
    //                                 WsT&... Ws,
    //                                 LsT... Ls) {

    //             auto A = forward_one(A1,W1);
    //             if constexpr (sizeof...(WsT)) {
    //                 return forward<list<A1T,LsT...>,
    //                                list<WsT...>>
    //                        ::f(A, Ws..., A1, Ls...);
    //             } else {
    //                 return forward<list<decltype(A),A1T,LsT...>,
    //                                list<>>
    //                        ::f(A, A1, Ls...);
    //             }
    //         }
    //     };
    // }

    // template<typename A1T, typename... WsT>
    // constexpr auto forward(const A1T &A1, WsT&... Ws) {
    //     return detail::forward<list<>,list<WsT...>>::f(A1, Ws...);
    // }


    namespace detail {

        template<typename...>
        struct train_backward;

        template<typename L1T, typename... LsT, typename WT, typename... LWs>
        struct train_backward<list<L1T,LsT...>,list<WT,LWs...>> {
            template<typename L2DT, typename L1T, typename WT, typename... LWs>
            constexpr static auto f(L2DT L2D, L1T L1, LsT... Ls, WT &W WsT&... Ws) {
                // return std::tuple { Ls... };
                auto D = dot(L2D, W.T()).map(g_);
                W += eta * dot(L1, L2D);
                if constexpr (sizeof...(LWs))
                                    backwards(D, rest...);
            }
        };

        template<typename LAT, typename... LsT, typename... WsT, typename YT>
        struct train_backward<list<LAT,LsT...>,list<WsT...>> {
            constexpr static auto f(LAT LA, LsT... Ls, WsT&... Ws, const YT &Y) {
                auto D = (Y - LA).map(g_);
                if constexpr (sizeof...(LsT))
                    train_backward<decltype(D),list<LsT...>,list<WsT...>>
                                  ::f(D, Ls..., Ws...);

                // return std::tuple { Ls..., Ws... };

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
            }
        };


        template<typename...>
        struct train_forward;

        template<typename... LsT,
                 typename... WsCT>
        struct train_forward<list<LsT...>,
                             list</*MISSCHIEN DOOR LEGE LISTS?*/>,
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
                if constexpr (sizeof...(WsT)) {
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

    template<typename AT, typename YT, typename... WsT>
    constexpr auto train(const AT &A, const YT &Y, WsT&... Ws) {
        return detail::train_forward<list<>,list<WsT...>,list<>>::f(A, Ws..., Y);
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
                    sum += SE(c,1);
                return sum / (2*A.ncols);
            }
        };
    }

    template<typename T1, uint rows, uint cols>
    constexpr double get_mse(const Matrix<T1,rows,cols> &A, const Matrix<T1,rows,cols> &Y) {
        return detail::get_mse<T1,rows,cols>::f(A, Y);
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



    // namespace detail {
    //     template<typename AT, typename YT, typename... LsT, typename... WsT>
    //     struct backward {
    //     };

    //     template<typename XT, typename YT, typename... WsT>
    //     struct train {
    //         // constexpr static double f(LsT... Ls, WsT... Ws) {

    //         template<typename... LsT>
    //         struct forward_into_backward {
    //             constexpr static double f(XT X, YT Y, WsT... Ws) {
    //             }
    //         };

    //         constexpr static double f(XT X, YT Y, WsT... Ws) {

    //             auto Ls = nn::forward<forward_into_backward,XT,WsT>(X, w0, w1);
    //             return 0;
    //         }
    //     };
    // }


    // struct train_impl<A1T,YT,list<LsT...>,list<WsT...>> {
    //     constexpr auto operator()(LsT... Ls) {
    //         return list<LsT...>{ Ls... };
    //     }
    // };
    // constexpr auto train(A1T A1, YT Y, WsT... &Ws,) {
    // }


}
