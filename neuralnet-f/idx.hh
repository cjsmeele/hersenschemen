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
#include "nn.hh"
#include <vector>
#include <iostream>
#include <fstream>
#include <string_view>

namespace idx {

    template<typename T>
    [[nodiscard]] constexpr T big_little_swap(T v) {
        static_assert(std::is_integral<T>::value, "T must be of integral type");
        static_assert(std::is_unsigned<T>::value, "swap is only defined for unsigned types");
        alignas(T) uint8_t r[sizeof(T)] = {};
        for (size_t i = 0; i < sizeof(T); ++i)
            r[i] = ((uint8_t*)&v)[sizeof(T)-i-1];
        T *x = (T*)r;
        return *x;
    }

    template<typename T>
    bool get_the_thing_from_the_thing(std::istream &is, T &v) {
        is.read((char*)&v, sizeof(T));
        return (bool)is;
    }

    template<typename = void> /* hush */
    std::vector<uint8_t> read_idx1(std::string_view filename) {
        constexpr auto max_records = 1'000'000;
        std::ifstream file(std::string(filename), std::ios::binary);
        if (!file)
            throw std::runtime_error("Could not open label file");
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size < 8)
            throw std::runtime_error("short short too short for NN");

        { uint32_t magic;
          if (!get_the_thing_from_the_thing(file, magic)
              || magic != big_little_swap(2049U)) {

              throw std::runtime_error("magic number mismatch");
          } }

        uint32_t record_count;
        get_the_thing_from_the_thing(file, record_count);
        record_count = big_little_swap(record_count);

        std::cout << "records: " << record_count << "\n";
        if (record_count > max_records)
            throw std::runtime_error("too many records");

        std::vector<uint8_t> result;
        result.resize(record_count);
        if (!file.read((char*)result.data(), record_count))
            throw std::runtime_error("could not read records");

        return result;
    }

    // constexpr unsigned Rows = 28;
    // constexpr unsigned Cols = 28;
    // constexpr auto max_records = 1'000'000;

    template<uint Rows, uint Cols, uint max_records = 1'000'000>
    std::vector<uint8_t> read_idx3(std::string_view filename) {

        std::ifstream file(std::string(filename), std::ios::binary);
        if (!file)
            throw std::runtime_error("Could not open data file");
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size < 4*4)
            throw std::runtime_error("short short too short for NN");

        { uint32_t magic;
          if (!get_the_thing_from_the_thing(file, magic)
              || magic != big_little_swap(2051U)) {

              throw std::runtime_error("magic number mismatch");
          } }

        uint32_t record_count; get_the_thing_from_the_thing(file, record_count);
        record_count = big_little_swap(record_count);

        std::cout << "records: " << record_count << "\n";
        if (record_count > max_records)
            throw std::runtime_error("too many records");

        uint32_t rows; get_the_thing_from_the_thing(file, rows);
        uint32_t cols; get_the_thing_from_the_thing(file, cols);
        if (rows != big_little_swap(Rows) || cols != big_little_swap(Cols))
            throw std::runtime_error("image dims do not match");

        std::vector<uint8_t> result;
        result.resize(record_count * Rows * Cols);
        if (!file.read((char*)result.data(), record_count * Rows * Cols))
            throw std::runtime_error("could not read records");

        return result;
    }

    template<uint SZ,uint W, uint H>
    void print_images(const Matrixd<SZ, W*H> &n) {
        for(int i = 1; i <= SZ; ++i) {
            for (int y = 1; y <= H; ++y) {
                for (int j = 1; j <= W; ++j) {
                    auto p = n(i, (y-1) * W + j);
                    if (p <= 0.2)
                        std::cout << " ";
                    else if (p <= 0.4)
                        std::cout << "░";
                    else if (p <= 0.6)
                        std::cout << "▒";
                    else if (p <= 0.8)
                        std::cout << "▓";
                    else
                        std::cout << "█";
                }
                std::cout << '\n';
            }
        }
    }

    template<typename Net>
    struct RunResult {
        Net net;
        double mse;
        int correct;
        int total;
    };

    template<uint rows, uint cols,
             uint output_layer_size,
             uint batch_size,
             uint hidden_layers,
             uint neurons_per_layer>
    auto run(std::string_view train_images,
             std::string_view train_labels,
             std::string_view test_images,
             std::string_view test_labels,
             int training_rounds) {

        constexpr auto input_layer_size = rows*cols;

        Matrixd<batch_size,  input_layer_size> X_training;
        Matrixd<batch_size, output_layer_size> Y_training;
        Matrixd<batch_size,  input_layer_size> X_test;
        Matrixd<batch_size, output_layer_size> Y_test;

        auto label_buffer = read_idx1(train_labels);
        auto data_buffer  = read_idx3<rows,cols>(train_images);

        // This does the thing.
        auto net = nn::make_net<double,
                                input_layer_size,
                                output_layer_size,
                                hidden_layers,
                                neurons_per_layer>();

        std::apply([](auto& ...x){(x.mip([](auto) {return (double)rand()/RAND_MAX*2 - 1;}), ...);}, net);

        for (auto i = 0; i < training_rounds; ++i) {
            std::cout << "round " << (i+1) << "/" << training_rounds << "\n";
            for (uint j = 0; j < label_buffer.size() / batch_size /*j < 1*/; ++j) {
                X_training *= 0; Y_training *= 0;
                for (uint k = 0; k < batch_size; ++k) {
                    for (uint l = 0; l < input_layer_size; ++l)
                        X_training(k+1, l+1) = (double)data_buffer[j*batch_size*input_layer_size + k*input_layer_size + l] / 255;
                    Y_training(k+1, label_buffer[j*batch_size + k]+1) = 1;
                }
                std::apply([&](auto&...x) { nn::train(X_training, Y_training, x...); }, net);
            }
        }

        label_buffer = read_idx1(test_labels);
        data_buffer  = read_idx3<rows,cols>(test_images);

        int correct      = 0;
        int total        = 0;
        double total_mse = 0;

        for (size_t j = 0; j < label_buffer.size() / batch_size; ++j) {
            X_test *= 0; Y_test *= 0;
            for (uint k = 0; k < batch_size; ++k) {
                for (uint l = 0; l < input_layer_size; ++l)
                    X_test(k+1, l+1) = (double)data_buffer[j*batch_size*input_layer_size + k*input_layer_size + l] / 255;
                Y_test(k+1, label_buffer[j*batch_size + k]+1) = 1;
                // print_images<batch_size,28,28>(X_test);
            }
            auto A = std::apply([&](auto&...x) { return nn::forwards(X_test, x...); }, net);

            //std::cout << "A:\n" << A;
            //std::cout << "Y:\n" << Y_test;
            auto mse = nn::get_mse(A, Y_test);
            std::cout << "Δ:\n" << (A - Y_test);
            std::cout << "MSE:\n" << mse << "\n";

            total_mse += mse;

            for (uint r = 1; r <= A.nrows; ++r) {
                ++total;
                bool waarom_heeft_cpp_geen_continues_naar_outer_loops = false;
                for (uint c = 1; c <= A.ncols; ++c) {
                    if ((A(r,c) > 0.5) != (Y_test(r,c) > 0.5)) {
                        waarom_heeft_cpp_geen_continues_naar_outer_loops = true;
                        break;
                    }
                }
                if (!waarom_heeft_cpp_geen_continues_naar_outer_loops)
                    ++correct;
            }
        }
        std::cout << "Correct:  " << correct << "/" << total << "\n";
        std::cout << "Percentage:  " << double(correct) / double(total) * 100.0 << "\n";

        return RunResult<decltype(net)> { net, total_mse/(label_buffer.size() / batch_size),
                                          correct, total };
    }

}
