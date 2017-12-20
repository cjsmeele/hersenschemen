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
#include "nn.hh"
#include "matrix.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <typeinfo>
#include <cxxabi.h>
#include <tuple>
#include <fstream>
#include <type_traits>
#include <vector>
#include <string_view>

template<typename T>
[[nodiscard]] constexpr T big_little_swap(T v) {
    static_assert(std::is_integral<T>::value, "T must be of integral type");
    static_assert(std::is_unsigned<T>::value, "swap is only defined for unsigned types");
    uint8_t r[sizeof(T)] alignas(T) = {};
    for (size_t i = 0; i < sizeof(T); ++i)
        r[i] = ((uint8_t*)&v)[sizeof(T)-i-1];
    return *(T*)r;
}

template<typename T>
bool get_the_thing_from_the_thing(std::istream &is, T &v) {
    is.read((char*)&v, sizeof(T));
    return (bool)is;
}

std::vector<uint8_t> read_mnist_idx1(std::string_view filename) {
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

constexpr unsigned image_rows = 28;
constexpr unsigned image_cols = 28;

std::vector<uint8_t> read_mnist_idx3(std::string_view filename) {

    constexpr auto max_records = 1'000'000;

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
    if (rows != big_little_swap(image_rows) || cols != big_little_swap(image_cols))
        throw std::runtime_error("image dims do not match");

    std::vector<uint8_t> result;
    result.resize(record_count * image_rows * image_cols);
    if (!file.read((char*)result.data(), record_count * image_rows * image_cols))
        throw std::runtime_error("could not read records");

    return result;
}

template<uint I>
void print_number(const Matrixd<I, 784>& n) {
    for(int i = 1; i <= I; ++i) {
        for (int y = 1; y <= 28; ++y) {
            for (int j = 1; j <= 28; ++j) {
                auto p = n(i, (y-1) * 28 + j);
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

void run_mnist() {
// Parser yay

   constexpr auto input_layer_size  = 28*28;
   constexpr auto output_layer_size =    10;
   constexpr auto batch_size        =   100;
   constexpr auto training_rounds   =    25;
   const std::string_view filename_train_images = "../../mnist/train-images.idx3-ubyte";
   const std::string_view filename_train_labels = "../../mnist/train-labels.idx1-ubyte";
   const std::string_view filename_test_images  = "../../mnist/t10k-images.idx3-ubyte";
   const std::string_view filename_test_labels  = "../../mnist/t10k-labels.idx1-ubyte";
   Matrixd<batch_size,  input_layer_size> X_training;
   Matrixd<batch_size, output_layer_size> Y_training;
   Matrixd<batch_size,  input_layer_size> X_test;
   Matrixd<batch_size, output_layer_size> Y_test;

   auto label_buffer = read_mnist_idx1(filename_train_labels);
   auto data_buffer  = read_mnist_idx3(filename_train_images);

   // This does the thing.
   auto net = nn::make_net<double,input_layer_size,output_layer_size,2,15>();
   std::apply([](auto& ...x){(x.mip([](auto) {return (double)rand()/RAND_MAX*2 - 1;}), ...);}, net);

   for (auto i = 0; i < training_rounds; ++i) {
       std::cout << "round " << i << "/" << training_rounds << "\n";
       for (int j = 0; j < label_buffer.size() / batch_size /*j < 1*/; ++j) {
           X_training *= 0; Y_training *= 0;
           for (int k = 0; k < batch_size; ++k) {
               for (int l = 0; l < input_layer_size; ++l)
                    X_training(k+1, l+1) = (double)data_buffer[j*batch_size*input_layer_size + k*input_layer_size + l] / 255;
               Y_training(k+1, label_buffer[j*batch_size + k]+1) = 1;
           }
//           std::cout << "Batch X:";
           //print_number(X_training); // Looks like after a while inputs still get scrambled, very strange, will investagate further
//           std::cout << Y_training;
           std::apply([&](auto&...x) { nn::train(X_training, Y_training, x...); }, net);
       }
   }

   label_buffer = read_mnist_idx1(filename_test_labels);
   data_buffer  = read_mnist_idx3(filename_test_images);

   int correct = 0;
   int total   = 0;

   for (int j = 0; j < label_buffer.size() / batch_size ; ++j) {
       X_test *= 0; Y_test *= 0;
       for (int k = 0; k < batch_size; ++k) {
           for (int l = 0; l < input_layer_size; ++l)
               X_test(k+1, l+1) = (double)data_buffer[j*batch_size*input_layer_size + k*input_layer_size + l] / 255;
           Y_test(k+1, label_buffer[j*batch_size + k]+1) = 1;
//           print_number(X_test);
       }
       auto A = std::apply([&](auto&...x) { return nn::forwards(X_test, x...); }, net);
  //         auto A = std::apply([&](auto&...x) {
  //             Matrixd<X_test.nrows,X_test.ncols+1> stuff;
  //             for (uint row = 1; row <= X_test.nrows; row++) {
  //                 for (uint col = 1; col <= X_test.ncols; col++)
  //                     stuff(row, col) = A(row, col);
  //                 stuff(row, X_test.ncols+1) = 1;
  //             }
  //             return nn::forwards(stuff, x...);
  //             //return nn::forwards(X_test, x...);
  //         }, net);



       //std::cout << "A:\n" << A;
       //std::cout << "Y:\n" << Y_test;
       std::cout << "A:\n" << (A - Y_test);
       std::cout << "MSE(all):\n" << nn::get_mse(A, Y_test) << "\n";

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
   std::cout << "Correct:  " << correct << "/" << total << "\\n";
}

int main() {
    srand(time(NULL));
    std::cout.precision(2);

    run_mnist();
    return 0;
//
//    Matrixd<4,3> X {
//            0, 0, 1,
//            0, 1, 1,
//            1, 0, 1,
//            1, 1, 1
//    };
//
//    Matrixd<4,1> y {
//            0,
//            1,
//            1,
//            0
//    };
//
//    auto net = nn::make_net<double,3,1,1,4>{};
//    char *name = abi::__cxa_demangle(typeid(net).name(),
//                                     nullptr,
//                                     nullptr,
//                                     nullptr);
//    std::cout << "🦄🦄🦄🦄🦄🦄🦄: " << name << "\n";
//
//
//    auto rd = [](auto) {return (double)rand()/RAND_MAX*2 - 1;};
//    std::apply([&rd](auto& ...x){(x.mip(rd) , ...);}, net);
//    //auto net = std::apply([&](auto... x){ return std::make_tuple(x.map(rd)...);}, net);
//
//    //std::apply([&](const auto&... x){ ((std::cout << x), ...);}, net);
//
//    for (int _ = 0; _ < 100000; ++_) {
//        std::apply([&](auto&...x) { nn::train(X, y, x...); }, net);
//    }
//
//    auto A = std::apply([&](auto&...x) { return nn::forwards(X, x...); }, net);
//    std::cout << A;
//    std::cout << "MSE(all): " << nn::get_mse(A, y) << "\n";
//
//    return 0;
//    run();
}
