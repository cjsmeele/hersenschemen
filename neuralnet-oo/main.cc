#include "common.hh"
#include "neural-network.hh"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace nn;

// void adder() {
//     Net ketnet {{
//         layer_t{ Neuron{1}, Neuron{}, Neuron{} },
//         layer_t{ Neuron{1}, Neuron{} },
//         layer_t{ Neuron{1}, Neuron{}, Neuron{}},
//         layer_t{ Neuron{},  Neuron{}}
//     }};

//     // NAND adder (with step fn and no backprop):
//     ketnet.connect({ /* 1,1: NAND 1.   */ { 0,0, 1,1,  2 }, { 0,1, 1,1, -1 }, { 0,2, 1,1, -1 },
//                      /* 2,1: NAND 2.   */ { 1,0, 2,1,  2 }, { 0,1, 2,1, -1 }, { 1,1, 2,1, -1 },
//                      /* 2,2: NAND 3.   */ { 0,2, 2,2, -1 }, { 1,0, 2,2,  2 }, { 1,1, 2,2, -1 },
//                      /* 3,0: Inverter. */ { 2,0, 3,0,  1 }, { 1,1, 3,0, -1 },
//                      /* 3,1: NAND 4.   */ { 2,0, 3,1,  2 }, { 2,1, 3,1, -1 }, { 2,2, 3,1, -1 }});

//     auto result = ketnet.run({0, 0});
//     for (const auto &r : result) std::cout << r << ' ';
//     std::cout << '\n';
//     result = ketnet.run({0, 1});
//     for (const auto &r : result) std::cout << r << ' ';
//     std::cout << '\n';
//     result = ketnet.run({1, 0});
//     for (const auto &r : result) std::cout << r << ' ';
//     std::cout << '\n';
//     result = ketnet.run({1, 1});
//     for (const auto &r : result) std::cout << r << ' ';
//     std::cout << '\n';
// }

void inverter() {
    std::cout << "\nInverter:\n";
    Net net(1, 1, 0, 0);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0 }, { 1 });
        net.train({ 1 }, { 0 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 ] => " << net.run({ 0 }) << "\n";
    std::cout << "[ 1 ] => " << net.run({ 1 }) << "\n";
}

void nand() {
    std::cout << "\n3-way equality:\n";
    Net net(3, 1, 2, 4);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0, 0, 0 }, { 1 });
        net.train({ 0, 0, 1 }, { 0 });
        net.train({ 0, 1, 0 }, { 0 });
        net.train({ 0, 1, 1 }, { 0 });
        net.train({ 1, 0, 0 }, { 0 });
        net.train({ 1, 0, 1 }, { 0 });
        net.train({ 1, 1, 0 }, { 0 });
        net.train({ 1, 1, 1 }, { 1 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 0 0 ] => " << net.run({ 0, 0, 0 }) << "\n";
    std::cout << "[ 0 0 1 ] => " << net.run({ 0, 0, 1 }) << "\n";
    std::cout << "[ 0 1 0 ] => " << net.run({ 0, 1, 0 }) << "\n";
    std::cout << "[ 0 1 1 ] => " << net.run({ 0, 1, 1 }) << "\n";
    std::cout << "[ 1 0 0 ] => " << net.run({ 1, 0, 0 }) << "\n";
    std::cout << "[ 1 0 1 ] => " << net.run({ 1, 0, 1 }) << "\n";
    std::cout << "[ 1 1 0 ] => " << net.run({ 1, 1, 0 }) << "\n";
    std::cout << "[ 1 1 1 ] => " << net.run({ 1, 1, 1 }) << "\n";
}

void xor_() {
    std::cout << "\n3-way XOR-ish gate:\n";
    Net net(3, 1, 2, 4);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0, 0, 0 }, { 0 });
        net.train({ 0, 0, 1 }, { 1 });
        net.train({ 0, 1, 0 }, { 1 });
        net.train({ 0, 1, 1 }, { 0 });
        net.train({ 1, 0, 0 }, { 1 });
        net.train({ 1, 0, 1 }, { 0 });
        net.train({ 1, 1, 0 }, { 0 });
        net.train({ 1, 1, 1 }, { 0 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 0 0 ] => " << net.run({ 0, 0, 0 }) << "\n";
    std::cout << "[ 0 0 1 ] => " << net.run({ 0, 0, 1 }) << "\n";
    std::cout << "[ 0 1 0 ] => " << net.run({ 0, 1, 0 }) << "\n";
    std::cout << "[ 0 1 1 ] => " << net.run({ 0, 1, 1 }) << "\n";
    std::cout << "[ 1 0 0 ] => " << net.run({ 1, 0, 0 }) << "\n";
    std::cout << "[ 1 0 1 ] => " << net.run({ 1, 0, 1 }) << "\n";
    std::cout << "[ 1 1 0 ] => " << net.run({ 1, 1, 0 }) << "\n";
    std::cout << "[ 1 1 1 ] => " << net.run({ 1, 1, 1 }) << "\n";
}

void adder() {
    std::cout << "\nHalf adder:\n";

    Net net(2, 2, 1, 2);
    for (int i = 0; i < 10000; ++i) {
        net.train({ 0, 0 }, { 0, 0 });
        net.train({ 0, 1 }, { 1, 0 });
        net.train({ 1, 0 }, { 1, 0 });
        net.train({ 1, 1 }, { 1, 1 });
    }

    std::cout << net << "\n";

    std::cout << "[ 0 0 ] => " << net.run({ 0, 0 }) << "\n";
    std::cout << "[ 0 1 ] => " << net.run({ 0, 1 }) << "\n";
    std::cout << "[ 1 0 ] => " << net.run({ 1, 0 }) << "\n";
    std::cout << "[ 1 1 ] => " << net.run({ 1, 1 }) << "\n";
}

void divisible_by_three() {
    std::cout << "\n6-bit divisible by three checker (YMMV):\n";

    Net net(6, 1, 3, 4);
    for (int i = 0; i < 10000; ++i) {
        for (int j = 0; j < 64; ++j)
            net.train({ (double)((j>>5)&1),
                        (double)((j>>4)&1),
                        (double)((j>>3)&1),
                        (double)((j>>2)&1),
                        (double)((j>>1)&1),
                        (double)( j    &1) },
                      { (double)(j % 3 == 0) });
    }

    std::cout << net << "\n";

    for (int i = 0; i < 64; ++i) {
        std::cout << "[";
        for (int j = 0; j < 6; ++j)
            std::cout << ' ' << ((i>>(5-j))&1);
        std::cout << " ] => "
                  << net.run({ (double)((i>>5)&1),
                               (double)((i>>4)&1),
                               (double)((i>>3)&1),
                               (double)((i>>2)&1),
                               (double)((i>>1)&1),
                               (double)( i    &1) }) << "\n";
    }
}

int main() {
    srand(time(0));
    std::cout.precision(2);

    inverter();
    nand();
    xor_();
    adder();
    divisible_by_three();

    return 0;
}
