#include "neural-network.hh"
#include <cstdlib>
#include <ctime>

using namespace nn;

int main() {
    srand(time(0));

    Net ketnet {{
        layer_t{ Neuron{1}, Neuron{}, Neuron{} },
        layer_t{ Neuron{1}, Neuron{} },
        layer_t{ Neuron{1}, Neuron{}, Neuron{}},
        layer_t{ Neuron{},  Neuron{}}
    }};

    // NAND adder (with step fn and no backprop):
    ketnet.connect({ /* 1,1: NAND 1.   */ { 0,0, 1,1,  2 }, { 0,1, 1,1, -1 }, { 0,2, 1,1, -1 },
                     /* 2,1: NAND 2.   */ { 1,0, 2,1,  2 }, { 0,1, 2,1, -1 }, { 1,1, 2,1, -1 },
                     /* 2,2: NAND 3.   */ { 0,2, 2,2, -1 }, { 1,0, 2,2,  2 }, { 1,1, 2,2, -1 },
                     /* 3,0: Inverter. */ { 2,0, 3,0,  1 }, { 1,1, 3,0, -1 },
                     /* 3,1: NAND 4.   */ { 2,0, 3,1,  2 }, { 2,1, 3,1, -1 }, { 2,2, 3,1, -1 }});

    auto result = ketnet.run({0, 0});
    for (const auto &r : result) std::cout << r << ' ';
    std::cout << '\n';
    result = ketnet.run({0, 1});
    for (const auto &r : result) std::cout << r << ' ';
    std::cout << '\n';
    result = ketnet.run({1, 0});
    for (const auto &r : result) std::cout << r << ' ';
    std::cout << '\n';
    result = ketnet.run({1, 1});
    for (const auto &r : result) std::cout << r << ' ';
    std::cout << '\n';

    std::cout << "Skynet sais hello" << std::endl;
    return 0;
}
