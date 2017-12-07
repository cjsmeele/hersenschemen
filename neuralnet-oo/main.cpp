#include "neural-network.hh"


int main() {
    srand(time(0));

    Net ketnet {{
        layer_t{ Neuron{1}, Neuron{}, Neuron{} },
        layer_t{ Neuron{1}, Neuron{} },
        layer_t{ Neuron{1}, Neuron{}, Neuron{}},
        layer_t{ Neuron{},  Neuron{}}
    }};

    Neuron x1, x2, b1, b2, b3, n1, n2, i1, sum, carry;
    Net NandNet;
    NandNet.

    ketnet.interConnect();

    auto result = ketnet.run({50, 1293});
    std::cout << "Results are in!:\n";
    for (const auto &r : result) {
        std::cout << r << ' ';
    }
    std::cout << '\n';

    std::cout << "Skynet sais hello" << std::endl;
    return 0;
}
