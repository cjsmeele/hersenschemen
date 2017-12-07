#include "neural-network.hh"


int main() {
    srand(time(0));

    Net ketnet {{
        layer_t{ Neuron{1}, Neuron{}, Neuron{} },
        layer_t{ Neuron{1}, Neuron{} },
        layer_t{ Neuron{1}, Neuron{}, Neuron{}},
        layer_t{ Neuron{},  Neuron{}}
    }};

    ketnet.gibNeuron(1, 1).addInput(ketnet.gibLayer(0), {2, -1, -1});
    ketnet.gibNeuron(2, 1).addInput(ketnet.gibLayer(1), {2, -1});
    ketnet.gibNeuron(2, 1).addInput(ketnet.gibNeuron(0, 1), -1);
    ketnet.gibNeuron(2, 2).addInput(ketnet.gibLayer(1), {2, -1});
    ketnet.gibNeuron(2, 2).addInput(ketnet.gibNeuron(0, 2), -1);
    ketnet.gibNeuron(3, 0).addInput(ketnet.gibNeuron(2, 0), 1);
    ketnet.gibNeuron(3, 0).addInput(ketnet.gibNeuron(1, 1), -1);
    ketnet.gibNeuron(3, 1).addInput(ketnet.gibLayer(2), {2, -1, -1});

//  ketnet.interConnect()

    auto result = ketnet.run({0, 0});
    std::cout << "Results are in!:\n";
    for (const auto &r : result) {
        std::cout << r << ' ';
    }
    std::cout << '\n';
    result = ketnet.run({0, 1});
    std::cout << "Results are in!:\n";
    for (const auto &r : result) {
        std::cout << r << ' ';
    }
    std::cout << '\n';
    result = ketnet.run({1, 0});
    std::cout << "Results are in!:\n";
    for (const auto &r : result) {
        std::cout << r << ' ';
    }
    std::cout << '\n';
    result = ketnet.run({1, 1});
    std::cout << "Results are in!:\n";
    for (const auto &r : result) {
        std::cout << r << ' ';
    }
    std::cout << '\n';

    std::cout << "Skynet sais hello" << std::endl;
    return 0;
}
