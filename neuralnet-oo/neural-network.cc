#include "neural-network.hh"

#include <utility>
#include <cstdlib>
#include <cmath>

namespace nn {

double randomWeight() {
    return ((double)rand() / RAND_MAX) * 2 - 1;
}

void Neuron::addInput(Neuron *n) {
    addInput(n, randomWeight());
}


void Neuron::addInput(Neuron *n, double weight) {
    inputs.emplace_back(n, weight);
}

void Neuron::update() {
    if (inputs.empty())
        // This is a neuron without inputs (i.e. bias).
        return;

    value = 0;
    for (const auto &n : inputs)
        value += n.first->getOutput() * n.second;

    //value = 1 / (1 + exp(-value)); // sigmoid away
    value = value > 0;
}

void Net::connect() {
    for (size_t i = 1; i < layers.size(); i++)
        for (auto &n: layers[i])
            for (auto &n2: layers[i-1])
                n.addInput(&n2);
}

void Net::connect(const Connection &c) {
    getNeuron(c.dstL, c.dstN)
        .addInput(&getNeuron(c.srcL, c.srcN),
                  c.weight);
}
void Net::connect(const std::vector<Net::Connection> &cs) {
    for (const auto &c : cs)
        connect(c);
}

std::vector<double> Net::run(std::vector<double> input) {
    // Set input neurons.
    for (size_t i = 1; i < layers[0].size(); i++)
        layers[0][i].setOutput(input[i-1]);

    // Update all other neurons.
    for (size_t i = 1; i < layers.size(); i++) {
        for (auto &n : layers[i])
            n.update();
    }

    // Collect output values.
    std::vector<double> res;
    for (auto &n : layers.back())
        res.push_back(n.getOutput());

    return res;
}

}
