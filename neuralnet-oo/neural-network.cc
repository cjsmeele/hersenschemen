#include "neural-network.hh"

#include <utility>
#include <cstdlib>
#include <cmath>

namespace nn {

double randomWeight() {
    double random = (double) rand() / (double) RAND_MAX;
    double r = random * 2;
    return r - 1;
}

void Neuron::addInput(layer_t &l) {
    for (auto &n : l)
        addInput(&n);
}

void Neuron::addInput(layer_t &l, const std::vector<double> &weights) {
    for (size_t i = 0; i < l.size(); i++)
        addInput(&l[i], weights[i]);
}

void Neuron::addInput(Neuron *n) {
    addInput(n, randomWeight());
}


void Neuron::addInput(Neuron *n, double weight) {
    inputs.emplace_back(n, weight);
}

void Neuron::removeInput(Neuron *np) {
    for (size_t i = 0; i < inputs.size(); i++) {
        if (inputs[i].first == np) {
            std::swap(inputs[i], inputs.back());
            inputs.pop_back();
            break;
        }
    }
}

void Neuron::setWeight(Neuron *np, double weight) {
    for (auto &n : inputs) {
        if (n.first == np) {
            n.second = weight;
            break;
        }
    }
}

void Neuron::calculate() {
    if (inputs.empty())
        // This is a neuron without inputs (i.e. bias).
        return;

    outputValue = 0;
    for (const auto &n : inputs) {
        outputValue += n.first->getOutput() * n.second;
        //std::cout << "I GOT DIS: " << n.first->getOutput() << " TIMES DIS: " << n.second << '\n';
    }
    //std::cout << "HERRO I EM KEKULATING " << outputValue << "\n";
    //outputValue = 1 / (1 + exp(-outputValue)); // sigmoid away
    outputValue = outputValue > 0;
}

void Neuron::setWeights(const std::vector<double> &w) {
    for (size_t i = 0; i < w.size(); i++)
        inputs[i].second = w[i];
}


void Net::addNeuron(uint layer, Neuron n) {
    if (layers.size() <  layer - 1)
        return;
    if (layers.size() == layer - 1)
        layers.emplace_back(layer_t{ Neuron{-1} }); // if the layer does not exist, make one and add a bias
    layers[layer].emplace_back(std::move(n));
}

void Net::addLayer(uint size) {
    auto layer = layers.size() + 1;
    for (size_t i = 0; i < size; i++)
        addNeuron(layer, Neuron{});
}

void Net::connect() {
    for (size_t i = 1; i < layers.size(); i++)
        for (auto &n: layers[i])
            n.addInput(layers[i - 1]);
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
    for (size_t i = 1; i < layers[0].size(); i++)
        layers[0][i].setOutput(input[i-1]);

    for (size_t i = 1; i < layers.size(); i++) {
        for (auto &n : layers[i])
            n.calculate();
    }

    std::vector<double> res;
    for (auto &n : layers.back())
        res.push_back(n.getOutput());

    return res;
}

}
