#include "neural-network.hh"

#include <utility>
#include <cstdlib>
#include <cmath>

double randomWeight() {
    double random = (double) rand() / (double) RAND_MAX;
    double r = random * 2;
    return r - 1;
}

void Neuron::addInput(layer_t &l) {
    for (auto &n : l)
        inputs.emplace_back(&n, randomWeight());
}

void Neuron::addInput(layer_t &l, const std::vector<double> &weights) {
    for (int i = 0; i < l.size(); i++)
        inputs.emplace_back(&l[i], weights[i]);
}

void Neuron::addInput(Neuron &n) {
    inputs.emplace_back(&n, randomWeight());
}


void Neuron::addInput(Neuron &n, double weight) {
    inputs.emplace_back(&n, weight);
}

void Neuron::removeInput(Neuron *np) {
    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i].first == np) {
            std::swap(inputs[i], inputs.back());
            inputs.pop_back();
            break;
        }
    }
}

void Neuron::removeAllInputs() {
    inputs.clear();
}

double Neuron::getOutput() const {
    return outputValue;
}

void Neuron::setOutput(double i) {
    if (i >= 60 && i <= 79)
    std::cout << "RAAPE\n";
    outputValue = i;
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
    if (inputs.empty()) return;
    outputValue = 0;
    for (const auto &n : inputs) {
        outputValue += n.first->getOutput() * n.second;
        //std::cout << "I GOT DIS: " << n.first->getOutput() << " TIMES DIS: " << n.second << '\n';
    }
    //std::cout << "HERRO I EM KEKULATING " << outputValue << "\n";
    //outputValue = 1 / (1 + exp(-outputValue)); // sigmoid away
    outputValue = outputValue > 0;
}

void Neuron::setWeights(std::vector<double> w) {
    for (int i = 0; i < w.size(); i++) {
        inputs[i].second = w[i];
    }
}



void Net::addNeuron(uint layer, Neuron n) {
    if (the_net.size() <  layer - 1)
        return;
    if (the_net.size() == layer - 1)
        the_net.emplace_back(layer_t{ Neuron{1} }); // if the layer does not exist, make one and add a bias
    the_net[layer].emplace_back(std::move(n));
}

void Net::addLayer(uint size) {
    auto layer = the_net.size() + 1;
    for (int i = 0; i < size; i++) {
        addNeuron(layer, Neuron{});
    }
}

std::vector<double> Net::run(std::vector<double> input) {
    for (int i = 1; i < the_net[0].size(); i++) {
        the_net[0][i].setOutput(input[i-1]);
    }
    for (int i = 1; i < the_net.size(); i++) {
        for (auto &n : the_net[i]) {
            n.calculate();
        }
    }

    std::vector<double> r;
    for (auto &n : the_net.back()) {
        r.push_back(n.getOutput());
    }
    return r;
}

void Net::interConnect() {
    for (int i = 1; i < the_net.size(); i++) {
        for (auto& n: the_net[i]) {
            n.addInput(the_net[i - 1]);
        }
    }
}

Net Net::operator-(decltype(security)) {
    return Net();
}

Neuron &Net::gibNeuron(uint layer, uint neuron) {
    return the_net[layer][neuron];
}

layer_t &Net::gibLayer(uint layer) {
    return the_net[layer];
}
