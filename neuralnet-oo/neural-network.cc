#include "neural-network.hh"

#include <utility>
#include <cstdlib>

double randomWeight() {
    double random = (double) rand() / (double) RAND_MAX;
    double r = random * 2;
    return r - 1;
}

void Neuron::addInput(layer_t &l) {
    for (auto& n : l)
        inputs.push_back(input_t(&n, randomWeight()));
}

void Neuron::addInput(Neuron &n) {
    inputs.push_back(input_t(&n, randomWeight()));
}

void Neuron::removeInput(Neuron *np) {
    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i].first == np) {
            std::swap(inputs[i], inputs.back());
            inputs.pop_back();
        }
    }
}

void Neuron::removeAllInputs() {
    inputs.clear();
}

double Neuron::getOutput() {
    return outputValue;
}

void Neuron::setOutput(double i) {
    outputValue = i;
}

void Neuron::setWeight(Neuron *np, double weight) {
    for (auto& n : inputs) {
        if (n.first == np) {
            n.second = weight;
        }
    }
}

void Neuron::calculate() {
    outputValue = 0;
    for (const auto&n : inputs)
        outputValue += n.first->getOutput() * n.second;

}

void Net::addNeuron(uint layer, Neuron n) {
    if (the_net.size() < layer)
        the_net.reserve(layer);

    the_net[layer].push_back(std::move(n));
}

void Net::addLayer(uint size) {
    
}

std::vector<double> Net::run(std::vector<double> input) {
    for (int i = 0; i < the_net[0].size(); i++) {
        the_net[0][i].setOutput(input[i]);
    }
    for (int i = 1; i < the_net.size(); i++) {
        for (auto& n : the_net[i]) {
            n.calculate();
        }
    }

    std::vector<double> r;
    for (auto& n : the_net.back()) {
        r.push_back(n.getOutput());
    }
    return r;
}
